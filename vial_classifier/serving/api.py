"""FastAPI app transferred from Notebook 2."""

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import uvicorn
from fastapi import FastAPI, Query, Request, Response

from vial_classifier.config import (
    API_HOST,
    API_PORT,
    DRIFT_DEFAULT_DAYS,
    MODEL_EXPORT_DIR,
    UPSTREAM_HOST,
    UPSTREAM_PORT,
)
from vial_classifier.serving.drift import (
    detect_drift,
    load_baseline_stats,
    load_inference_logs,
)

model_path = Path(MODEL_EXPORT_DIR)
port = API_PORT
upstream_port = UPSTREAM_PORT
upstream_host = UPSTREAM_HOST
drift_default_days = DRIFT_DEFAULT_DAYS

scoring_proc = None

app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)


def _start_scoring_server():
    """Start MLflow's scoring server as a subprocess."""
    env = os.environ.copy()
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "serve",
            "-m",
            str(model_path),
            "-p",
            str(upstream_port),
            "--workers",
            "1",
            "--host",
            upstream_host,
            "--no-conda",
        ],
        env=env,
    )
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc


def _upstream_health():
    """Return the scoring server's health status, or None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{upstream_host}:{upstream_port}/health",
            timeout=1.0,
        ) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


@app.on_event("startup")
def startup():
    """Start the scoring subprocess when the API server starts."""
    global scoring_proc
    scoring_proc = _start_scoring_server()


@app.on_event("shutdown")
def shutdown():
    """Stop the API thread and scoring subprocess when the server stops."""
    api_server = getattr(app.state, "api_server", None)
    if api_server is not None:
        api_server.should_exit = True

    if scoring_proc is not None and scoring_proc.poll() is None:
        scoring_proc.terminate()
        scoring_proc.wait(timeout=10)


@app.get("/health")
def health():
    if scoring_proc is not None and scoring_proc.poll() is not None:
        return Response(
            content='{"status": "error"}',
            status_code=503,
            media_type="application/json",
        )

    if _upstream_health() == 200:
        return Response(
            content='{"status": "ok"}',
            media_type="application/json",
        )

    return Response(
        content='{"status": "loading"}',
        status_code=503,
        media_type="application/json",
    )


@app.get("/version")
def version():
    """Return the exported model's version metadata."""
    version_file = model_path / "version.json"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as version_file_handle:
            return json.load(version_file_handle)
    return {"model_version": "unknown", "trained_at": None}


@app.post("/invocations")
@app.post("/infer")
async def invocations(request: Request):
    """Forward the request body unchanged to MLflow's scoring server."""
    body = await request.body()
    upstream_request = urllib.request.Request(
        f"http://{upstream_host}:{upstream_port}/invocations",
        data=body,
        headers={
            "Content-Type": request.headers.get(
                "content-type",
                "application/json",
            )
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(upstream_request, timeout=300) as response:
            return Response(
                content=response.read(),
                status_code=response.status,
                media_type=response.headers.get(
                    "content-type",
                    "application/json",
                ),
            )
    except urllib.error.HTTPError as error:
        return Response(
            content=error.read(),
            status_code=error.code,
            media_type=error.headers.get("content-type", "application/json"),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return Response(
            content=f'{{"status": "error", "detail": "{error}"}}',
            status_code=502,
            media_type="application/json",
        )


@app.get("/drift")
def drift(days: int = Query(default=drift_default_days, ge=1)):
    """Return drift results for recent inference requests."""
    logs = load_inference_logs()
    baseline = load_baseline_stats()
    return detect_drift(logs, days=days, baseline_stats=baseline)


def run():
    """Start the FastAPI server, corresponding to Notebook 2's API thread."""
    config = uvicorn.Config(
        app,
        host=API_HOST,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    app.state.api_server = server

    api_thread = threading.Thread(target=server.run, daemon=True)
    api_thread.start()
    return server, api_thread


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=port, log_level="info")