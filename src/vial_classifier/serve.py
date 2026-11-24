"""Serving: FastAPI front for the MLflow scoring server.

The MLflow pyfunc model runs as a subprocess on an internal port; this app
forwards to it and adds /health, /version and /drift.

    GET  /health       {"status": "ok"} | "loading" (503) | "error" (503)
    POST /invocations  forwarded unchanged to the scoring server
    POST /infer        alias of /invocations
    GET  /version      model version and training timestamp
    GET  /drift        drift check over the logged requests

CLI: ``python -m vial_classifier.serve``
"""

import atexit
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query, Request, Response

from vial_classifier import config, drift

app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

_scoring_proc = None


def start_scoring_server():
    """Start `mlflow models serve` for the exported model on the internal port."""
    env = os.environ.copy()
    # The scoring server execs `uvicorn` through PATH; make sure the
    # interpreter's bin directory is on it
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    proc = subprocess.Popen([
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", config.EXPORT_DIR,
        "-p", str(config.UPSTREAM_PORT),
        "--workers", "1",
        "--host", config.UPSTREAM_HOST,
        "--no-conda",
    ], env=env)
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    print("Scoring server starting on port", config.UPSTREAM_PORT)
    return proc


def stop_scoring_server(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10)
        print("Scoring server stopped")


def _upstream_health():
    """Return the scoring server's /health status code, None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{config.UPSTREAM_HOST}:{config.UPSTREAM_PORT}/health",
            timeout=1.0,
        ) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


@app.get("/health")
def health():
    if _scoring_proc is not None and _scoring_proc.poll() is not None:
        return Response(content='{"status": "error"}', status_code=503, media_type="application/json")
    if _upstream_health() == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(content='{"status": "loading"}', status_code=503, media_type="application/json")


@app.get("/version")
def version():
    """Model version of the served model (watched by the monitoring platform)."""
    version_file = Path(config.EXPORT_DIR) / "version.json"
    if version_file.exists():
        with open(version_file) as f:
            return json.load(f)
    return {"model_version": "unknown", "trained_at": None}


@app.post("/invocations")
@app.post("/infer")
async def invocations(request: Request):
    body = await request.body()
    req = urllib.request.Request(
        f"http://{config.UPSTREAM_HOST}:{config.UPSTREAM_PORT}/invocations",
        data=body,
        headers={"Content-Type": request.headers.get("content-type", "application/json")},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return Response(
                content=resp.read(),
                status_code=resp.status,
                media_type=resp.headers.get("content-type", "application/json"),
            )
    except urllib.error.HTTPError as e:
        return Response(
            content=e.read(),
            status_code=e.code,
            media_type=e.headers.get("content-type", "application/json"),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return Response(
            content=json.dumps({"status": "error", "detail": str(e)}),
            status_code=502,
            media_type="application/json",
        )


@app.get("/drift")
def drift_report(days: int = Query(default=config.DRIFT_DEFAULT_DAYS, ge=1)):
    """Run drift detection over the requests logged by the scoring server."""
    logs = drift.load_inference_logs()
    baseline = drift.load_baseline_stats()
    return drift.detect_drift(logs, days=days, baseline_stats=baseline)


def main():
    global _scoring_proc
    _scoring_proc = start_scoring_server()
    try:
        uvicorn.run(app, host="0.0.0.0", port=config.SERVE_PORT, log_level="info")
    finally:
        stop_scoring_server(_scoring_proc)


if __name__ == "__main__":
    main()