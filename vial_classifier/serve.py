"""Serve the model: a public FastAPI app in front of MLflow's scoring server.

Run from the project root:

    python -m vial_classifier.serve
"""

import atexit
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

from fastapi import FastAPI, Query, Request, Response
import uvicorn

from vial_classifier import config
from vial_classifier.drift import detect_drift, load_baseline_stats, load_inference_logs

app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

# The scoring server process. It is set by start_scoring_server().
scoring_proc = None


def start_scoring_server():
    """Start MLflow's model server as a background process on the internal port."""
    global scoring_proc
    env = os.environ.copy()
    # The scoring server runs `uvicorn` through PATH, so make sure the
    # interpreter's bin directory is on it.
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    scoring_proc = subprocess.Popen(
        [
            sys.executable, "-m", "mlflow", "models", "serve",
            "-m", str(config.EXPORT_DIR),
            "-p", str(config.UPSTREAM_PORT),
            "--workers", "1",
            "--host", config.UPSTREAM_HOST,
            "--no-conda",
        ],
        env=env,
    )
    atexit.register(lambda: scoring_proc.poll() is None and scoring_proc.terminate())


def _upstream_health():
    """Return the scoring server's /health status code, or None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{config.UPSTREAM_HOST}:{config.UPSTREAM_PORT}/health", timeout=1.0
        ) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


@app.get("/health")
def health():
    if scoring_proc is not None and scoring_proc.poll() is not None:
        return Response(content='{"status": "error"}', status_code=503, media_type="application/json")
    if _upstream_health() == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(content='{"status": "loading"}', status_code=503, media_type="application/json")


@app.get("/version")
def version():
    """Model version of the served model (watched by the monitoring platform)."""
    version_file = config.EXPORT_DIR / "version.json"
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
            content=f'{{"status": "error", "detail": "{e}"}}',
            status_code=502,
            media_type="application/json",
        )


@app.get("/drift")
def drift(days: int = Query(default=config.DRIFT_DEFAULT_DAYS, ge=1)):
    """Run drift detection over the requests logged by the scoring server."""
    logs = load_inference_logs(config.INFERENCE_LOG_PATH)
    baseline = load_baseline_stats()
    return detect_drift(logs, days=days, baseline_stats=baseline)


def main():
    start_scoring_server()
    uvicorn.run(app, host="0.0.0.0", port=config.API_PORT)


if __name__ == "__main__":
    main()