"""HTTP serving: a FastAPI front end around the MLflow scoring server.

Process layout (same as the original notebook and the production container):

    client --> FastAPI (API_PORT) --> mlflow models serve subprocess (SCORING_PORT)

The front end owns the response contract (``/health`` returns JSON, which
MLflow's own health endpoint does not), adds ``/version`` and ``/drift``, and
forwards prediction requests unchanged.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from . import config
from .drift import detect_drift, load_baseline_stats, load_inference_logs

logger = logging.getLogger(__name__)

FORWARD_TIMEOUT_SECONDS = 300


def start_scoring_server(model_dir: Path, port: int = config.SCORING_PORT,
                         host: str = config.SCORING_HOST) -> subprocess.Popen:
    """Launch ``mlflow models serve`` for ``model_dir`` as a child process."""
    env = os.environ.copy()
    # The scoring server execs `uvicorn` via PATH, so put our interpreter's bin first
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlflow", "models", "serve",
         "-m", str(model_dir), "-p", str(port), "--workers", "1",
         "--host", host, "--no-conda"],
        env=env,
    )
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    logger.info("Scoring server starting on port %d", port)
    return proc


def create_app(
    model_dir: Path = config.EXPORT_MODEL_DIR,
    scoring_host: str = config.SCORING_HOST,
    scoring_port: int = config.SCORING_PORT,
    scoring_proc: subprocess.Popen | None = None,
    inference_log_path: Path = config.INFERENCE_LOG_PATH,
    baseline_path: Path = config.DRIFT_BASELINE_STATS_PATH,
) -> FastAPI:
    """Build the FastAPI application. ``scoring_proc`` is used to report a crashed scorer."""
    app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)
    upstream = f"http://{scoring_host}:{scoring_port}"

    def upstream_status() -> int | None:
        """HTTP status of the scorer's /health, or None if unreachable."""
        try:
            with urllib.request.urlopen(f"{upstream}/health", timeout=1.0) as resp:
                return resp.status
        except urllib.error.HTTPError as exc:
            return exc.code
        except (urllib.error.URLError, TimeoutError, OSError):
            return None

    def forward(body: bytes, content_type: str) -> Response:
        req = urllib.request.Request(
            f"{upstream}/invocations",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=FORWARD_TIMEOUT_SECONDS) as resp:
                return Response(content=resp.read(), status_code=resp.status,
                                media_type=resp.headers.get("content-type", "application/json"))
        except urllib.error.HTTPError as exc:
            return Response(content=exc.read(), status_code=exc.code,
                            media_type=exc.headers.get("content-type", "application/json"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=502)

    @app.get("/health")
    def health() -> JSONResponse:
        if scoring_proc is not None and scoring_proc.poll() is not None:
            return JSONResponse({"status": "error"}, status_code=503)
        if upstream_status() == 200:
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "loading"}, status_code=503)

    @app.get("/version")
    def version() -> dict:
        version_file = Path(model_dir) / "version.json"
        if version_file.exists():
            return json.loads(version_file.read_text())
        return {"model_version": "unknown", "trained_at": None}

    @app.post("/invocations")
    @app.post("/infer")
    async def invocations(request: Request) -> Response:
        body = await request.body()
        content_type = request.headers.get("content-type", "application/json")
        return await run_in_threadpool(forward, body, content_type)

    @app.get("/drift")
    def drift(days: int = Query(default=config.DRIFT_DEFAULT_DAYS, ge=1)) -> dict:
        logs = load_inference_logs(inference_log_path)
        baseline = load_baseline_stats(baseline_path)
        return detect_drift(logs, days=days, baseline_stats=baseline)

    return app


def run_server(model_dir: Path = config.EXPORT_MODEL_DIR, api_port: int = config.API_PORT) -> None:
    """Start the scorer and the front end, and block until the front end exits."""
    import uvicorn

    scoring_proc = start_scoring_server(model_dir)
    app = create_app(model_dir=model_dir, scoring_proc=scoring_proc)
    try:
        uvicorn.run(app, host="0.0.0.0", port=api_port, log_level="info")
    finally:
        if scoring_proc.poll() is None:
            scoring_proc.terminate()
            scoring_proc.wait(timeout=10)
