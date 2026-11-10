"""FastAPI serving app (runs in the Docker image on port 5001).

The exported pyfunc model is served by the MLflow scoring server as a subprocess on
`UPSTREAM_PORT`, with this small FastAPI app in front of it:

- GET  /health      -> {"status": "ok"} once the model is loaded
- POST /invocations -> forwarded unchanged to the scoring server
- POST /infer       -> alias of /invocations
- GET  /version     -> model version + training timestamp from version.json
- GET  /drift       -> monitoring.drift.detect_drift over the inference log

We start the scoring server ourselves (what `mlflow models serve` does under the hood)
to control the response bodies: MLflow's own /health answers with a bare newline,
which JSON clients reject.

CLI: `python -m vial_classifier.serving.app`
"""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

from fastapi import FastAPI, Query, Request, Response
from loguru import logger
import typer
import uvicorn

from vial_classifier import config
from vial_classifier.monitoring.drift import detect_drift, load_baseline_stats, load_inference_logs

cli = typer.Typer()

scoring_proc = None


def start_scoring_server(model_path=config.EXPORT_DIR):
    """Start the MLflow scoring server for the exported model as a subprocess."""
    env = os.environ.copy()
    # The scoring server execs `uvicorn` through PATH; make sure the
    # interpreter's bin directory is on it
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "serve",
            "-m",
            str(model_path),
            "-p",
            str(config.UPSTREAM_PORT),
            "--workers",
            "1",
            "--host",
            config.UPSTREAM_HOST,
            "--no-conda",
        ],
        env=env,
    )


@asynccontextmanager
async def lifespan(app):
    global scoring_proc
    scoring_proc = start_scoring_server()
    logger.info(f"Scoring server starting on port {config.UPSTREAM_PORT}")
    try:
        yield
    finally:
        if scoring_proc is not None and scoring_proc.poll() is None:
            scoring_proc.terminate()
            scoring_proc.wait(timeout=10)
            logger.info("Scoring server stopped")


app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None, lifespan=lifespan)


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
    if scoring_proc is not None and scoring_proc.poll() is not None:
        return Response(
            content='{"status": "error"}', status_code=503, media_type="application/json"
        )
    if _upstream_health() == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(
        content='{"status": "loading"}', status_code=503, media_type="application/json"
    )


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
        with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: ASYNC210
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


@cli.command()
def main(host: str = "0.0.0.0", port: int = config.SERVING_PORT):
    """Run the serving app (the container's main process)."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    cli()
