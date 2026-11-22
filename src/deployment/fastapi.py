from fastapi import FastAPI, Query, Request, Response
import urllib.error
import urllib.request
from pathlib import Path
import json
from src.deployment.serve_model_locally import start_scoring_server
from src.deployment.drift_detection import load_baseline_stats, detect_drift, load_inference_logs

port = 5001
upstream_port = 5002
upstream_host = "127.0.0.1"
artifact_dir = "artifacts"
inference_log_path = artifact_dir + "/inference_log.json"
drift_default_days = 7
export_dir = "models/model"
model_path = export_dir


app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

#scoring_proc = start_scoring_server()



def _upstream_health():
    """Return the scoring server's /health status code, None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{upstream_host}:{upstream_port}/health",
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
        return Response(content='{"status": "error"}', status_code=503, media_type="application/json")
    if _upstream_health() == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(content='{"status": "loading"}', status_code=503, media_type="application/json")

@app.get("/version")
def version():
    """Model version of the served model (watched by the monitoring platform)."""
    version_file = Path(model_path) / "version.json"
    if version_file.exists():
        with open(version_file) as f:
            return json.load(f)
    return {"model_version": "unknown", "trained_at": None}

@app.post("/invocations")
@app.post("/infer")
async def invocations(request: Request):
    body = await request.body()
    req = urllib.request.Request(
        f"http://{upstream_host}:{upstream_port}/invocations",
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
def drift(days: int = Query(default=drift_default_days, ge=1)):
    """Run drift detection over the requests logged by the scoring server."""
    logs = load_inference_logs(inference_log_path)
    try:
        baseline = load_baseline_stats()
    except (FileNotFoundError, ValueError) as e:
        print(f"Failed to load the baseline stats: {e}")
        return Response(content=json.dumps({"status": "error", "detail": str(e)}), status_code = 503, media_type="application/json",)
    return detect_drift(logs, days=days, baseline_stats=baseline)