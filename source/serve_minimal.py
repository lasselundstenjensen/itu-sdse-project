"""
Production model server - this is what runs inside the Docker image on
DigitalOcean App Platform.

Run it from the project root (the folder that contains source/ and models/):

    uvicorn source.serve:app --host 0.0.0.0 --port 5001

Endpoints (the same ones deployment.py serves locally, and the ones the
ingress rules in infra/digitalocean/app.yaml route to):

    GET  /health        200 once the model is loaded
    GET  /version       model version + training time (models/model/version.json)
    POST /invocations   predict, body: {"instances": ["<base64 image>", ...]}
    POST /infer         same as /invocations
    GET  /drift?days=7  drift check over the requests served so far

Differences from deployment.py:
- deployment.py starts `mlflow models serve` as a second process and proxies
  requests to it. Here the model is loaded once, in this process, so the
  container runs a single server and needs no MLflow tracking server.
- Nothing here trains, promotes or exports - that already happened in the
  pipeline. The image only contains the exported model (models/model) and
  the drift baseline (artifacts/drift_baseline_stats.json).
"""

from contextlib import asynccontextmanager
import threading

from fastapi import Body, FastAPI, HTTPException
import mlflow.pyfunc

# Reuse the drift + version logic from deployment.py so there is one source
# of truth for it. Importing deployment.py only defines functions/config;
# its training-promotion code is under `if __name__ == "__main__"`.
from source.deployment import drift, export_dir, version_endpoint

# The SimpleLoggingWrapper (see training.py) appends every prediction to
# artifacts/inference_log.json by reading the whole file and rewriting it.
# Two requests doing that at the same time would corrupt the file, so
# predictions are run one at a time.
_predict_lock = threading.Lock()
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, before the server accepts requests.

    If loading fails the exception stops the server, the container exits and
    App Platform shows the error in the deploy logs (instead of a server that
    is "up" but can't predict).
    """
    global _model
    print(f"Loading model from {export_dir} ...")
    pyfunc_model = mlflow.pyfunc.load_model(export_dir)
    # unwrap_python_model() gives us the SimpleLoggingWrapper instance itself.
    # Its predict() already parses {"instances": [...]} and base64 images, so
    # we call it directly instead of going through MLflow's generic
    # (tabular/DataFrame-oriented) input conversion.
    _model = pyfunc_model.unwrap_python_model()
    print("Model loaded, ready to serve")
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, openapi_url=None, redoc_url=None)


@app.get("/health")
def health():
    # Requests are only accepted after lifespan() finished loading the model,
    # so reaching this function means the model is ready.
    return {"status": "ok"}


@app.post("/invocations")
@app.post("/infer")
def invocations(payload: dict = Body(...)):
    # Accept MLflow's "inputs" key as well as "instances"/"data"
    if "inputs" in payload and "instances" not in payload:
        payload = {"instances": payload["inputs"]}

    instances = payload.get("instances", payload.get("data"))
    if not isinstance(instances, list) or not instances:
        raise HTTPException(
            status_code=400,
            detail='Expected a JSON body like {"instances": ["<base64-encoded image>", ...]}',
        )

    with _predict_lock:
        result = _model.predict(None, {"instances": instances})

    # Same response shape as MLflow's scoring server (what deployment.py's
    # /infer returned), so clients don't notice the difference.
    return {"predictions": result}


# GET /version and GET /drift are the exact functions from deployment.py
app.get("/version")(version_endpoint)
app.get("/drift")(drift)
