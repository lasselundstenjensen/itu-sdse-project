import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Where we write one entry per prediction (drift detection reads this later)
INFERENCE_LOG_PATH = Path(os.getenv("INFERENCE_LOG_PATH", "artifacts/inference_log.json"))


class InferRequest(BaseModel):
    image: str


def log_inference(prediction: int, confidence: float) -> None:
    INFERENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        logs = json.loads(INFERENCE_LOG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []
    logs.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": prediction,
            "confidence": confidence,
        }
    )
    INFERENCE_LOG_PATH.write_text(json.dumps(logs))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": "0.1.0", "model": "dummy"}


@app.post("/invocations")
@app.post("/infer")
def infer(request: InferRequest):
    try:
        base64.b64decode(request.image, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="image must be base64")

    # dummy model: always "clean" (0) with confidence 0.5
    prediction, confidence = 0, 0.5
    log_inference(prediction, confidence)
    label = "defective" if prediction == 1 else "clean"
    return {"prediction": label, "probability": confidence}