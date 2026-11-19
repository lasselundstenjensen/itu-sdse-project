import base64

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class InferRequest(BaseModel):
    image: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": "0.1.0", "model": "dummy"}


@app.post("/infer")
def infer(request: InferRequest):
    try:
        base64.b64decode(request.image, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="image must be base64")
    return {"prediction": "clean", "probability": 0.5}