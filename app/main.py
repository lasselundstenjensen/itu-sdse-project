
from pathlib import Path
import sys

import torch
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vial_classifier.models.cnn import SimpleCNN

app = FastAPI(title="Vial Classification API")

CHECKPOINT_PATH = ROOT / "models" / "model" / "vial_cnn_model.pth"

# Load the trained CNN
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

model = SimpleCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "architecture": checkpoint["architecture"],
        "input_shape": list(checkpoint["input_shape"]),
        "num_classes": int(checkpoint["num_classes"]),
        "model_loaded": True
    }
