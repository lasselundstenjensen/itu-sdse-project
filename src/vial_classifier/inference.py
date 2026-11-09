"""MLflow pyfunc wrapper that turns the CNN into a servable model.

The wrapper accepts an image as a file path, raw bytes or a base64 string,
applies the evaluation transform, returns predictions and confidences, and
appends every request to a JSON log that the drift check reads later.

Note: the wrapper is pickled by MLflow with a reference to this module, so
``vial_classifier`` must be importable wherever the model is loaded.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import torch
from PIL import Image

from . import config
from .data import build_eval_transform
from .model import ARCHITECTURE_NAME, SimpleCNN

logger = logging.getLogger(__name__)


def create_model_from_checkpoint(checkpoint: dict) -> SimpleCNN:
    architecture = checkpoint.get("architecture", ARCHITECTURE_NAME)
    if architecture != ARCHITECTURE_NAME:
        raise ValueError(f"Unknown architecture: {architecture}")
    model = SimpleCNN()
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return model


class SimpleLoggingWrapper(mlflow.pyfunc.PythonModel):
    """Predicts vial quality and logs each request to ``log_path``."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        model_path: str | Path | None = None,
        log_path: str | Path = config.INFERENCE_LOG_PATH,
        threshold: float = config.THRESHOLD,
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.log_path = Path(log_path)
        self.threshold = threshold
        self.transform = build_eval_transform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def get_raw_model(self) -> torch.nn.Module | None:
        return self.model

    # -- MLflow hooks ------------------------------------------------------
    def load_context(self, context) -> None:
        """Called by MLflow when loading. No-op if the model was pickled with weights."""
        if self.model is None:
            if self.model_path and Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model = create_model_from_checkpoint(checkpoint)
            else:
                raise FileNotFoundError("Could not find model file")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context, model_input):
        if self.model is None:
            self.load_context(context)

        image_inputs = self._extract_inputs(model_input)
        timestamp = datetime.now(timezone.utc).isoformat()

        predictions, confidences, raw_outputs = [], [], []
        for image_input in image_inputs:
            try:
                tensor = self.preprocess_image(image_input).to(self.device)
                with torch.no_grad():
                    prob = torch.sigmoid(self.model(tensor)).item()
                pred = 1 if prob >= self.threshold else 0

                predictions.append(int(pred))
                confidences.append(round(float(prob), 4))
                raw_outputs.append(round(float(prob), 4))

                self._log_request({
                    "timestamp": timestamp,
                    "image_path": "<bytes>" if isinstance(image_input, bytes) else str(image_input),
                    "prediction": int(pred),
                    "confidence": float(prob),
                    "raw_output": float(prob),
                })
            except Exception as exc:  # noqa: BLE001 - one bad image must not fail the batch
                logger.warning("Error processing image %s: %s", str(image_input)[:50], exc)
                predictions.append(None)
                confidences.append(None)
                raw_outputs.append(None)

        return {
            "predictions": predictions,
            "confidences": confidences,
            "raw_outputs": raw_outputs,
            "num_samples": len(image_inputs),
        }

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _extract_inputs(model_input) -> list:
        """Normalise the many accepted request shapes into a flat list of images."""
        if isinstance(model_input, dict):
            if "instances" in model_input:
                items = model_input["instances"]
            elif "data" in model_input:
                items = model_input["data"]
            else:
                for value in model_input.values():
                    if isinstance(value, (list, str, np.ndarray, bytes, Path)):
                        items = [value] if isinstance(value, (str, Path, np.ndarray, bytes)) else value
                        break
                else:
                    raise ValueError(f"Could not extract image inputs from input: {model_input}")
        elif isinstance(model_input, (list, tuple, np.ndarray)):
            items = model_input
        else:
            items = [model_input]

        if isinstance(items, np.ndarray):
            items = items.tolist()

        flat: list = []
        for item in items:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    def preprocess_image(self, image_input) -> torch.Tensor:
        """Return a ``(1, 3, H, W)`` tensor for one image given as path, bytes or base64."""
        if isinstance(image_input, bytes):
            image = Image.open(BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            image = self._open_path_or_base64(image_input)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}. Expected str, Path, or bytes.")
        return self.transform(image).unsqueeze(0)

    @staticmethod
    def _open_path_or_base64(value: str | Path) -> Image.Image:
        candidate = None
        try:
            path = Path(value)
            if path.exists():
                candidate = path
        except OSError:
            candidate = None

        if candidate is not None:
            return Image.open(candidate).convert("RGB")

        try:
            return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Cannot process string input '{str(value)[:50]}...': "
                f"not an existing file path and not valid base64 image data: {exc}"
            ) from exc

    def _log_request(self, entry: dict) -> None:
        """Append one entry to the JSON log. Logging failures never fail a prediction."""
        try:
            logs: list = []
            if self.log_path.exists():
                try:
                    loaded = json.loads(self.log_path.read_text())
                    logs = loaded if isinstance(loaded, list) else []
                except json.JSONDecodeError:
                    logs = []
            logs.append(entry)
            self.log_path.write_text(json.dumps(logs, indent=2))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log request: %s", exc)
