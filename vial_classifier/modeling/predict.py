"""MLflow pyfunc wrapper for vial image inference and request logging."""

import base64
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from vial_classifier.config import (
    IMAGE_SIZE,
    INFERENCE_LOG_PATH,
    TRANSFORM_MEAN,
    TRANSFORM_STD,
    THRESHOLD,
)
from vial_classifier.modeling.model import SimpleCNN


class SimpleLoggingWrapper(mlflow.pyfunc.PythonModel):
    """Load image inputs, return predictions, and append inference records."""

    def __init__(
        self,
        model=None,
        model_path=None,
        log_path=INFERENCE_LOG_PATH,
        threshold=THRESHOLD,
    ):
        self.model = model
        self.model_path = Path(model_path) if model_path is not None else None
        self.log_path = Path(log_path)
        self.threshold = threshold
        self.transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def get_raw_model(self):
        """Return the underlying PyTorch model."""
        return self.model

    def load_context(self, context):
        """Load the checkpoint when MLflow restores this pyfunc model."""
        if self.model is None:
            if self.model_path is None or not self.model_path.exists():
                raise FileNotFoundError("Could not find model file")
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,
            )
            self.model = self._create_model_from_checkpoint(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _create_model_from_checkpoint(checkpoint):
        architecture = checkpoint.get("architecture", "SimpleCNN")
        if architecture != "SimpleCNN":
            raise ValueError(f"Unknown architecture: {architecture}")
        model = SimpleCNN()
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _get_timestamp_iso():
        return datetime.now(timezone.utc).isoformat()

    def _log_request(self, request_data):
        """Append one request record to the inference log."""
        try:
            logs = []
            if self.log_path.exists():
                try:
                    logs = json.loads(self.log_path.read_text(encoding="utf-8"))
                    if not isinstance(logs, list):
                        logs = []
                except json.JSONDecodeError:
                    logs = []
            logs.append(request_data)
            self.log_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
        except Exception as error:
            print(f"Warning: Failed to log request: {error}")

    def preprocess_image(self, image_input):
        """Preprocess an image path, raw bytes, or base64 encoded string."""
        if isinstance(image_input, bytes):
            image = Image.open(BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            try:
                candidate = Path(image_input)
                exists = candidate.exists()
            except OSError:
                exists = False
            if exists:
                image = Image.open(candidate).convert("RGB")
            else:
                try:
                    decoded_bytes = base64.b64decode(image_input)
                    image = Image.open(BytesIO(decoded_bytes)).convert("RGB")
                except Exception as error:
                    raise ValueError(
                        "Input is neither an existing image path nor valid base64 image data: "
                        f"{error}"
                    ) from error
        elif isinstance(image_input, Path):
            if not image_input.exists():
                raise FileNotFoundError(f"File not found: {image_input}")
            image = Image.open(image_input).convert("RGB")
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}. "
                "Expected str, Path, or bytes."
            )
        return self.transform(image).unsqueeze(0)

    def predict(self, context, model_input):
        """Predict each supplied image and record its result."""
        if self.model is None:
            self.load_context(context)

        if isinstance(model_input, dict):
            if "instances" in model_input:
                image_inputs = model_input["instances"]
            elif "data" in model_input:
                image_inputs = model_input["data"]
            else:
                for value in model_input.values():
                    if isinstance(value, (list, str, np.ndarray, bytes, Path)):
                        image_inputs = (
                            [value]
                            if isinstance(value, (str, Path, np.ndarray, bytes))
                            else value
                        )
                        break
                else:
                    raise ValueError(f"Could not extract image inputs from input: {model_input}")
        elif isinstance(model_input, (list, tuple, np.ndarray)):
            image_inputs = model_input
        else:
            image_inputs = [model_input]

        if isinstance(image_inputs, np.ndarray):
            image_inputs = image_inputs.tolist()
        image_inputs = [
            nested
            for item in image_inputs
            for nested in (item if isinstance(item, (list, tuple)) else [item])
        ]

        predictions, confidences, raw_outputs = [], [], []
        timestamp = self._get_timestamp_iso()
        for image_input in image_inputs:
            try:
                image_tensor = self.preprocess_image(image_input).to(self.device)
                with torch.no_grad():
                    logit = self.model(image_tensor)
                    probability = torch.sigmoid(logit).item()
                prediction = int(probability >= self.threshold)
                predictions.append(prediction)
                confidences.append(round(float(probability), 4))
                raw_outputs.append(round(float(probability), 4))
                image_identifier = (
                    "<bytes>" if isinstance(image_input, bytes) else str(image_input)
                )
                self._log_request(
                    {
                        "timestamp": timestamp,
                        "image_path": image_identifier,
                        "prediction": prediction,
                        "confidence": float(probability),
                        "raw_output": float(probability),
                    }
                )
            except Exception as error:
                print(f"Error processing image {image_input}: {error}")
                predictions.append(None)
                confidences.append(None)
                raw_outputs.append(None)

        return {
            "predictions": predictions,
            "confidences": confidences,
            "raw_outputs": raw_outputs,
            "num_samples": len(image_inputs),
        }