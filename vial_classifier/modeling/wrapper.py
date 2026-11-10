import base64
from datetime import UTC, datetime
from io import BytesIO
import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from vial_classifier import config
from vial_classifier.modeling.model import SimpleCNN


class SimpleLoggingWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper that logs inference requests and returns predictions."""

    def __init__(
        self,
        model=None,
        model_path=None,
        log_path=Path(config.INFERENCE_LOG_PATH),
        threshold=config.THRESHOLD,
    ):
        self.model = model
        self.model_path = model_path
        self.log_path = Path(log_path)
        self.threshold = threshold

        # Transform for preprocessing images (same as training, no augmentation)
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.TRANSFORM_MEAN, std=config.TRANSFORM_STD),
            ]
        )

        # Device for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def get_raw_model(self):
        """Return the underlying PyTorch model for direct access."""
        return self.model

    def load_context(self, context):
        """Called by MLflow when the model is loaded. Nothing to do when the
        model was pickled together with the wrapper."""
        print("Loading SimpleLoggingWrapper context...")
        if self.model is None:
            if self.model_path and Path(self.model_path).exists():
                checkpoint = torch.load(
                    self.model_path, map_location=self.device, weights_only=False
                )
                self.model = self._create_model_from_checkpoint(checkpoint)
            else:
                raise FileNotFoundError("Could not find model file")
        self.model.to(self.device)
        self.model.eval()
        print("SimpleLoggingWrapper context loaded successfully!")

    def _create_model_from_checkpoint(self, checkpoint):
        architecture = checkpoint.get("architecture", "SimpleCNN")
        if architecture == "SimpleCNN":
            model = SimpleCNN()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        return model

    def _get_timestamp_iso(self):
        return datetime.now(UTC).isoformat()

    def _log_request(self, request_data):
        """Append a request log entry to the JSON log file."""
        try:
            logs = []
            if self.log_path.exists():
                with open(self.log_path, "r") as f:
                    try:
                        logs = json.load(f)
                        if not isinstance(logs, list):
                            logs = []
                    except json.JSONDecodeError:
                        logs = []

            logs.append(request_data)

            with open(self.log_path, "w") as f:
                json.dump(logs, f, indent=2)
        except (OSError, ValueError) as e:
            print(f"Warning: Failed to log request: {e}")

    def preprocess_image(self, image_input):
        """Preprocess a single image: path, raw bytes or base64 string."""
        if isinstance(image_input, bytes):
            image = Image.open(BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            path = None
            try:
                candidate = Path(image_input)
                if candidate.exists():
                    path = candidate
            except OSError:
                path = None
            if path is not None:
                image = Image.open(path).convert("RGB")
            else:
                # Treat as base64-encoded image data (the portable contract
                # for containerized serving)
                try:
                    decoded_bytes = base64.b64decode(image_input)
                    image = Image.open(BytesIO(decoded_bytes)).convert("RGB")
                except (OSError, ValueError) as e:
                    raise ValueError(
                        f"Cannot process string input '{image_input[:50]}...': "
                        f"not an existing file path and not valid base64 image data: {e}"
                    )
        elif isinstance(image_input, Path):
            if image_input.exists():
                image = Image.open(image_input).convert("RGB")
            else:
                raise FileNotFoundError(f"File not found: {image_input}")
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}. Expected str, Path, or bytes."
            )

        return self.transform(image).unsqueeze(0)

    def predict(self, context, model_input):
        """Make predictions on input data and log the request."""
        if self.model is None:
            self.load_context(context)

        # Handle different input formats
        if isinstance(model_input, dict):
            if "instances" in model_input:
                image_inputs = model_input["instances"]
            elif "data" in model_input:
                image_inputs = model_input["data"]
            else:
                # Try to extract inputs from any key
                for value in model_input.values():
                    if isinstance(value, (list, str, np.ndarray, bytes, Path)):
                        image_inputs = (
                            [value] if isinstance(value, (str, Path, np.ndarray, bytes)) else value
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

        # Convert to flat list of inputs
        image_inputs_list = []
        for item in image_inputs:
            if isinstance(item, (list, tuple)):
                image_inputs_list.extend(item)
            else:
                image_inputs_list.append(item)
        image_inputs = image_inputs_list

        predictions = []
        confidences = []
        raw_outputs = []

        timestamp = self._get_timestamp_iso()

        for image_input in image_inputs:
            try:
                image_tensor = self.preprocess_image(image_input)
                image_tensor = image_tensor.to(self.device)

                with torch.no_grad():
                    logit = self.model(image_tensor)
                    prob = torch.sigmoid(logit).item()
                    pred = 1 if prob >= self.threshold else 0

                predictions.append(int(pred))
                confidences.append(round(float(prob), 4))
                raw_outputs.append(round(float(prob), 4))

                image_identifier = (
                    "<bytes>" if isinstance(image_input, bytes) else str(image_input)
                )
                log_entry = {
                    "timestamp": timestamp,
                    "image_path": image_identifier,
                    "prediction": int(pred),
                    "confidence": float(prob),
                    "raw_output": float(prob),
                }
                self._log_request(log_entry)

            except (OSError, ValueError) as e:
                print(f"Error processing image {image_input}: {e}")
                predictions.append(None)
                confidences.append(None)
                raw_outputs.append(None)

        return {
            "predictions": predictions,
            "confidences": confidences,
            "raw_outputs": raw_outputs,
            "num_samples": len(image_inputs),
        }
