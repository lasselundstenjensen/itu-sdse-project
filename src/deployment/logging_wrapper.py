"""
Logging Wrapper for MLflow Model Serving

Provides a minimal PyTorch model wrapper that logs inference requests to a JSON file.
This is a simplified version that separates logging from drift detection.
"""

import base64
import json
import os
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.pyfunc
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from ..config import (
    IMAGE_SIZE,
    THRESHOLD,
    TRANSFORM_MEAN,
    TRANSFORM_STD,
    INFERENCE_LOG_PATH,
    MLFLOW_TRACKING_URI,
)


class SimpleLoggingWrapper(mlflow.pyfunc.PythonModel):
    """
    Simple wrapper that logs inference requests and returns predictions.
    
    This is a minimal wrapper that:
    1. Loads a PyTorch model
    2. Logs each inference request to a JSON file
    3. Returns predictions
    
    No drift detection is built-in - that's handled by a separate CLI command.
    """
    
    def get_raw_model(self):
        """Return the underlying PyTorch model for direct access."""
        return self.model
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_path: Optional[Path] = None,
        log_path: Path = INFERENCE_LOG_PATH,
        threshold: float = THRESHOLD,
        mlflow_run_id: Optional[str] = None,
    ):
        """
        Initialize the logging wrapper.
        
        Parameters
        ----------
        model : nn.Module, optional
            PyTorch model to wrap. If None, will be loaded from model_path.
        model_path : Path, optional
            Path to saved model file.
        log_path : Path, default=INFERENCE_LOG_PATH from config
            Path to the JSON log file.
        threshold : float, default=THRESHOLD from config
            Classification threshold.
        mlflow_run_id : str, optional
            MLflow run ID to log metrics to. If provided, inference metrics
            (latency, predictions, confidence) will be logged to this run.
        """
        self.model = model
        self.model_path = model_path
        self.log_path = log_path
        self.threshold = threshold
        self.mlflow_run_id = mlflow_run_id
        self.metrics_step = 0
        
        # Transform for preprocessing images
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
        ])
        
        # Device for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_context(self, context: Any) -> None:
        """
        Load the model from the MLflow context or model_path.
        
        This method is automatically called by MLflow when the model is loaded.
        
        Parameters
        ----------
        context : Any
            MLflow context containing paths to model artifacts.
        """
        print("Loading SimpleLoggingWrapper context...")
        
        # Load model if not already loaded
        if self.model is None:
            if self.model_path and self.model_path.exists():
                # Load from local path
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model = self._create_model_from_checkpoint(checkpoint)
                self.model.to(self.device)
                self.model.eval()
                print(f"Loaded model from {self.model_path}")
            else:
                # Try to find model in artifacts
                artifacts_dir = Path(context.artifacts)
                possible_paths = [
                    artifacts_dir / "vial_cnn_model.pth",
                    Path("artifacts/vial_cnn_model.pth"),
                ]
                for path in possible_paths:
                    if path.exists():
                        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                        self.model = self._create_model_from_checkpoint(checkpoint)
                        self.model.to(self.device)
                        self.model.eval()
                        print(f"Loaded model from {path}")
                        break
                else:
                    raise FileNotFoundError("Could not find model file. Please ensure the model is in artifacts/")
        
        print("SimpleLoggingWrapper context loaded successfully!")
    
    def _create_model_from_checkpoint(self, checkpoint: Dict) -> nn.Module:
        """
        Create a model instance from a saved checkpoint.
        
        Parameters
        ----------
        checkpoint : Dict
            Dictionary containing model state and metadata.
            
        Returns
        -------
        nn.Module
            Recreated model instance.
        """
        # Import the model class dynamically
        from ..models.train import SimpleCNN
        
        architecture = checkpoint.get('architecture', 'SimpleCNN')
        
        if architecture == 'SimpleCNN':
            model = SimpleCNN()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _get_timestamp_iso(self) -> str:
        """Get current timestamp in ISO format with timezone."""
        return datetime.now(timezone.utc).isoformat()
    
    def _log_request(self, request_data: Dict[str, Any]) -> None:
        """
        Append a request log entry to the JSON log file.
        
        Parameters
        ----------
        request_data : Dict[str, Any]
            Dictionary containing request information to log.
        """
        try:
            # Read existing logs
            logs = []
            if self.log_path.exists():
                with open(self.log_path, 'r') as f:
                    try:
                        logs = json.load(f)
                        if not isinstance(logs, list):
                            logs = []
                    except json.JSONDecodeError:
                        logs = []
            
            # Append new log entry
            logs.append(request_data)
            
            # Write back to file
            with open(self.log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
        except Exception as e:
            print(f"Warning: Failed to log request: {e}")
    
    def _log_to_mlflow(self, metrics: Dict[str, float]) -> None:
        """
        Log metrics to MLflow tracking for the model's run.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of metrics to log (e.g., latency, prediction counts).
        """
        if not self.mlflow_run_id:
            return
        
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            with mlflow.start_run(run_id=self.mlflow_run_id):
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=self.metrics_step)
            self.metrics_step += 1
        except Exception as e:
            print(f"Warning: Failed to log metrics to MLflow: {e}")
    
    def preprocess_image(self, image_input: Union[str, Path, bytes]) -> torch.Tensor:
        """
        Preprocess a single image for inference.
        
        Parameters
        ----------
        image_input : Union[str, Path, bytes]
            Either a path to the image file (str or Path), raw image bytes,
            or a base64-encoded string.
            
        Returns
        -------
        torch.Tensor
            Preprocessed image tensor of shape (1, C, H, W).
        """
        # Handle bytes input
        if isinstance(image_input, bytes):
            image = Image.open(BytesIO(image_input)).convert('RGB')
        # Handle string input - check for an existing file path first (cheap and
        # unambiguous), then fall back to base64. Checking the path first avoids
        # b64decode silently mangling paths like "/tmp/Clean_001.jpg" into garbage
        # bytes that Image.open then rejects with an error we would otherwise miss.
        # NOTE: a base64 blob can be long enough that Path(...).exists() raises
        # OSError ("File name too long") on some platforms (Python 3.13 / macOS),
        # so that case must fall through to base64 rather than propagate.
        elif isinstance(image_input, str):
            path = None
            try:
                candidate = Path(image_input)
                if candidate.exists():
                    path = candidate
            except OSError:
                path = None
            if path is not None:
                image = Image.open(path).convert('RGB')
            else:
                # Treat as base64-encoded image data (the portable contract for
                # containerized serving). Catch broadly: b64decode(validate=False)
                # rarely raises, so the real failure surfaces in Image.open as
                # UnidentifiedImageError, which is not a binascii/ValueError.
                try:
                    decoded_bytes = base64.b64decode(image_input)
                    image = Image.open(BytesIO(decoded_bytes)).convert('RGB')
                except Exception as e:
                    raise ValueError(
                        f"Cannot process string input '{image_input[:50]}...': "
                        f"not an existing file path and not valid base64 image data: {e}"
                    )
        # Handle Path input
        elif isinstance(image_input, Path):
            if image_input.exists():
                image = Image.open(image_input).convert('RGB')
            else:
                raise FileNotFoundError(f"File not found: {image_input}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}. Expected str, Path, or bytes.")
        
        return self.transform(image).unsqueeze(0)
    
    def predict(self, context, model_input):
        """
        Make predictions on input data and log the request.
        
        This method is automatically called by MLflow when predictions are requested.
        
        Parameters
        ----------
        context : Any
            MLflow context (currently unused but required by interface).
        model_input : Any
            Input data. Can be:
            - Path to a single image (str or Path)
            - Raw image bytes
            - List of paths or bytes
            - Dictionary with 'data' or 'instances' key containing image paths or bytes
            
        Returns
        -------
        Dict
            Prediction results with:
            - predictions: List of class predictions (0 or 1)
            - confidences: List of raw probability scores
            - raw_outputs: List of raw model outputs
            - timestamp: ISO timestamp of the request
            - num_samples: Number of images processed
        """
        start_time = time.time()
        
        if self.model is None:
            self.load_context(context)
        
        # Handle different input formats
        if isinstance(model_input, dict):
            if 'instances' in model_input:
                image_inputs = model_input['instances']
            elif 'data' in model_input:
                image_inputs = model_input['data']
            else:
                # Try to extract inputs from any key
                for key, value in model_input.items():
                    if isinstance(value, (list, str, np.ndarray, bytes)) or isinstance(value, Path):
                        image_inputs = [value] if isinstance(value, (str, Path, np.ndarray, bytes)) else value
                        break
                else:
                    raise ValueError(f"Could not extract image inputs from input: {model_input}")
        elif isinstance(model_input, (list, tuple, np.ndarray)):
            image_inputs = model_input
        else:
            image_inputs = [model_input]
        
        # Convert numpy arrays to lists
        if isinstance(image_inputs, np.ndarray):
            image_inputs = image_inputs.tolist()
        
        # Convert to list of inputs, handling various types
        image_inputs_list = []
        for item in image_inputs:
            # Already a list/tuple - flatten
            if isinstance(item, (list, tuple)):
                image_inputs_list.extend(item)
            elif isinstance(item, np.ndarray):
                # Handle numpy array (could be bytes or string)
                if item.dtype == object and len(item) == 1:
                    image_inputs_list.append(item[0])
                else:
                    image_inputs_list.append(item.tolist() if item.ndim > 0 else item)
            else:
                # bytes, str, Path, or other types
                image_inputs_list.append(item)
        image_inputs = image_inputs_list
        
        # Collect results
        predictions = []
        confidences = []
        raw_outputs = []
        log_entries = []
        
        timestamp = self._get_timestamp_iso()
        
        # Process each image input (can be path, bytes, etc.)
        for image_input in image_inputs:
            try:
                # Preprocess image (handles both paths and bytes)
                image_tensor = self.preprocess_image(image_input)
                image_tensor = image_tensor.to(self.device)
                
                # Get raw model output
                with torch.no_grad():
                    output = self.model(image_tensor)
                    prob = output.item()
                    pred = 1 if prob >= self.threshold else 0
                
                predictions.append(int(pred))
                confidences.append(float(prob))
                raw_outputs.append(float(prob))
                
                # Create log entry for this request
                # For bytes input, use a placeholder for the image identifier
                image_identifier = "<bytes>" if isinstance(image_input, bytes) else str(image_input)
                log_entry = {
                    'timestamp': timestamp,
                    'image_path': image_identifier,
                    'prediction': int(pred),
                    'confidence': float(prob),
                    'raw_output': float(prob),
                }
                log_entries.append(log_entry)
                
                # Log immediately (append to file)
                self._log_request(log_entry)
                
            except Exception as e:
                print(f"Error processing image {image_input}: {e}")
                predictions.append(None)
                confidences.append(None)
                raw_outputs.append(None)
                
                image_identifier = "<bytes>" if isinstance(image_input, bytes) else str(image_input)
                error_log = {
                    'timestamp': timestamp,
                    'image_path': image_identifier,
                    'prediction': None,
                    'confidence': None,
                    'raw_output': None,
                    'error': str(e),
                }
                log_entries.append(error_log)
                self._log_request(error_log)
        
        # Log metrics to MLflow
        if self.mlflow_run_id and predictions:
            latency = time.time() - start_time
            # Filter out None values for valid predictions
            valid_predictions = [p for p in predictions if p is not None]
            valid_confidences = [c for c in confidences if c is not None]
            
            metrics = {
                "inference_latency_ms": latency * 1000,
                "predictions_total": len(valid_predictions),
                "predictions_class_0": valid_predictions.count(0),
                "predictions_class_1": valid_predictions.count(1),
            }
            
            # Add average confidence if we have valid confidences
            if valid_confidences:
                metrics["avg_confidence"] = sum(valid_confidences) / len(valid_confidences)
            
            self._log_to_mlflow(metrics)
        
        # Return predictions with raw outputs
        return {
            'predictions': predictions,
            'confidences': confidences,
            'raw_outputs': raw_outputs,
            'timestamp': timestamp,
            'num_samples': len(image_inputs),
        }


if __name__ == "__main__":
    """
    Test the SimpleLoggingWrapper.
    
    Usage:
        python -m src.deployment.logging_wrapper
    """
    print("Testing SimpleLoggingWrapper...")
    
    try:
        # Create a logging wrapper (without actual model for now)
        wrapper = SimpleLoggingWrapper()
        print("SimpleLoggingWrapper created successfully!")
        print(f"Log path: {wrapper.log_path}")
        print(f"Device: {wrapper.device}")
        print(f"MLflow run ID: {wrapper.mlflow_run_id}")
        
        # Test with mlflow_run_id
        wrapper_with_run = SimpleLoggingWrapper(mlflow_run_id="test_run_123")
        print(f"\nWrapper with run_id created:")
        print(f"MLflow run ID: {wrapper_with_run.mlflow_run_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
