# Imports
# ----------------

import os
import time
import torch
import json
import base64
import mlflow
import mlflow.pyfunc
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from pathlib import Path
from io import BytesIO
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timezone
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from pprint import pprint

from source_code.preprocessing import apply_preprocessing

from source_code.helper import (
    setup_mlflow_tracking,
    GlassVialDataset
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from source_code.config import (
    mlflow_state_dir,
    mlflow_tracking_uri,
    mlflow_artifact_location,
    experiment_name
)

from source_code.config import (
    num_epochs,
    learning_rate,
    random_state,
    scheduler_patience,
    early_stop_patience,
    threshold,
    num_classes,
    batch_size,
    image_size,
    transform_mean,
    transform_std,
    model_name
)

from source_code.config import (
    image_data_dir,
    metadata_path,
    image_statistics_path,
    drift_baseline_stats_path,
    inference_log_path,
    pytorch_model_path,
    model_results_path
)

# ----------------


class SimpleCNN(nn.Module):
    """
    Simple CNN for binary classification of glass vial images.

    Input: 64x64 RGB images (3 channels) - resized from native 1600x768
    Output: Single logit (positive = defective, negative = clean)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers. After 2 max pooling layers: 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Convolutional layers with ReLU and max pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 16 * 16)

        # Fully connected layers. No sigmoid here: the model is trained with
        # BCEWithLogitsLoss, callers apply torch.sigmoid for probabilities.
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def train_cnn_model(train_loader, val_loader, device, num_epochs=num_epochs, learning_rate=learning_rate, random_seed=random_state):
    """Train the SimpleCNN model on glass vial images."""
    print(f"\n{'='*60}")
    print("TRAINING CNN MODEL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = SimpleCNN().to(device)

    # Weighted BCE-with-logits loss for binary classification. The dataset is
    # imbalanced, so the positive (defective) class is down-weighted by the
    # negative/positive count ratio to rebalance the loss.
    train_labels = train_loader.dataset.labels
    num_positive = int(sum(train_labels))
    num_negative = len(train_labels) - num_positive
    print(f"Training class counts: clean(0)={num_negative}, defective(1)={num_positive}")
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32, device=device)
    print(f"Loss pos_weight: {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Halve the learning rate when validation macro-F1 plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=scheduler_patience)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1': [], 'val_f1': [], 'val_f1_macro': [],
        'best_val_f1_macro': 0.0, 'best_model_state': None,
    }
    epochs_without_improvement = 0

    print("\nStarting training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            # Forward pass (raw logits)
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics (logits -> probabilities)
            train_running_loss += loss.item()
            probs = torch.sigmoid(logits)
            predicted = (probs >= threshold).float().view(-1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.view(-1)).sum().item()

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_accuracy = train_correct / train_total
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                logits = model(images)
                loss = criterion(logits, labels)

                val_running_loss += loss.item()
                probs = torch.sigmoid(logits)
                predicted = (probs >= threshold).float().view(-1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.view(-1)).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='binary')
        epoch_val_f1_macro = f1_score(all_val_labels, all_val_preds, average='macro')

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_accuracy'].append(epoch_train_accuracy)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['train_f1'].append(epoch_train_f1)
        history['val_f1'].append(epoch_val_f1)
        history['val_f1_macro'].append(epoch_val_f1_macro)

        # Step the LR scheduler on validation macro-F1
        scheduler.step(epoch_val_f1_macro)

        # Save best model based on validation macro-F1
        if epoch_val_f1_macro > history['best_val_f1_macro']:
            history['best_val_f1_macro'] = epoch_val_f1_macro
            # Deep-copy the weights so the "best" state does not keep changing
            history['best_model_state'] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping: no macro-F1 improvement for {early_stop_patience} consecutive epochs")
                break

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}] (lr={optimizer.param_groups[0]['lr']:.6f}):")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.4f} | Train F1: {epoch_train_f1:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.4f} | Val F1: {epoch_val_f1:.4f} | Val Macro F1: {epoch_val_f1_macro:.4f}")

        # Log to MLflow (only if run is active)
        if mlflow.active_run():
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_accuracy, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_accuracy, step=epoch)
            mlflow.log_metric("train_f1_score", epoch_train_f1, step=epoch)
            mlflow.log_metric("val_f1_score", epoch_val_f1, step=epoch)
            mlflow.log_metric("val_f1_macro", epoch_val_f1_macro, step=epoch)

    # Load best model weights
    if history['best_model_state'] is not None:
        model.load_state_dict(history['best_model_state'])
        print(f"\nRestored best model weights (best val macro F1: {history['best_val_f1_macro']:.4f})")

    print(f"\nTraining complete!")
    print(f"Best validation macro F1-score: {history['best_val_f1_macro']:.4f}")

    return model, history


def evaluate_cnn_model(model, test_loader, device, threshold=threshold):
    """Evaluate the model on the test set: accuracy, F1, precision, recall, ROC-AUC."""
    print(f"\n{'='*60}")
    print("EVALUATING CNN MODEL")
    print(f"{'='*60}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Threshold: {threshold}")

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)
            probs = torch.sigmoid(logits)

            probs = probs.cpu().numpy()
            preds = (probs >= threshold).astype(float).reshape(-1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    roc_auc = roc_auc_score(all_labels, all_probs)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True)

    print(f"\nTest Set Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  [[TN, FP],")
    print(f"   [FN, TP]] = {conf_matrix.tolist()}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # Log metrics to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_roc_auc", roc_auc)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
    }


def print_section_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def tune_threshold(model, val_loader, device, threshold_range=None):
    """Tune the classification threshold to maximize macro F1 on the validation set.

    Macro-F1 is used because maximizing the positive-class F1 alone rewards
    predicting "defective" for every image (defective is the majority class).
    """
    if threshold_range is None:
        # Fine grid: the model's probabilities can compress into a narrow band
        threshold_range = [round(t, 3) for t in np.arange(0.025, 0.976, 0.025)]

    print(f"\n{'='*60}")
    print("TUNING CLASSIFICATION THRESHOLD")
    print(f"{'='*60}")
    print(f"Threshold range: {threshold_range}")

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)

            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels)

    best_threshold = threshold
    best_f1 = 0.0

    for t in threshold_range:
        preds = (all_probs >= t).astype(int)
        f1_macro = f1_score(all_labels, preds, average='macro')
        f1_positive = f1_score(all_labels, preds, average='binary')

        print(f"  Threshold {t}: MacroF1={f1_macro:.4f}, F1(defective)={f1_positive:.4f}, Acc={accuracy_score(all_labels, preds):.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = t

    print(f"\nOptimal threshold: {best_threshold} (macro F1-score: {best_f1:.4f})")

    # Log to MLflow
    mlflow.log_param("optimal_threshold", best_threshold)
    mlflow.log_metric("optimal_threshold_macro_f1", best_f1)

    return best_threshold


def calculate_image_statistics(data_loader):
    """Calculate mean and standard deviation per channel for a dataset."""
    print("Calculating image statistics...")

    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_samples = 0

    for images, _ in data_loader:
        batch_images = images.numpy()
        batch_size = batch_images.shape[0]

        # Reshape to (batch_size, -1, 3) for channel-wise stats
        batch_images = batch_images.transpose(0, 2, 3, 1).reshape(batch_size, -1, 3)

        batch_mean = batch_images.mean(axis=1)
        batch_std = batch_images.std(axis=1)

        total_mean += batch_mean.sum(axis=0)
        total_std += batch_std.sum(axis=0)
        total_samples += batch_size

    overall_mean = (total_mean / total_samples).tolist()
    overall_std = (total_std / total_samples).tolist()

    print(f"Image statistics: Mean={overall_mean}, Std={overall_std}")
    return {
        "mean": overall_mean,
        "std": overall_std,
        "total_samples": total_samples,
    }


def compute_baseline_statistics(model, data_loader, device, feature_layer="fc1", threshold=threshold):
    """Compute baseline statistics for drift detection from a dataset."""
    print(f"\nComputing baseline statistics for drift detection...")
    print(f"Feature layer: {feature_layer}")

    model.eval()

    image_means = []
    image_stds = []
    class_counts = {0: 0, 1: 0}
    prediction_counts = {0: 0, 1: 0}
    confidences = []
    all_embeddings = []

    # Get the feature extraction layer
    if hasattr(model, feature_layer):
        feature_extractor = getattr(model, feature_layer)
    else:
        raise ValueError(f"Model does not have layer '{feature_layer}'")

    # Hook to capture embeddings
    embeddings = []
    def embedding_hook(module, input, output):
        embeddings.append(output.detach().cpu().numpy())

    handle = feature_extractor.register_forward_hook(embedding_hook)

    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                # Get predictions (logits -> probabilities)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= threshold).astype(int)

                # Compute image statistics (per channel)
                batch_mean = images.mean(dim=(0, 2, 3)).cpu().numpy()
                batch_std = images.std(dim=(0, 2, 3)).cpu().numpy()

                image_means.append(batch_mean)
                image_stds.append(batch_std)

                # Update class counts
                for label in labels.cpu().numpy().flatten():
                    class_counts[int(label)] += 1

                # Update prediction counts
                for pred in preds:
                    prediction_counts[int(pred)] += 1

                # Collect confidences
                confidences.extend(probs)

                # Get embeddings from the hook
                if embeddings:
                    all_embeddings.extend([e for e in embeddings])
                    embeddings.clear()

        # Compute final statistics
        image_means = np.array(image_means)
        image_stds = np.array(image_stds)

        baseline_stats = {
            'image_statistics': {
                'mean_r': float(np.mean(image_means[:, 0])),
                'mean_g': float(np.mean(image_means[:, 1])),
                'mean_b': float(np.mean(image_means[:, 2])),
                'std_r': float(np.mean(image_stds[:, 0])),
                'std_g': float(np.mean(image_stds[:, 1])),
                'std_b': float(np.mean(image_stds[:, 2])),
            },
            'class_distribution': {
                'class_0_count': class_counts[0],
                'class_1_count': class_counts[1],
                'total': class_counts[0] + class_counts[1],
            },
            'prediction_distribution': {
                'pred_0_count': prediction_counts[0],
                'pred_1_count': prediction_counts[1],
                'total': prediction_counts[0] + prediction_counts[1],
            },
            'confidence_statistics': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
            },
            'embedding_statistics': {
                'mean': [],
                'std': [],
            },
        }

        # Compute embedding statistics if embeddings were captured
        if all_embeddings:
            all_embeddings_array = np.concatenate(all_embeddings, axis=0)
            embedding_mean = np.mean(all_embeddings_array, axis=0)
            embedding_std = np.std(all_embeddings_array, axis=0)
            baseline_stats['embedding_statistics'] = {
                'mean': embedding_mean.tolist(),
                'std': embedding_std.tolist(),
                'shape': int(all_embeddings_array.shape[1]),
            }

        print(f"Baseline statistics computed:")
        print(f"  Class distribution: 0={class_counts[0]}, 1={class_counts[1]}")
        print(f"  Confidence mean: {baseline_stats['confidence_statistics']['mean']:.4f}")
        if all_embeddings:
            print(f"  Embedding shape: {baseline_stats['embedding_statistics']['shape']}")

        return baseline_stats

    finally:
        handle.remove()


class SimpleLoggingWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper that logs inference requests and returns predictions."""

    def __init__(self, model=None, model_path=None, log_path=Path(inference_log_path), threshold=threshold):
        self.model = model
        self.model_path = model_path
        self.log_path = Path(log_path)
        self.threshold = threshold

        # Transform for preprocessing images (same as training, no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std),
        ])

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
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model = self._create_model_from_checkpoint(checkpoint)
            else:
                raise FileNotFoundError("Could not find model file")
        self.model.to(self.device)
        self.model.eval()
        print("SimpleLoggingWrapper context loaded successfully!")

    def _create_model_from_checkpoint(self, checkpoint):
        architecture = checkpoint.get('architecture', 'SimpleCNN')
        if architecture == 'SimpleCNN':
            model = SimpleCNN()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model

    def _get_timestamp_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _log_request(self, request_data):
        """Append a request log entry to the JSON log file."""
        try:
            logs = []
            if self.log_path.exists():
                with open(self.log_path, 'r') as f:
                    try:
                        logs = json.load(f)
                        if not isinstance(logs, list):
                            logs = []
                    except json.JSONDecodeError:
                        logs = []

            logs.append(request_data)

            with open(self.log_path, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to log request: {e}")

    def preprocess_image(self, image_input):
        """Preprocess a single image: path, raw bytes or base64 string."""
        if isinstance(image_input, bytes):
            image = Image.open(BytesIO(image_input)).convert('RGB')
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
                # Treat as base64-encoded image data (the portable contract
                # for containerized serving)
                try:
                    decoded_bytes = base64.b64decode(image_input)
                    image = Image.open(BytesIO(decoded_bytes)).convert('RGB')
                except Exception as e:
                    raise ValueError(
                        f"Cannot process string input '{image_input[:50]}...': "
                        f"not an existing file path and not valid base64 image data: {e}"
                    )
        elif isinstance(image_input, Path):
            if image_input.exists():
                image = Image.open(image_input).convert('RGB')
            else:
                raise FileNotFoundError(f"File not found: {image_input}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}. Expected str, Path, or bytes.")

        return self.transform(image).unsqueeze(0)

    def predict(self, context, model_input):
        """Make predictions on input data and log the request."""
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

                image_identifier = "<bytes>" if isinstance(image_input, bytes) else str(image_input)
                log_entry = {
                    'timestamp': timestamp,
                    'image_path': image_identifier,
                    'prediction': int(pred),
                    'confidence': float(prob),
                    'raw_output': float(prob),
                }
                self._log_request(log_entry)

            except Exception as e:
                print(f"Error processing image {image_input}: {e}")
                predictions.append(None)
                confidences.append(None)
                raw_outputs.append(None)

        return {
            'predictions': predictions,
            'confidences': confidences,
            'raw_outputs': raw_outputs,
            'num_samples': len(image_inputs),
        }


def wait_until_ready(model_name, model_version, max_attempts=10):
    """Wait until the model version is READY in the registry."""
    client = MlflowClient()
    for attempt in range(max_attempts):
        try:
            model_version_details = client.get_model_version(name=model_name, version=model_version)
            status = ModelVersionStatus.from_string(model_version_details.status)
            print(f"Model status: {ModelVersionStatus.to_string(status)}")
            if status == ModelVersionStatus.READY:
                print(f"Model {model_name} version {model_version} is READY")
                return True
        except Exception as e:
            print(f"Error checking model status (attempt {attempt + 1}/{max_attempts}): {e}")
        time.sleep(1)
    print(f"Model {model_name} version {model_version} did not become READY after {max_attempts} attempts")
    return False



def main():

    setup_mlflow_tracking(experiment_name)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    train_loader, val_loader, test_loader = apply_preprocessing()

    # Save the image statistics of the training set as an artifact
    image_stats = calculate_image_statistics(train_loader)
    with open(image_statistics_path, "w") as f:
        json.dump(image_stats, f, indent=2)
    print(f"Image statistics saved to {image_statistics_path}")


    print_section_header("CNN MODEL TRAINING")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if mlflow.active_run():
        mlflow.end_run()
        print("Ended previous active MLflow run")

    run = mlflow.start_run(run_name="training_notebook")

    # Log parameters to MLflow
    mlflow.log_param("model_architecture", "SimpleCNN")
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("initial_threshold", threshold)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("image_size", "64x64")
    mlflow.log_param("random_seed", random_state)

    # Log model architecture details
    model_params = sum(p.numel() for p in SimpleCNN().parameters())
    trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
    mlflow.log_param("total_parameters", model_params)
    mlflow.log_param("trainable_parameters", trainable_params)

    cnn_model, training_history = train_cnn_model(
        train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        random_seed=random_state,
    )

    # Tune threshold on validation set
    optimal_threshold = tune_threshold(cnn_model, val_loader, device=device)

    # Evaluate on test set with optimal threshold
    test_results = evaluate_cnn_model(cnn_model, test_loader, device=device, threshold=optimal_threshold)

    baseline_stats = compute_baseline_statistics(cnn_model, val_loader, device= device, threshold=optimal_threshold)

    with open(drift_baseline_stats_path, 'w') as f:
        json.dump(baseline_stats, f, indent=2)
    print(f"Baseline statistics saved to {drift_baseline_stats_path}")

    # Log baseline statistics to MLflow
    mlflow.log_metric("baseline_image_mean_r", baseline_stats['image_statistics']['mean_r'])
    mlflow.log_metric("baseline_image_mean_g", baseline_stats['image_statistics']['mean_g'])
    mlflow.log_metric("baseline_image_mean_b", baseline_stats['image_statistics']['mean_b'])
    mlflow.log_metric("baseline_image_std_r", baseline_stats['image_statistics']['std_r'])
    mlflow.log_metric("baseline_image_std_g", baseline_stats['image_statistics']['std_g'])
    mlflow.log_metric("baseline_image_std_b", baseline_stats['image_statistics']['std_b'])
    mlflow.log_metric("baseline_class_0_count", baseline_stats['class_distribution']['class_0_count'])
    mlflow.log_metric("baseline_class_1_count", baseline_stats['class_distribution']['class_1_count'])
    mlflow.log_metric("baseline_confidence_mean", baseline_stats['confidence_statistics']['mean'])
    mlflow.log_metric("baseline_confidence_std", baseline_stats['confidence_statistics']['std'])
    mlflow.log_artifact(drift_baseline_stats_path, "drift_detection")

    print("\nSaving model artifacts...")

    # Save model state dict
    torch.save({
        'model_state_dict': cnn_model.state_dict(),
        'architecture': SimpleCNN.__name__,
        'input_shape': (3, 64, 64),
        'output_shape': (1,),
        'threshold': optimal_threshold,
        'num_classes': num_classes,
    }, pytorch_model_path)
    print(f"CNN model saved to {pytorch_model_path}")

    # Save model results
    with open(model_results_path, "w") as results_file:
        json.dump(test_results, results_file, indent=2)
    print(f"Model results saved to {model_results_path}")

    mlflow.log_param("classification_threshold", optimal_threshold)

    # Log the pyfunc model with the SimpleLoggingWrapper as the model artifact
    wrapper = SimpleLoggingWrapper(model=cnn_model)
    mlflow.pyfunc.log_model("model_pyfunc", python_model=wrapper)
    print("Logged pyfunc model to the MLflow run")

    model_uri = f"runs:/{run.info.run_id}/model_pyfunc"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    wait_until_ready(model_details.name, model_details.version)

    print(f"Model registered successfully:")
    print(f"  Name: {model_details.name}")
    print(f"  Version: {model_details.version}")
    print(f"  Stage: {model_details.current_stage}")

    # Transition to staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(model_name, model_details.version, "deployed_to", "staging")
    client.set_model_version_tag(model_name, model_details.version, "task", "classification")
    client.set_registered_model_alias(name=model_name, alias="best", version=str(model_details.version))

    print(f"Successfully transitioned {model_name} version {model_details.version} to Staging")
    pprint(dict(model_details))

    mlflow.end_run()

    print("\n" + "=" * 50)
    print("TRAINING JOB COMPLETE")
    print("=" * 50)
    print("\nSummary:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Test F1-score: {test_results['f1_score']:.4f}")
    print(f"  Registered model: {model_name} version {model_details.version} (Staging)")

if __name__ == "__main__":
    main()