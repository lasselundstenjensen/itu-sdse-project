"""Training, threshold tuning, evaluation and drift-baseline computation."""

from __future__ import annotations

import logging

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from . import config
from .model import SimpleCNN

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_eval_batches(model: nn.Module, loader: DataLoader, device: torch.device):
    """Yield ``(logits, labels)`` for each batch with gradients disabled."""
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)
            yield model(images), labels


def train_cnn_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = config.NUM_EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    device: torch.device | None = None,
    random_seed: int = config.RANDOM_STATE,
    threshold: float = config.THRESHOLD,
) -> tuple[nn.Module, dict]:
    """Train ``SimpleCNN`` with early stopping on validation macro-F1.

    Returns the model restored to its best validation weights and the
    per-epoch history. Metrics are logged to the active MLflow run, if any.
    """
    device = device or get_device()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = SimpleCNN().to(device)

    # The dataset is imbalanced (defective is the majority), so the positive
    # class is down-weighted by the negative/positive ratio.
    train_labels = train_loader.dataset.labels
    num_positive = int(sum(train_labels))
    num_negative = len(train_labels) - num_positive
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config.SCHEDULER_PATIENCE
    )

    history: dict = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_f1": [], "val_f1": [], "val_f1_macro": [],
        "best_val_f1_macro": 0.0, "best_model_state": None,
    }
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # --- training phase -------------------------------------------------
        model.train()
        running_loss = 0.0
        correct = total = 0
        train_preds, train_true = [], []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(logits) >= threshold).float().view(-1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(labels.cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total
        epoch_train_f1 = f1_score(train_true, train_preds, average="binary")

        # --- validation phase -----------------------------------------------
        val_loss_sum = 0.0
        val_correct = val_total = 0
        val_preds, val_true = [], []
        for logits, labels in _run_eval_batches(model, val_loader, device):
            val_loss_sum += criterion(logits, labels).item()
            predicted = (torch.sigmoid(logits) >= threshold).float().view(-1)
            val_total += labels.size(0)
            val_correct += (predicted == labels.view(-1)).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss_sum / len(val_loader)
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(val_true, val_preds, average="binary")
        epoch_val_f1_macro = f1_score(val_true, val_preds, average="macro")

        for key, value in [
            ("train_loss", epoch_train_loss), ("val_loss", epoch_val_loss),
            ("train_accuracy", epoch_train_acc), ("val_accuracy", epoch_val_acc),
            ("train_f1", epoch_train_f1), ("val_f1", epoch_val_f1),
            ("val_f1_macro", epoch_val_f1_macro),
        ]:
            history[key].append(value)

        scheduler.step(epoch_val_f1_macro)

        if epoch_val_f1_macro > history["best_val_f1_macro"]:
            history["best_val_f1_macro"] = epoch_val_f1_macro
            history["best_model_state"] = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                logger.info("Early stopping: no macro-F1 improvement for %d epochs",
                            config.EARLY_STOP_PATIENCE)
                break

        logger.info(
            "Epoch [%d/%d] lr=%.6f train_loss=%.4f train_f1=%.4f val_loss=%.4f val_macro_f1=%.4f",
            epoch + 1, num_epochs, optimizer.param_groups[0]["lr"],
            epoch_train_loss, epoch_train_f1, epoch_val_loss, epoch_val_f1_macro,
        )

        if mlflow.active_run():
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)
            mlflow.log_metric("train_f1_score", epoch_train_f1, step=epoch)
            mlflow.log_metric("val_f1_score", epoch_val_f1, step=epoch)
            mlflow.log_metric("val_f1_macro", epoch_val_f1_macro, step=epoch)

    if history["best_model_state"] is not None:
        model.load_state_dict(history["best_model_state"])
        logger.info("Restored best weights (val macro F1 %.4f)", history["best_val_f1_macro"])

    return model, history


def tune_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device | None = None,
    threshold_range: list[float] | None = None,
) -> float:
    """Pick the threshold that maximises macro-F1 on the validation set.

    Macro-F1 is used because positive-class F1 alone rewards predicting
    "defective" for every image.
    """
    device = device or get_device()
    if threshold_range is None:
        # The probabilities can compress into a narrow band, so use a fine grid
        threshold_range = [round(t, 3) for t in np.arange(0.025, 0.976, 0.025)]

    probs, labels = [], []
    for logits, batch_labels in _run_eval_batches(model, val_loader, device):
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())
    probs = np.array(probs).flatten()
    labels = np.array(labels)

    best_threshold, best_f1 = config.THRESHOLD, 0.0
    for t in threshold_range:
        preds = (probs >= t).astype(int)
        macro = f1_score(labels, preds, average="macro")
        logger.debug("threshold %.3f macro_f1=%.4f", t, macro)
        if macro > best_f1:
            best_f1, best_threshold = macro, t

    logger.info("Optimal threshold %.3f (macro F1 %.4f)", best_threshold, best_f1)
    if mlflow.active_run():
        mlflow.log_param("optimal_threshold", best_threshold)
        mlflow.log_metric("optimal_threshold_macro_f1", best_f1)
    return best_threshold


def evaluate_cnn_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device | None = None,
    threshold: float = config.THRESHOLD,
) -> dict:
    """Score the test set and log headline metrics to the active MLflow run."""
    device = device or get_device()

    probs, labels = [], []
    for logits, batch_labels in _run_eval_batches(model, test_loader, device):
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())
    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= threshold).astype(float).reshape(-1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    roc_auc = roc_auc_score(labels, probs)
    conf_matrix = confusion_matrix(labels, preds)

    logger.info("Test accuracy=%.4f f1=%.4f precision=%.4f recall=%.4f roc_auc=%.4f",
                accuracy, f1, precision, recall, roc_auc)
    logger.info("Classification report:\n%s", classification_report(labels, preds))

    if mlflow.active_run():
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_roc_auc", roc_auc)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": classification_report(labels, preds, output_dict=True),
    }


def compute_baseline_statistics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | None = None,
    feature_layer: str = "fc1",
    threshold: float = config.THRESHOLD,
) -> dict:
    """Summary statistics of the model and data, used later as the drift reference.

    Captures per-channel image statistics, class and prediction counts,
    confidence statistics and the mean/std of the ``feature_layer`` activations.
    """
    device = device or get_device()
    model.eval()

    if not hasattr(model, feature_layer):
        raise ValueError(f"Model does not have layer '{feature_layer}'")

    image_means, image_stds = [], []
    class_counts = {0: 0, 1: 0}
    prediction_counts = {0: 0, 1: 0}
    confidences: list[float] = []
    embeddings: list[np.ndarray] = []

    def capture(_module, _inputs, output):
        embeddings.append(output.detach().cpu().numpy())

    handle = getattr(model, feature_layer).register_forward_hook(capture)
    try:
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                probs = torch.sigmoid(model(images)).cpu().numpy().flatten()
                preds = (probs >= threshold).astype(int)

                image_means.append(images.mean(dim=(0, 2, 3)).cpu().numpy())
                image_stds.append(images.std(dim=(0, 2, 3)).cpu().numpy())
                for label in labels.cpu().numpy().flatten():
                    class_counts[int(label)] += 1
                for pred in preds:
                    prediction_counts[int(pred)] += 1
                confidences.extend(probs)
    finally:
        handle.remove()

    image_means = np.array(image_means)
    image_stds = np.array(image_stds)
    stats = {
        "image_statistics": {
            "mean_r": float(np.mean(image_means[:, 0])),
            "mean_g": float(np.mean(image_means[:, 1])),
            "mean_b": float(np.mean(image_means[:, 2])),
            "std_r": float(np.mean(image_stds[:, 0])),
            "std_g": float(np.mean(image_stds[:, 1])),
            "std_b": float(np.mean(image_stds[:, 2])),
        },
        "class_distribution": {
            "class_0_count": class_counts[0],
            "class_1_count": class_counts[1],
            "total": class_counts[0] + class_counts[1],
        },
        "prediction_distribution": {
            "pred_0_count": prediction_counts[0],
            "pred_1_count": prediction_counts[1],
            "total": prediction_counts[0] + prediction_counts[1],
        },
        "confidence_statistics": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
        },
        "embedding_statistics": {"mean": [], "std": []},
    }

    if embeddings:
        stacked = np.concatenate(embeddings, axis=0)
        stats["embedding_statistics"] = {
            "mean": np.mean(stacked, axis=0).tolist(),
            "std": np.std(stacked, axis=0).tolist(),
            "shape": int(stacked.shape[1]),
        }
    return stats
