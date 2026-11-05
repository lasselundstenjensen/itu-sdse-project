"""Model evaluation on the test split."""

from loguru import logger
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

from vial_classifier import config
from vial_classifier.modeling.model import get_device


def evaluate_cnn_model(model, test_loader, threshold=config.THRESHOLD, device=None):
    """Evaluate the model on the test set: accuracy, F1, precision, recall, ROC-AUC."""
    if device is None:
        device = get_device()

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
    f1 = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    roc_auc = roc_auc_score(all_labels, all_probs)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True)

    logger.info(
        f"Test results (threshold={threshold}): accuracy={accuracy:.4f}, F1={f1:.4f}, "
        f"precision={precision:.4f}, recall={recall:.4f}, ROC-AUC={roc_auc:.4f}, "
        f"confusion matrix={conf_matrix.tolist()}"
    )

    # Log metrics to MLflow
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
        "classification_report": class_report,
    }
