"""Model evaluation: test-set metrics and classification threshold tuning."""

import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from vial_classifier import config
from vial_classifier.model import get_device


def evaluate_cnn_model(model, test_loader, device=None, threshold=config.THRESHOLD):
    """Evaluate on the test set: accuracy, F1, precision, recall, ROC-AUC."""
    device = device or get_device()

    print(f"\n{'=' * 60}")
    print("EVALUATING CNN MODEL")
    print(f"{'=' * 60}")
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

            probs = torch.sigmoid(model(images)).cpu().numpy()
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

    print("\nTest Set Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print("  [[TN, FP],")
    print("   [FN, TP]] =", conf_matrix.tolist())
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

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


def tune_threshold(model, val_loader, device=None, threshold_range=None):
    """Tune the classification threshold to maximize macro F1 on the validation set.

    Macro-F1 is used because maximizing the positive-class F1 alone rewards
    predicting "defective" for every image (defective is the majority class).
    """
    device = device or get_device()

    if threshold_range is None:
        # Fine grid: the model's probabilities can compress into a narrow band
        threshold_range = [round(t, 3) for t in np.arange(0.025, 0.976, 0.025)]

    print(f"\n{'=' * 60}")
    print("TUNING CLASSIFICATION THRESHOLD")
    print(f"{'=' * 60}")
    print(f"Threshold range: {threshold_range}")

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            all_probs.extend(torch.sigmoid(model(images)).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels)

    best_threshold = config.THRESHOLD
    best_f1 = 0.0

    for t in threshold_range:
        preds = (all_probs >= t).astype(int)
        f1_macro = f1_score(all_labels, preds, average="macro")
        f1_positive = f1_score(all_labels, preds, average="binary")

        print(f"  Threshold {t}: MacroF1={f1_macro:.4f}, F1(defective)={f1_positive:.4f}, "
              f"Acc={accuracy_score(all_labels, preds):.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = t

    print(f"\nOptimal threshold: {best_threshold} (macro F1-score: {best_f1:.4f})")

    mlflow.log_param("optimal_threshold", best_threshold)
    mlflow.log_metric("optimal_threshold_macro_f1", best_f1)

    return best_threshold