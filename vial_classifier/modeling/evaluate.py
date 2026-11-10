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

from vial_classifier.config import THRESHOLD
from vial_classifier.utils import get_device


def evaluate_cnn_model(model, test_loader, device=None, threshold=THRESHOLD):
    """Evaluate the model on the test set: accuracy, F1, precision, recall, ROC-AUC."""
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

    print("\nTest Set Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print("  [[TN, FP],")
    print(f"   [FN, TP]] = {conf_matrix.tolist()}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

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


def compute_baseline_statistics(
    model, data_loader, device=None, feature_layer="fc1", threshold=THRESHOLD
):
    """Compute baseline statistics for drift detection from a dataset."""
    device = device or get_device()
    print("\nComputing baseline statistics for drift detection...")
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
            for images, labels in data_loader:
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
            "embedding_statistics": {
                "mean": [],
                "std": [],
            },
        }

        # Compute embedding statistics if embeddings were captured
        if all_embeddings:
            all_embeddings_array = np.concatenate(all_embeddings, axis=0)
            embedding_mean = np.mean(all_embeddings_array, axis=0)
            embedding_std = np.std(all_embeddings_array, axis=0)
            baseline_stats["embedding_statistics"] = {
                "mean": embedding_mean.tolist(),
                "std": embedding_std.tolist(),
                "shape": int(all_embeddings_array.shape[1]),
            }

        print("Baseline statistics computed:")
        print(f"  Class distribution: 0={class_counts[0]}, 1={class_counts[1]}")
        print(f"  Confidence mean: {baseline_stats['confidence_statistics']['mean']:.4f}")
        if all_embeddings:
            print(f"  Embedding shape: {baseline_stats['embedding_statistics']['shape']}")

        return baseline_stats

    finally:
        handle.remove()
