import json

import mlflow
import numpy as np
import torch

from vial_classifier.config import DRIFT_BASELINE_STATS_PATH, THRESHOLD


def compute_baseline_statistics(
    model, data_loader, device=None, feature_layer="fc1", threshold=THRESHOLD
):
    """Compute baseline statistics for drift detection from a dataset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def save_and_log_baseline(baseline_stats, path=DRIFT_BASELINE_STATS_PATH):
    """Save the baseline to disk and log it to the active MLflow run."""
    with open(path, "w") as f:
        json.dump(baseline_stats, f, indent=2)
    print(f"Baseline statistics saved to {path}")

    if mlflow.active_run():
        image_stats = baseline_stats["image_statistics"]
        mlflow.log_metric("baseline_image_mean_r", image_stats["mean_r"])
        mlflow.log_metric("baseline_image_mean_g", image_stats["mean_g"])
        mlflow.log_metric("baseline_image_mean_b", image_stats["mean_b"])
        mlflow.log_metric("baseline_image_std_r", image_stats["std_r"])
        mlflow.log_metric("baseline_image_std_g", image_stats["std_g"])
        mlflow.log_metric("baseline_image_std_b", image_stats["std_b"])
        mlflow.log_metric(
            "baseline_class_0_count", baseline_stats["class_distribution"]["class_0_count"]
        )
        mlflow.log_metric(
            "baseline_class_1_count", baseline_stats["class_distribution"]["class_1_count"]
        )
        mlflow.log_metric(
            "baseline_confidence_mean", baseline_stats["confidence_statistics"]["mean"]
        )
        mlflow.log_metric(
            "baseline_confidence_std", baseline_stats["confidence_statistics"]["std"]
        )
        mlflow.log_artifact(str(path), "drift_detection")