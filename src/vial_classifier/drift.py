"""Drift detection: training-time baseline, and KS tests on logged requests.

CLI: ``python -m vial_classifier.drift [--days N] [--endpoint URL] [--fail-on-drift]``
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import numpy as np
import requests
import torch
from scipy.stats import ks_2samp

from vial_classifier import config
from vial_classifier.model import get_device


# --- Baseline (computed at training time) -------------------------------------

def compute_baseline_statistics(model, data_loader, device=None, feature_layer="fc1",
                                threshold=config.THRESHOLD):
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

                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= threshold).astype(int)

                image_means.append(images.mean(dim=(0, 2, 3)).cpu().numpy())
                image_stds.append(images.std(dim=(0, 2, 3)).cpu().numpy())

                for label in labels.cpu().numpy().flatten():
                    class_counts[int(label)] += 1
                for pred in preds:
                    prediction_counts[int(pred)] += 1

                confidences.extend(probs)

                if embeddings:
                    all_embeddings.extend(embeddings)
                    embeddings.clear()

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
            "embedding_statistics": {"mean": [], "std": []},
        }

        if all_embeddings:
            all_embeddings_array = np.concatenate(all_embeddings, axis=0)
            baseline_stats["embedding_statistics"] = {
                "mean": np.mean(all_embeddings_array, axis=0).tolist(),
                "std": np.std(all_embeddings_array, axis=0).tolist(),
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


def save_baseline_stats(baseline_stats, path=config.DRIFT_BASELINE_STATS_PATH):
    with open(path, "w") as f:
        json.dump(baseline_stats, f, indent=2)
    print(f"Baseline statistics saved to {path}")


def log_baseline_to_mlflow(baseline_stats, path=config.DRIFT_BASELINE_STATS_PATH):
    """Log the baseline as metrics and as an artifact of the active MLflow run."""
    image = baseline_stats["image_statistics"]
    for channel in ("r", "g", "b"):
        mlflow.log_metric(f"baseline_image_mean_{channel}", image[f"mean_{channel}"])
    for channel in ("r", "g", "b"):
        mlflow.log_metric(f"baseline_image_std_{channel}", image[f"std_{channel}"])
    mlflow.log_metric("baseline_class_0_count", baseline_stats["class_distribution"]["class_0_count"])
    mlflow.log_metric("baseline_class_1_count", baseline_stats["class_distribution"]["class_1_count"])
    mlflow.log_metric("baseline_confidence_mean", baseline_stats["confidence_statistics"]["mean"])
    mlflow.log_metric("baseline_confidence_std", baseline_stats["confidence_statistics"]["std"])
    mlflow.log_artifact(path, "drift_detection")


# --- Detection (runs at serving time) -----------------------------------------

def load_inference_logs(log_path=config.INFERENCE_LOG_PATH):
    """Load the inference log JSON file (may not exist yet)."""
    if not Path(log_path).exists():
        print(f"Warning: Inference log not found at {log_path}")
        return []
    try:
        with open(log_path, "r") as f:
            logs = json.load(f)
            if not isinstance(logs, list):
                return []
            return logs
    except Exception as e:
        print(f"Warning: Failed to load inference log: {e}")
        return []


def filter_logs_by_time(logs, days=config.DRIFT_DEFAULT_DAYS):
    """Only keep log entries from the last N days."""
    if not logs:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for log in logs:
        try:
            timestamp_str = log.get("timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp >= cutoff:
                    filtered.append(log)
        except (ValueError, TypeError):
            filtered.append(log)
    return filtered


def load_baseline_stats(path=config.DRIFT_BASELINE_STATS_PATH):
    """Load the baseline statistics written during training."""
    if not Path(path).exists():
        print(f"Warning: Baseline stats not found at {path}")
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load baseline stats: {e}")
        return None


def calculate_confidence_metrics(confidences):
    arr = np.array(confidences)
    return {
        "mean": float(np.mean(arr)) if len(arr) > 0 else 0.0,
        "std": float(np.std(arr)) if len(arr) > 0 else 0.0,
        "min": float(np.min(arr)) if len(arr) > 0 else 0.0,
        "max": float(np.max(arr)) if len(arr) > 0 else 0.0,
    }


def _no_drift():
    return {
        "ks_statistic": 0.0,
        "ks_pvalue": 1.0,
        "drift_detected": False,
        "threshold_ks": config.DRIFT_KS_PVALUE_THRESHOLD,
    }


def check_confidence_drift(current_confidences, baseline_stats):
    """KS test of the current confidence scores against the baseline."""
    if not baseline_stats or "confidence_statistics" not in baseline_stats:
        return _no_drift()

    baseline = baseline_stats["confidence_statistics"]
    current = np.array(current_confidences)

    if len(current) < 2:
        return _no_drift()

    baseline_mean = baseline["mean"]
    baseline_std = baseline["std"]
    current_mean = float(np.mean(current))

    baseline_samples = np.random.normal(baseline_mean, baseline_std, size=len(current))
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current)

    mean_diff = abs(current_mean - baseline_mean)
    ks_drift = ks_pvalue < config.DRIFT_KS_PVALUE_THRESHOLD
    mean_drift = mean_diff > config.DRIFT_MEAN_THRESHOLD

    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "mean_diff": mean_diff,
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
        "drift_detected": bool(ks_drift or mean_drift),
        "threshold_ks": config.DRIFT_KS_PVALUE_THRESHOLD,
        "threshold_mean": config.DRIFT_MEAN_THRESHOLD,
    }


def check_class_distribution_drift(predictions, baseline_stats):
    """KS test of the predicted class distribution against the baseline."""
    if not baseline_stats or "class_distribution" not in baseline_stats:
        return _no_drift()

    baseline = baseline_stats["class_distribution"]
    baseline_total = baseline["total"]

    if len(predictions) == 0:
        return _no_drift()

    baseline_dist = np.array([
        baseline["class_0_count"] / baseline_total,
        baseline["class_1_count"] / baseline_total,
    ])

    baseline_samples = np.random.choice([0, 1], size=baseline_total, p=baseline_dist)
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, predictions)

    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "drift_detected": bool(ks_pvalue < config.DRIFT_KS_PVALUE_THRESHOLD),
        "threshold_ks": config.DRIFT_KS_PVALUE_THRESHOLD,
    }


def detect_drift(logs, days=config.DRIFT_DEFAULT_DAYS, baseline_stats=None):
    """Main drift detection: confidences + class distribution."""
    filtered_logs = filter_logs_by_time(logs, days=days)

    if not filtered_logs:
        print(f"Warning: No inference logs found in the last {days} days")
        return {
            "overall_drift_detected": False,
            "num_samples": 0,
            "confidence_drift": None,
            "class_distribution_drift": None,
            "error": "No logs found",
        }

    confidences = []
    predictions = []
    for log in filtered_logs:
        if "confidence" in log and log["confidence"] is not None:
            confidences.append(float(log["confidence"]))
        if "prediction" in log and log["prediction"] is not None:
            predictions.append(int(log["prediction"]))

    confidence_results = check_confidence_drift(confidences, baseline_stats or {})
    class_dist_results = check_class_distribution_drift(predictions, baseline_stats or {})

    overall_drift = (
        confidence_results.get("drift_detected", False)
        or class_dist_results.get("drift_detected", False)
    )

    return {
        "overall_drift_detected": overall_drift,
        "num_samples": len(filtered_logs),
        "confidence_drift": confidence_results,
        "class_distribution_drift": class_dist_results,
        "confidence_stats": calculate_confidence_metrics(confidences),
        "class_distribution": {
            "class_0_count": predictions.count(0),
            "class_1_count": predictions.count(1),
            "total": len(predictions),
        },
    }


def run_drift_check(days=config.DRIFT_DEFAULT_DAYS, endpoint=None):
    """Run the drift check locally, or ask a running service via GET /drift."""
    if endpoint:
        response = requests.get(f"{endpoint.rstrip('/')}/drift", params={"days": days}, timeout=60)
        response.raise_for_status()
        return response.json()
    return detect_drift(load_inference_logs(), days=days, baseline_stats=load_baseline_stats())


def main():
    parser = argparse.ArgumentParser(description="Check for drift in the served model's traffic.")
    parser.add_argument("--days", type=int, default=config.DRIFT_DEFAULT_DAYS)
    parser.add_argument("--endpoint", help="Base URL of a running service (default: read local logs)")
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit with code 1 if drift is detected")
    args = parser.parse_args()

    results = run_drift_check(days=args.days, endpoint=args.endpoint)
    print(json.dumps(results, indent=2))

    if results["overall_drift_detected"]:
        print("\n!! Drift detected - time to retrain and redeploy")
        if args.fail_on_drift:
            sys.exit(1)
    else:
        print("\nNo drift detected")


if __name__ == "__main__":
    main()