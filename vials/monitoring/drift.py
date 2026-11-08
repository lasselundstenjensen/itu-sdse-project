from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from loguru import logger
import numpy as np
from scipy.stats import ks_2samp

from vials.config import (
    DRIFT_BASELINE_STATS_PATH,
    DRIFT_DEFAULT_DAYS,
    DRIFT_KS_PVALUE_THRESHOLD,
    DRIFT_MEAN_DIFF_THRESHOLD,
    INFERENCE_LOG_PATH,
)


def load_inference_logs(log_path=INFERENCE_LOG_PATH):
    """Load the inference log JSON file (may not exist yet)."""
    if not Path(log_path).exists():
        logger.warning(f"Inference log not found at {log_path}")
        return []
    try:
        with open(log_path, "r") as f:
            logs = json.load(f)
            if not isinstance(logs, list):
                return []
            return logs
    except (ValueError, OSError) as e:
        logger.warning(f"Failed to load inference log: {e}")
        return []


def filter_logs_by_time(logs, days=DRIFT_DEFAULT_DAYS):
    """Only keep log entries from the last N days."""
    if not logs:
        return []
    cutoff = datetime.now(UTC) - timedelta(days=days)
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


def load_baseline_stats(path=DRIFT_BASELINE_STATS_PATH):
    """Load the baseline statistics written during training."""
    if not Path(path).exists():
        logger.warning(f"Baseline stats not found at {path}")
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (ValueError, OSError) as e:
        logger.warning(f"Failed to load baseline stats: {e}")
        return None


def calculate_confidence_metrics(confidences):
    confidences_array = np.array(confidences)
    return {
        "mean": float(np.mean(confidences_array)) if len(confidences_array) > 0 else 0.0,
        "std": float(np.std(confidences_array)) if len(confidences_array) > 0 else 0.0,
        "min": float(np.min(confidences_array)) if len(confidences_array) > 0 else 0.0,
        "max": float(np.max(confidences_array)) if len(confidences_array) > 0 else 0.0,
    }


def check_confidence_drift(current_confidences, baseline_stats):
    """KS test of the current confidence scores against the baseline."""
    if not baseline_stats or "confidence_statistics" not in baseline_stats:
        return {
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "drift_detected": False,
            "threshold_ks": DRIFT_KS_PVALUE_THRESHOLD,
        }

    baseline = baseline_stats["confidence_statistics"]
    current_confidences_array = np.array(current_confidences)

    if len(current_confidences_array) < 2:
        return {
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "drift_detected": False,
            "threshold_ks": DRIFT_KS_PVALUE_THRESHOLD,
        }

    baseline_mean = baseline["mean"]
    baseline_std = baseline["std"]
    current_mean = float(np.mean(current_confidences_array))

    baseline_samples = np.random.normal(
        baseline_mean, baseline_std, size=len(current_confidences_array)
    )
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current_confidences_array)

    mean_diff = abs(current_mean - baseline_mean)
    ks_drift = ks_pvalue < DRIFT_KS_PVALUE_THRESHOLD
    mean_drift = mean_diff > DRIFT_MEAN_DIFF_THRESHOLD
    drift_detected = bool(ks_drift or mean_drift)

    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "mean_diff": mean_diff,
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
        "drift_detected": drift_detected,
        "threshold_ks": DRIFT_KS_PVALUE_THRESHOLD,
        "threshold_mean": DRIFT_MEAN_DIFF_THRESHOLD,
    }


def check_class_distribution_drift(predictions, baseline_stats):
    """KS test of the predicted class distribution against the baseline."""
    if not baseline_stats or "class_distribution" not in baseline_stats:
        return {
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "drift_detected": False,
            "threshold_ks": DRIFT_KS_PVALUE_THRESHOLD,
        }

    baseline = baseline_stats["class_distribution"]
    baseline_total = baseline["total"]
    current_total = len(predictions)

    if current_total == 0:
        return {
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "drift_detected": False,
            "threshold_ks": DRIFT_KS_PVALUE_THRESHOLD,
        }

    baseline_prop_0 = baseline["class_0_count"] / baseline_total
    baseline_prop_1 = baseline["class_1_count"] / baseline_total
    baseline_dist = np.array([baseline_prop_0, baseline_prop_1])

    baseline_samples = np.random.choice([0, 1], size=baseline_total, p=baseline_dist)
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, predictions)

    ks_drift = ks_pvalue < DRIFT_KS_PVALUE_THRESHOLD
    drift_detected = bool(ks_drift)

    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "drift_detected": drift_detected,
        "threshold_ks": DRIFT_KS_PVALUE_THRESHOLD,
    }


def detect_drift(logs, days=DRIFT_DEFAULT_DAYS, baseline_stats=None):
    """Main drift detection: confidences + class distribution."""
    filtered_logs = filter_logs_by_time(logs, days=days)

    if not filtered_logs:
        logger.warning(f"No inference logs found in the last {days} days")
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

    num_samples = len(filtered_logs)

    confidence_results = check_confidence_drift(confidences, baseline_stats or {})
    class_dist_results = check_class_distribution_drift(predictions, baseline_stats or {})

    overall_drift = confidence_results.get("drift_detected", False) or class_dist_results.get(
        "drift_detected", False
    )

    return {
        "overall_drift_detected": overall_drift,
        "num_samples": num_samples,
        "confidence_drift": confidence_results,
        "class_distribution_drift": class_dist_results,
        "confidence_stats": calculate_confidence_metrics(confidences),
        "class_distribution": {
            "class_0_count": predictions.count(0),
            "class_1_count": predictions.count(1),
            "total": len(predictions),
        },
    }
