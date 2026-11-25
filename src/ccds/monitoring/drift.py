from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import ks_2samp


DEFAULT_DAYS = 7
DEFAULT_KS_PVALUE_THRESHOLD = 0.05
MEAN_DRIFT_THRESHOLD = 0.1


def filter_logs_by_time(
    logs: Sequence[Mapping[str, Any]], days: int = DEFAULT_DAYS
) -> list[Mapping[str, Any]]:
    if not logs:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered: list[Mapping[str, Any]] = []
    for log in logs:
        timestamp_string = log.get("timestamp", "")
        try:
            if timestamp_string:
                timestamp = datetime.fromisoformat(str(timestamp_string))
                if timestamp >= cutoff:
                    filtered.append(log)
        except (ValueError, TypeError):
            filtered.append(log)
    return filtered


def calculate_confidence_metrics(
    confidences: Sequence[float],
) -> dict[str, float]:
    values = np.asarray(confidences, dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def check_confidence_drift(
    current_confidences: Sequence[float],
    baseline_stats: Mapping[str, Any] | None,
) -> dict[str, float | bool]:
    neutral_result: dict[str, float | bool] = {
        "ks_statistic": 0.0,
        "ks_pvalue": 1.0,
        "drift_detected": False,
        "threshold_ks": DEFAULT_KS_PVALUE_THRESHOLD,
    }
    if not baseline_stats or "confidence_statistics" not in baseline_stats:
        return neutral_result

    current = np.asarray(current_confidences, dtype=float)
    if current.size < 2:
        return neutral_result

    baseline = baseline_stats["confidence_statistics"]
    baseline_mean = float(baseline["mean"])
    baseline_std = float(baseline["std"])
    current_mean = float(np.mean(current))
    baseline_samples = np.random.normal(
        baseline_mean, baseline_std, size=current.size
    )
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current)
    mean_diff = abs(current_mean - baseline_mean)
    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "mean_diff": float(mean_diff),
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
        "drift_detected": bool(
            ks_pvalue < DEFAULT_KS_PVALUE_THRESHOLD
            or mean_diff > MEAN_DRIFT_THRESHOLD
        ),
        "threshold_ks": DEFAULT_KS_PVALUE_THRESHOLD,
        "threshold_mean": MEAN_DRIFT_THRESHOLD,
    }


def check_class_distribution_drift(
    predictions: Sequence[int],
    baseline_stats: Mapping[str, Any] | None,
) -> dict[str, float | bool]:
    neutral_result: dict[str, float | bool] = {
        "ks_statistic": 0.0,
        "ks_pvalue": 1.0,
        "drift_detected": False,
        "threshold_ks": DEFAULT_KS_PVALUE_THRESHOLD,
    }
    if not baseline_stats or "class_distribution" not in baseline_stats:
        return neutral_result

    baseline = baseline_stats["class_distribution"]
    baseline_total = int(baseline["total"])
    current_total = len(predictions)
    if current_total == 0 or baseline_total == 0:
        return neutral_result

    baseline_distribution = np.array(
        [
            baseline["class_0_count"] / baseline_total,
            baseline["class_1_count"] / baseline_total,
        ]
    )
    baseline_samples = np.random.choice(
        [0, 1], size=baseline_total, p=baseline_distribution
    )
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, predictions)
    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "drift_detected": bool(ks_pvalue < DEFAULT_KS_PVALUE_THRESHOLD),
        "threshold_ks": DEFAULT_KS_PVALUE_THRESHOLD,
    }


def detect_drift(
    logs: Sequence[Mapping[str, Any]],
    days: int = DEFAULT_DAYS,
    baseline_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    filtered_logs = filter_logs_by_time(logs, days=days)
    if not filtered_logs:
        return {
            "overall_drift_detected": False,
            "num_samples": 0,
            "confidence_drift": None,
            "class_distribution_drift": None,
            "error": "No logs found",
        }

    confidences = [
        float(log["confidence"])
        for log in filtered_logs
        if log.get("confidence") is not None
    ]
    predictions = [
        int(log["prediction"])
        for log in filtered_logs
        if log.get("prediction") is not None
    ]
    confidence_drift = check_confidence_drift(confidences, baseline_stats)
    class_drift = check_class_distribution_drift(predictions, baseline_stats)
    return {
        "overall_drift_detected": bool(
            confidence_drift["drift_detected"] or class_drift["drift_detected"]
        ),
        "num_samples": len(filtered_logs),
        "confidence_drift": confidence_drift,
        "class_distribution_drift": class_drift,
        "confidence_stats": calculate_confidence_metrics(confidences),
        "class_distribution": {
            "class_0_count": predictions.count(0),
            "class_1_count": predictions.count(1),
            "total": len(predictions),
        },
    }
