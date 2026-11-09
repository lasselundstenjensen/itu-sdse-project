"""Drift detection: compare served traffic against the training baseline.

Two-sample Kolmogorov-Smirnov tests are run on logged confidences and on the
predicted-class distribution. The confidence check also flags a shift in the
mean. The baseline comes from ``compute_baseline_statistics`` at train time.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from . import config

logger = logging.getLogger(__name__)


def _no_drift(threshold: float) -> dict:
    return {"ks_statistic": 0.0, "ks_pvalue": 1.0, "drift_detected": False, "threshold_ks": threshold}


def load_inference_logs(log_path: Path = config.INFERENCE_LOG_PATH) -> list[dict]:
    """Read the request log. Returns an empty list if it is missing or malformed."""
    if not Path(log_path).exists():
        logger.warning("Inference log not found at %s", log_path)
        return []
    try:
        logs = json.loads(Path(log_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load inference log: %s", exc)
        return []
    return logs if isinstance(logs, list) else []


def load_baseline_stats(path: Path = config.DRIFT_BASELINE_STATS_PATH) -> dict | None:
    if not Path(path).exists():
        logger.warning("Baseline stats not found at %s", path)
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load baseline stats: %s", exc)
        return None


def filter_logs_by_time(logs: list[dict], days: int = config.DRIFT_DEFAULT_DAYS, now: datetime | None = None) -> list[dict]:
    """Keep entries from the last ``days`` days.

    Behaviour matches the original notebook: entries without a timestamp or
    older than the cutoff are dropped; entries whose timestamp cannot be
    parsed are kept rather than silently lost.
    """
    if not logs:
        return []
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    kept = []
    for entry in logs:
        try:
            stamp = entry.get("timestamp", "")
            if not stamp:
                continue
            if datetime.fromisoformat(stamp) >= cutoff:
                kept.append(entry)
        except (ValueError, TypeError, AttributeError):
            kept.append(entry)
    return kept


def calculate_confidence_metrics(confidences) -> dict:
    arr = np.asarray(confidences, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def check_confidence_drift(
    current_confidences,
    baseline_stats: dict,
    rng: np.random.Generator | None = None,
) -> dict:
    """KS test of current confidences against a normal approximation of the baseline.

    The baseline is only stored as mean and std, so a sample is drawn from
    ``N(mean, std)`` and compared with the live confidences.
    """
    rng = rng or np.random.default_rng()
    threshold = config.DRIFT_KS_PVALUE_THRESHOLD
    if not baseline_stats or "confidence_statistics" not in baseline_stats:
        return _no_drift(threshold)

    current = np.asarray(current_confidences, dtype=float)
    if current.size < 2:
        return _no_drift(threshold)

    baseline = baseline_stats["confidence_statistics"]
    baseline_samples = rng.normal(baseline["mean"], baseline["std"], size=current.size)
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current)

    current_mean = float(current.mean())
    mean_diff = abs(current_mean - baseline["mean"])
    drift_detected = bool(ks_pvalue < threshold or mean_diff > config.DRIFT_MEAN_THRESHOLD)

    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "mean_diff": mean_diff,
        "current_mean": current_mean,
        "baseline_mean": baseline["mean"],
        "drift_detected": drift_detected,
        "threshold_ks": threshold,
        "threshold_mean": config.DRIFT_MEAN_THRESHOLD,
    }


def check_class_distribution_drift(
    predictions,
    baseline_stats: dict,
    rng: np.random.Generator | None = None,
) -> dict:
    """KS test of predicted labels against a sample drawn from the baseline class mix."""
    rng = rng or np.random.default_rng()
    threshold = config.DRIFT_KS_PVALUE_THRESHOLD
    if not baseline_stats or "class_distribution" not in baseline_stats:
        return _no_drift(threshold)

    baseline = baseline_stats["class_distribution"]
    baseline_total = baseline["total"]
    if len(predictions) == 0:
        return _no_drift(threshold)

    proportions = np.array([baseline["class_0_count"], baseline["class_1_count"]], dtype=float)
    proportions /= baseline_total
    baseline_samples = rng.choice([0, 1], size=baseline_total, p=proportions)
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, predictions)

    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "drift_detected": bool(ks_pvalue < threshold),
        "threshold_ks": threshold,
    }


def detect_drift(
    logs: list[dict],
    days: int = config.DRIFT_DEFAULT_DAYS,
    baseline_stats: dict | None = None,
    rng: np.random.Generator | None = None,
    now: datetime | None = None,
) -> dict:
    """Run both drift checks over the logs from the last ``days`` days."""
    recent = filter_logs_by_time(logs, days=days, now=now)
    if not recent:
        logger.warning("No inference logs found in the last %d days", days)
        return {
            "overall_drift_detected": False,
            "num_samples": 0,
            "confidence_drift": None,
            "class_distribution_drift": None,
            "error": "No logs found",
        }

    confidences = [float(e["confidence"]) for e in recent if e.get("confidence") is not None]
    predictions = [int(e["prediction"]) for e in recent if e.get("prediction") is not None]

    confidence_result = check_confidence_drift(confidences, baseline_stats or {}, rng=rng)
    class_result = check_class_distribution_drift(predictions, baseline_stats or {}, rng=rng)

    return {
        "overall_drift_detected": bool(
            confidence_result.get("drift_detected", False) or class_result.get("drift_detected", False)
        ),
        "num_samples": len(recent),
        "confidence_drift": confidence_result,
        "class_distribution_drift": class_result,
        "confidence_stats": calculate_confidence_metrics(confidences),
        "class_distribution": {
            "class_0_count": predictions.count(0),
            "class_1_count": predictions.count(1),
            "total": len(predictions),
        },
    }
