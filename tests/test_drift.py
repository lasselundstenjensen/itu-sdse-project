from datetime import datetime, timezone

import numpy as np
import pytest
from scipy.stats import ks_2samp

from monitoring.drift import detect_drift


def test_empty_drift_response_has_notebook_keys():
    result = detect_drift([], baseline_stats={})

    assert set(result) == {
        "overall_drift_detected",
        "num_samples",
        "confidence_drift",
        "class_distribution_drift",
        "error",
    }
    assert result["error"] == "No logs found"


def test_nonempty_drift_response_has_notebook_keys():
    timestamp = datetime.now(timezone.utc).isoformat()
    logs = [{"confidence": 0.5, "prediction": 0, "timestamp": timestamp}]
    baseline = {
        "confidence_statistics": {"mean": 0.5, "std": 0.1},
        "class_distribution": {
            "class_0_count": 4,
            "class_1_count": 6,
            "total": 10,
        },
    }

    result = detect_drift(logs, baseline_stats=baseline)

    assert set(result) == {
        "overall_drift_detected",
        "num_samples",
        "confidence_drift",
        "class_distribution_drift",
        "confidence_stats",
        "class_distribution",
    }
    assert set(result["confidence_drift"]) == {
        "ks_statistic",
        "ks_pvalue",
        "drift_detected",
        "threshold_ks",
    }
    assert set(result["class_distribution_drift"]) == {
        "ks_statistic",
        "ks_pvalue",
        "drift_detected",
        "threshold_ks",
    }


def test_drift_metrics_match_notebook_for_seeded_sample():
    confidences = [0.12, 0.21, 0.33, 0.44, 0.58, 0.63, 0.71, 0.88]
    predictions = [0, 1, 0, 1, 1, 0, 1, 1]
    baseline_mean = 0.52
    baseline_std = 0.18
    baseline_total = 20
    baseline_distribution = [0.45, 0.55]
    timestamp = datetime.now(timezone.utc).isoformat()
    logs = [
        {
            "confidence": confidence,
            "prediction": prediction,
            "timestamp": timestamp,
        }
        for confidence, prediction in zip(confidences, predictions)
    ]
    baseline = {
        "confidence_statistics": {
            "mean": baseline_mean,
            "std": baseline_std,
        },
        "class_distribution": {
            "class_0_count": 9,
            "class_1_count": 11,
            "total": baseline_total,
        },
    }

    np.random.seed(42)
    notebook_confidence_sample = np.random.normal(
        baseline_mean, baseline_std, size=len(confidences)
    )
    confidence_ks_stat, confidence_ks_pvalue = ks_2samp(
        notebook_confidence_sample, confidences
    )
    notebook_class_sample = np.random.choice(
        [0, 1], size=baseline_total, p=baseline_distribution
    )
    class_ks_stat, class_ks_pvalue = ks_2samp(notebook_class_sample, predictions)

    np.random.seed(42)
    result = detect_drift(logs, baseline_stats=baseline)

    assert result["confidence_drift"]["ks_statistic"] == pytest.approx(
        confidence_ks_stat, abs=1e-12
    )
    assert result["confidence_drift"]["ks_pvalue"] == pytest.approx(
        confidence_ks_pvalue, abs=1e-12
    )
    assert result["confidence_stats"]["mean"] == pytest.approx(
        np.mean(confidences), abs=1e-12
    )
    assert result["confidence_stats"]["std"] == pytest.approx(
        np.std(confidences), abs=1e-12
    )
    assert result["class_distribution_drift"]["ks_statistic"] == pytest.approx(
        class_ks_stat, abs=1e-12
    )
    assert result["class_distribution_drift"]["ks_pvalue"] == pytest.approx(
        class_ks_pvalue, abs=1e-12
    )
