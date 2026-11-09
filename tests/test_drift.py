from datetime import datetime, timedelta, timezone

import numpy as np

from vial_classifier.drift import (
    calculate_confidence_metrics,
    check_class_distribution_drift,
    check_confidence_drift,
    detect_drift,
    filter_logs_by_time,
)

NOW = datetime(2026, 1, 10, tzinfo=timezone.utc)


def _entry(days_ago: float, confidence: float, prediction: int) -> dict:
    stamp = (NOW - timedelta(days=days_ago)).isoformat()
    return {"timestamp": stamp, "confidence": confidence, "prediction": prediction}


BASELINE = {
    "confidence_statistics": {"mean": 0.5, "std": 0.2},
    "class_distribution": {"class_0_count": 50, "class_1_count": 50, "total": 100},
}


def test_filter_keeps_recent_and_drops_old_and_untimed():
    logs = [
        _entry(1, 0.1, 0),
        _entry(30, 0.1, 0),
        {"confidence": 0.1, "prediction": 0},  # no timestamp -> dropped
        {"timestamp": "not-a-date", "confidence": 0.1, "prediction": 0},  # unparseable -> kept
    ]
    kept = filter_logs_by_time(logs, days=7, now=NOW)
    assert len(kept) == 2
    assert kept[0]["timestamp"] == logs[0]["timestamp"]
    assert kept[1]["timestamp"] == "not-a-date"


def test_confidence_metrics_handles_empty():
    assert calculate_confidence_metrics([]) == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


def test_no_drift_when_distribution_matches_baseline():
    rng = np.random.default_rng(0)
    confidences = rng.normal(0.5, 0.2, size=500).tolist()
    result = check_confidence_drift(confidences, BASELINE, rng=np.random.default_rng(1))
    assert result["drift_detected"] is False


def test_confidence_drift_detected_on_shifted_scores():
    confidences = [0.95] * 200
    result = check_confidence_drift(confidences, BASELINE, rng=np.random.default_rng(1))
    assert result["drift_detected"] is True
    assert result["mean_diff"] > 0.1


def test_class_drift_detected_when_all_predictions_are_one_class():
    result = check_class_distribution_drift([1] * 200, BASELINE, rng=np.random.default_rng(2))
    assert result["drift_detected"] is True


def test_missing_baseline_never_reports_drift():
    assert check_confidence_drift([0.1, 0.9], {})["drift_detected"] is False
    assert check_class_distribution_drift([0, 1], {})["drift_detected"] is False


def test_detect_drift_without_logs_reports_error():
    result = detect_drift([], days=7, baseline_stats=BASELINE, now=NOW)
    assert result["overall_drift_detected"] is False
    assert result["error"] == "No logs found"


def test_detect_drift_summarises_recent_logs():
    logs = [_entry(0.5, 0.4, 0), _entry(0.5, 0.6, 1)]
    result = detect_drift(logs, days=7, baseline_stats=BASELINE,
                          rng=np.random.default_rng(3), now=NOW)
    assert result["num_samples"] == 2
    assert result["class_distribution"] == {"class_0_count": 1, "class_1_count": 1, "total": 2}
