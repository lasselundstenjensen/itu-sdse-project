from datetime import datetime, timedelta, timezone

import numpy as np

from vial_classifier.monitoring import drift

BASELINE = {
    "confidence_statistics": {"mean": 0.5, "std": 0.05},
    "class_distribution": {"class_0_count": 50, "class_1_count": 50, "total": 100},
}


def make_log(confidence, prediction, days_ago=0):
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {"timestamp": ts.isoformat(), "prediction": prediction, "confidence": confidence}


def test_detect_drift_with_no_logs_reports_error():
    result = drift.detect_drift([], baseline_stats=BASELINE)
    assert result["error"] == "No logs found"
    assert result["overall_drift_detected"] is False
    assert result["num_samples"] == 0


def test_detect_drift_flags_shifted_confidences():
    np.random.seed(0)
    logs = [make_log(0.99, 1) for _ in range(50)]
    result = drift.detect_drift(logs, baseline_stats=BASELINE)
    assert result["confidence_drift"]["drift_detected"] is True
    assert result["overall_drift_detected"] is True


def test_detect_drift_flags_shifted_class_distribution():
    np.random.seed(0)
    result = drift.check_class_distribution_drift([1] * 50, BASELINE)
    assert result["drift_detected"] is True


def test_filter_logs_by_time_drops_old_entries():
    recent = make_log(0.5, 1, days_ago=1)
    old = make_log(0.5, 1, days_ago=30)
    assert drift.filter_logs_by_time([recent, old], days=7) == [recent]


def test_check_confidence_drift_without_baseline_reports_no_drift():
    result = drift.check_confidence_drift([0.5, 0.6], {})
    assert result["drift_detected"] is False
    assert result["ks_pvalue"] == 1.0


def test_load_inference_logs_missing_file_returns_empty(tmp_path):
    assert drift.load_inference_logs(tmp_path / "missing.json") == []
