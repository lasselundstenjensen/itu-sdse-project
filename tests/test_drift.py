import json

from vials.monitoring.drift import load_inference_logs


def test_load_inference_logs_returns_empty_list_when_file_is_missing(tmp_path):
    log_path = tmp_path / "inference_log.json"

    result = load_inference_logs(log_path)

    assert result == []


def test_load_inference_logs_returns_empty_list_when_json_is_broken(tmp_path):
    log_path = tmp_path / "inference_log.json"
    log_path.write_text("{not valid json")

    result = load_inference_logs(log_path)

    assert result == []


def test_load_inference_logs_returns_empty_list_when_json_is_not_a_list(tmp_path):
    log_path = tmp_path / "inference_log.json"
    log_path.write_text(json.dumps({"prediction": 1}))

    result = load_inference_logs(log_path)

    assert result == []


def test_load_inference_logs_returns_logs_from_valid_file(tmp_path):
    log_path = tmp_path / "inference_log.json"
    logs = [{"prediction": 1, "confidence": 0.9}]
    log_path.write_text(json.dumps(logs))

    result = load_inference_logs(log_path)

    assert result == logs
