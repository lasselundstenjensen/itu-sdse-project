import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vial_classifier.serving import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    # Point the scorer at a port nothing listens on, so /health reports "loading"
    app = create_app(
        model_dir=tmp_path,
        scoring_port=1,
        inference_log_path=tmp_path / "inference_log.json",
        baseline_path=tmp_path / "drift_baseline_stats.json",
    )
    return TestClient(app)


def test_health_is_loading_while_scorer_is_unreachable(client):
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_version_defaults_to_unknown(client):
    assert client.get("/version").json() == {"model_version": "unknown", "trained_at": None}


def test_version_reads_version_file(tmp_path):
    (tmp_path / "version.json").write_text('{"model_version": "3", "trained_at": "2026-01-01T00:00:00+00:00"}')
    app = create_app(model_dir=tmp_path, scoring_port=1,
                     inference_log_path=tmp_path / "log.json", baseline_path=tmp_path / "b.json")
    assert TestClient(app).get("/version").json()["model_version"] == "3"


def test_drift_without_logs_reports_no_logs(client):
    body = client.get("/drift").json()
    assert body["overall_drift_detected"] is False
    assert body["error"] == "No logs found"


def test_invocations_returns_502_when_scorer_is_down(client):
    response = client.post("/invocations", json={"instances": []})
    assert response.status_code == 502
