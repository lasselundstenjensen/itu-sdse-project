import json

import pytest
from fastapi.testclient import TestClient

import serving.app as app_module
from serving.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def temp_log(tmp_path, monkeypatch):
    # tests write their log into a temporary folder, not into the repo
    monkeypatch.setattr(app_module, "INFERENCE_LOG_PATH", tmp_path / "inference_log.json")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()


def test_infer_valid_image():
    response = client.post("/infer", json={"image": "aGVsbG8="})
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_infer_invalid_image():
    response = client.post("/infer", json={"image": "not an image!!"})
    assert response.status_code == 400


def test_invocations_works_like_infer():
    response = client.post("/invocations", json={"image": "aGVsbG8="})
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_infer_writes_log_entry(tmp_path):
    client.post("/infer", json={"image": "aGVsbG8="})
    logs = json.loads((tmp_path / "inference_log.json").read_text())
    assert len(logs) == 1
    assert logs[0]["prediction"] in (0, 1)
    assert 0 <= logs[0]["confidence"] <= 1