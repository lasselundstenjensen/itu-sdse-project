from fastapi.testclient import TestClient

from serving.app import app

client = TestClient(app)


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
