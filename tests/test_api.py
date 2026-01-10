from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app


def get_client():
    # Using TestClient as a context manager ensures FastAPI startup events run,
    # so the model is loaded for prediction tests.
    return TestClient(app)


def test_health() -> None:
    with get_client() as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root() -> None:
    with get_client() as client:
        r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "docs" in body


def test_predict_happy_path() -> None:
    payload = {
        "age": 30,
        "sex": "male",
        "bmi": 28.0,
        "children": 0,
        "smoker": "no",
        "region": "southeast",
    }
    with get_client() as client:
        r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], (int, float))


def test_predict_validation_error() -> None:
    payload = {
        "age": 30,
        "sex": "robot",
        "bmi": 28.0,
        "children": 0,
        "smoker": "no",
        "region": "southeast",
    }
    with get_client() as client:
        r = client.post("/predict", json=payload)
    # our validate_schema raises ValueError -> mapped to 400
    assert r.status_code == 400