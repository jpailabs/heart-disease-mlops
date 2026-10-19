import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from heart.data import clean, split_xy
from heart.features import build_model_pipeline


@pytest.fixture
def client(raw_df, tmp_path, monkeypatch):
    X, y = split_xy(clean(raw_df))
    path = tmp_path / "model.joblib"
    joblib.dump(build_model_pipeline(LogisticRegression(max_iter=500)).fit(X, y), path)
    (tmp_path / "meta.json").write_text('{"model_name": "test", "model_version": "0"}')
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("METADATA_PATH", str(tmp_path / "meta.json"))
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_predict_returns_prediction_and_confidence(client, patient):
    r = client.post("/predict", json=patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert body["model_name"] == "test"
    assert "x-request-id" in r.headers


def test_predict_validation_error(client, patient):
    patient["age"] = -5
    assert client.post("/predict", json=patient).status_code == 422
    patient.pop("age")
    assert client.post("/predict", json=patient).status_code == 422


def test_batch(client, patient):
    r = client.post("/predict/batch", json={"patients": [patient, patient]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_metrics_exposed(client, patient):
    client.post("/predict", json=patient)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text
