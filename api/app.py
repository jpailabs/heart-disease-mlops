"""FastAPI service exposing the heart-disease classifier.

Endpoints
---------
GET  /health         liveness probe
GET  /ready          readiness probe (model loaded)
GET  /model-info     model metadata and metrics
POST /predict        single patient -> prediction + confidence
POST /predict/batch  list of patients
GET  /metrics        Prometheus metrics
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart import config  # noqa: E402
from heart.predict import load_metadata, load_model, predict_records  # noqa: E402


def settings() -> dict:
    """Read runtime settings from the environment (12-factor style)."""
    return {
        "model_path": Path(os.getenv("MODEL_PATH", config.MODEL_FILE)),
        "metadata_path": Path(os.getenv("METADATA_PATH", config.METADATA_FILE)),
        "threshold": float(os.getenv("DECISION_THRESHOLD", "0.5")),
    }


# ---------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
log = logging.getLogger("heart_api")
log.handlers = [handler]
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
log.propagate = False


# ---------------------------------------------------------------- metrics
REQUESTS = Counter("heart_api_requests_total", "HTTP requests",
                   ["method", "path", "status"])
LATENCY = Histogram("heart_api_request_latency_seconds", "Request latency",
                    ["path"], buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions served",
                      ["label", "risk_band"])
PROBABILITY = Histogram("heart_api_predicted_probability",
                        "Predicted probability of disease",
                        buckets=[i / 10 for i in range(1, 11)])
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 when the model is loaded")


# ---------------------------------------------------------------- schema
class Patient(BaseModel):
    """One patient record. Field names follow the UCI data dictionary."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 241, "fbs": 0,
        "restecg": 0, "thalach": 123, "exang": 1, "oldpeak": 0.2, "slope": 2,
        "ca": 0, "thal": 7}})

    age: int = Field(..., ge=1, le=120, description="age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="chest pain type")
    trestbps: float = Field(..., gt=50, lt=260, description="resting BP (mm Hg)")
    chol: float = Field(..., gt=80, lt=700, description="serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="resting ECG result")
    thalach: float = Field(..., gt=50, lt=250, description="max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="exercise-induced angina")
    oldpeak: float = Field(..., ge=-3, le=10, description="ST depression")
    slope: Literal[1, 2, 3] = Field(..., description="slope of peak ST segment")
    ca: Optional[int] = Field(None, ge=0, le=3, description="major vessels (0-3)")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed, 7 reversible")


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    label: str
    probability_disease: float
    confidence: float
    risk_band: str
    model_name: str
    model_version: str


class BatchRequest(BaseModel):
    patients: list[Patient] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


# ---------------------------------------------------------------- app
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = settings()
    state["threshold"] = cfg["threshold"]
    try:
        state["model"] = load_model(str(cfg["model_path"]))
        state["meta"] = load_metadata(cfg["metadata_path"])
        MODEL_LOADED.set(1)
        log.info("model loaded", extra={"extra_fields": {
            "model_path": str(cfg["model_path"]),
            "model_name": state["meta"].get("model_name")}})
    except Exception as exc:  # pragma: no cover - surfaced through /ready
        MODEL_LOADED.set(0)
        log.error("model load failed", extra={"extra_fields": {"error": str(exc)}})
    yield
    state.clear()


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(elapsed)
            log.info("request", extra={"extra_fields": {
                "request_id": request_id, "method": request.method,
                "path": path, "status": status,
                "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


def _require_model():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return state["model"]


def _run(patients: list[Patient]) -> list[Prediction]:
    model = _require_model()
    meta = state.get("meta", {})
    records = [p.model_dump() for p in patients]
    results = predict_records(records, model=model, threshold=state["threshold"])
    out = []
    for rec, res in zip(records, results):
        PREDICTIONS.labels(res["label"], res["risk_band"]).inc()
        PROBABILITY.observe(res["probability_disease"])
        log.info("prediction", extra={"extra_fields": {"input": rec, **res}})
        out.append(Prediction(**res,
                              model_name=meta.get("model_name", "unknown"),
                              model_version=meta.get("model_version", "unknown")))
    return out


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs",
            "endpoints": ["/health", "/ready", "/model-info", "/predict",
                          "/predict/batch", "/metrics"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    _require_model()
    return {"status": "ready"}


@app.get("/model-info")
def model_info():
    _require_model()
    return state.get("meta", {})


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    return _run([patient])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return BatchResponse(predictions=_run(req.patients))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
