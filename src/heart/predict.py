"""Inference helpers shared by the API, notebooks and CLI.

CLI:  python -m heart.predict '{"age": 63, "sex": 1, ...}'
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from heart import config

RISK_BANDS = [(0.3, "low"), (0.6, "moderate"), (1.01, "high")]


@lru_cache(maxsize=2)
def load_model(path: str | Path = config.MODEL_FILE):
    return joblib.load(path)


def load_metadata(path: Path = config.METADATA_FILE) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def risk_band(probability: float) -> str:
    for upper, label in RISK_BANDS:
        if probability < upper:
            return label
    return "high"


def predict_records(records: list[dict], model=None,
                    threshold: float = 0.5) -> list[dict]:
    model = model or load_model()
    frame = pd.DataFrame(records)
    missing = [c for c in config.INPUT_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"missing features: {missing}")
    X = frame[config.INPUT_FEATURES].apply(pd.to_numeric, errors="coerce")
    proba = model.predict_proba(X.astype(float))[:, 1]
    out = []
    for p in proba:
        pred = int(p >= threshold)
        out.append({
            "prediction": pred,
            "label": "disease" if pred else "no_disease",
            "probability_disease": round(float(p), 4),
            "confidence": round(float(p if pred else 1 - p), 4),
            "risk_band": risk_band(float(p)),
        })
    return out


if __name__ == "__main__":
    payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else json.load(sys.stdin)
    records = payload if isinstance(payload, list) else [payload]
    print(json.dumps(predict_records(records), indent=2))
