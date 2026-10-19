"""Model training/selection tests on synthetic data (fast, no network)."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from heart import train
from heart.data import clean, split_xy
from heart.features import build_model_pipeline
from heart.predict import predict_records, risk_band


def _fitted(raw_df):
    X, y = split_xy(clean(raw_df))
    return build_model_pipeline(LogisticRegression(max_iter=500)).fit(X, y), X, y


def test_test_metrics_keys():
    y = np.array([0, 1, 1, 0])
    m = train.test_metrics(y, y, np.array([.1, .9, .8, .2]))
    assert m["test_accuracy"] == 1.0 and m["test_roc_auc"] == 1.0


def test_feature_importance_for_linear_model(raw_df):
    pipe, _, _ = _fitted(raw_df)
    imp = train.feature_importance(pipe)
    assert imp is not None and (imp >= 0).all()
    assert "hr_reserve_pct" in imp.index


def test_train_candidate_logs_to_mlflow(raw_df, tmp_path):
    import mlflow
    X, y = split_xy(clean(raw_df))
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    mlflow.set_experiment("unit-test")
    est, _ = train.CANDIDATES["logistic_regression"]
    res = train.train_candidate(
        "logistic_regression", est, {"model__C": [0.1, 1.0]}, X, y, X, y,
        StratifiedKFold(3, shuffle=True, random_state=0), tmp_path)
    assert 0 <= res["cv_roc_auc_mean"] <= 1
    run = mlflow.get_run(res["run_id"])
    assert "test_roc_auc" in run.data.metrics
    assert run.data.params["n_grid_points"] == "2"


def test_model_roundtrip_and_predict(raw_df, tmp_path, patient):
    pipe, X, _ = _fitted(raw_df)
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    assert np.allclose(loaded.predict_proba(X), pipe.predict_proba(X))
    out = predict_records([patient], model=loaded)[0]
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1


def test_predict_accepts_missing_optional(raw_df, patient):
    pipe, _, _ = _fitted(raw_df)
    patient.update(ca=None, thal=None)
    assert predict_records([patient], model=pipe)[0]["prediction"] in (0, 1)


def test_risk_band():
    assert risk_band(0.1) == "low"
    assert risk_band(0.45) == "moderate"
    assert risk_band(0.95) == "high"
