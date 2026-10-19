import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from heart.data import clean, split_xy
from heart.features import ClinicalFeatures, build_model_pipeline, build_preprocessor


def test_clinical_features_values():
    X = pd.DataFrame({"age": [60, 40], "thalach": [120, 180], "chol": [240, 200]})
    out = ClinicalFeatures().transform(X)
    assert np.isclose(out.loc[0, "hr_reserve_pct"], 120 / 160)
    assert np.isclose(out.loc[1, "chol_age_ratio"], 5.0)


def test_clinical_features_handles_bad_age():
    X = pd.DataFrame({"age": [0], "thalach": [150], "chol": [200]})
    out = ClinicalFeatures().transform(X)
    assert np.isnan(out.loc[0, "chol_age_ratio"])


def test_preprocessor_no_nans_and_scaled(raw_df):
    X, _ = split_xy(clean(raw_df))
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(Xt).any()
    # first column is a standard-scaled numeric feature
    assert abs(Xt[:, 0].mean()) < 1e-6


def test_preprocessor_handles_unseen_category(raw_df):
    X, _ = split_xy(clean(raw_df))
    pre = build_preprocessor().fit(X)
    X_new = X.head(1).copy()
    X_new["cp"] = 99
    assert pre.transform(X_new).shape[1] == pre.transform(X.head(1)).shape[1]


def test_pipeline_fit_predict_proba(raw_df):
    X, y = split_xy(clean(raw_df))
    pipe = build_model_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)
