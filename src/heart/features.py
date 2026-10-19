"""Feature engineering and the preprocessing pipeline.

Everything that transforms raw patient fields into model inputs lives inside a
single scikit-learn ``Pipeline`` so the exact same code runs at training time,
in the notebooks, and inside the serving container.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart import config

ENGINEERED_NUMERIC = ["hr_reserve_pct", "chol_age_ratio"]


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds two domain-driven features.

    * ``hr_reserve_pct`` - peak heart rate achieved as a fraction of the
      age-predicted maximum (220 - age). Chronotropic incompetence (failing to
      reach ~85 % of predicted max) is a known cardiac risk marker.
    * ``chol_age_ratio`` - cholesterol normalised by age, so a high value in a
      young patient is not treated the same as in an elderly one.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        age = pd.to_numeric(X["age"], errors="coerce").astype(float)
        thalach = pd.to_numeric(X["thalach"], errors="coerce").astype(float)
        chol = pd.to_numeric(X["chol"], errors="coerce").astype(float)
        predicted_max = (220.0 - age).where(lambda s: s > 0)
        X["hr_reserve_pct"] = thalach / predicted_max
        X["chol_age_ratio"] = chol / age.where(age > 0)
        return X.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + ENGINEERED_NUMERIC, dtype=object)


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES + ENGINEERED_NUMERIC),
        ("bin", binary, config.BINARY_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([
        ("clinical", ClinicalFeatures()),
        ("columns", columns),
    ])


def build_model_pipeline(estimator) -> Pipeline:
    """Full pipeline: raw patient fields -> prediction."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])
