import numpy as np
import pandas as pd
import pytest

from heart import config


@pytest.fixture
def raw_df():
    """Small synthetic frame in the raw UCI layout (num = 0..4)."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 180, n),
        "chol": rng.integers(150, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 4, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
        "num": rng.integers(0, 5, n),
    })
    df.loc[[1, 5], "ca"] = np.nan
    df.loc[3, "thal"] = np.nan
    return df[config.RAW_COLUMNS]


@pytest.fixture
def patient():
    return {"age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 241,
            "fbs": 0, "restecg": 0, "thalach": 123, "exang": 1,
            "oldpeak": 0.2, "slope": 2, "ca": 0, "thal": 7}
