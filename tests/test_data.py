import numpy as np
import pandas as pd

from heart import config
from heart.data import clean, load_raw, split_xy


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert "num" not in out.columns
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert (out[config.TARGET] == (raw_df["num"] > 0).astype(int)).all()


def test_clean_keeps_missing_for_pipeline(raw_df):
    out = clean(raw_df)
    assert out["ca"].isna().sum() == 2
    assert out["thal"].isna().sum() == 1


def test_clean_nulls_out_of_range_codes(raw_df):
    raw_df.loc[0, "cp"] = 9
    raw_df.loc[2, "thal"] = 5
    raw_df.loc[4, "chol"] = 0
    out = clean(raw_df)
    assert pd.isna(out.loc[0, "cp"])
    assert pd.isna(out.loc[2, "thal"])
    assert pd.isna(out.loc[4, "chol"])


def test_clean_drops_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    assert len(clean(doubled)) == len(raw_df)


def test_load_raw_parses_question_marks(tmp_path):
    p = tmp_path / "raw.data"
    p.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n")
    df = load_raw(p)
    assert list(df.columns) == config.RAW_COLUMNS
    assert np.isnan(df.loc[0, "ca"])


def test_split_xy(raw_df):
    X, y = split_xy(clean(raw_df))
    assert list(X.columns) == config.INPUT_FEATURES
    assert len(X) == len(y)
