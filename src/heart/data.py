"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from heart import config

log = logging.getLogger(__name__)

# Valid code ranges taken from the UCI data dictionary (heart-disease.names).
VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def download_raw(dest: Path = config.RAW_FILE, url: str = config.UCI_URL,
                 force: bool = False) -> Path:
    """Download the raw Cleveland file from the UCI repository."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.info("Raw file already present at %s", dest)
        return dest
    log.info("Downloading %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    log.info("Saved %d bytes to %s", len(resp.content), dest)
    return dest


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the header-less raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS,
                       na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw dataframe.

    * coerce every column to numeric
    * drop exact duplicate rows
    * null-out codes outside the documented ranges (treated as missing,
      imputed later inside the model pipeline so train/serve stay identical)
    * binarise the 0-4 ``num`` diagnosis into ``target`` (0 = no disease,
      1 = disease present)
    """
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) != before:
        log.info("Dropped %d duplicate rows", before - len(df))

    for col, valid in VALID_CODES.items():
        bad = df[col].notna() & ~df[col].isin(valid)
        if bad.any():
            log.warning("%s: %d out-of-range codes set to NaN", col, bad.sum())
            df.loc[bad, col] = np.nan

    # Physiologically impossible values -> missing
    df.loc[df["trestbps"] <= 0, "trestbps"] = np.nan
    df.loc[df["chol"] <= 0, "chol"] = np.nan

    df = df.dropna(subset=["num"])
    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    int_cols = ["age", "sex", "cp", "fbs", "restecg", "exang", "slope"]
    df[int_cols] = df[int_cols].astype("Int64")
    return df


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def build_clean_dataset(force_download: bool = False) -> pd.DataFrame:
    """Download (if needed), clean and persist the dataset."""
    raw_path = download_raw(force=force_download)
    df = clean(load_raw(raw_path))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    log.info("Clean dataset: %d rows x %d cols -> %s", *df.shape,
             config.CLEAN_FILE)
    return df


def split_xy(df: pd.DataFrame):
    return df[config.INPUT_FEATURES], df[config.TARGET].astype(int)
