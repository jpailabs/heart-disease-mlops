"""Download the UCI Heart Disease (Cleveland) data and write the cleaned CSV.

Usage:  python scripts/download_data.py [--force]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart.data import build_clean_dataset  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download the raw file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = build_clean_dataset(force_download=args.force)
    print(f"rows={len(df)} cols={df.shape[1]} positives={int(df['target'].sum())}")
    print("missing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0].to_string())
