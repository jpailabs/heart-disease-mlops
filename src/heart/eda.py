"""Exploratory data analysis: produces the figures used in the report.

Run:  python -m heart.eda
"""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart import config  # noqa: E402
from heart.data import build_clean_dataset, load_clean, load_raw  # noqa: E402

log = logging.getLogger(__name__)

LABELS = {
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "sex": {0: "female", 1: "male"},
    "exang": {0: "no", 1: "yes"},
}
PALETTE = {0: "#4c9be8", 1: "#e8604c"}


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path)
    return path


def class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = df["target"].value_counts().sort_index()
    bars = ax.bar(["No disease (0)", "Disease (1)"], counts.values,
                  color=[PALETTE[0], PALETTE[1]])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2,
                f"{v} ({v / counts.sum():.0%})", ha="center")
    ax.set_ylabel("patients")
    ax.set_title("Class balance")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, "eda_class_balance.png")


def histograms(df):
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(data=df, x=col, hue="target", kde=True, ax=ax,
                     palette=PALETTE, element="step", stat="density",
                     common_norm=False)
        ax.set_title(col)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric feature distributions by diagnosis", y=1.01)
    fig.tight_layout()
    return _save(fig, "eda_histograms.png")


def correlation(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.75}, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Pearson correlation matrix")
    return _save(fig, "eda_correlation_heatmap.png")


def categorical_rates(df):
    cols = ["cp", "thal", "slope", "sex", "exang", "ca"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cols):
        tmp = df[[col, "target"]].dropna().copy()
        tmp[col] = tmp[col].astype(int).map(LABELS.get(col, {})).fillna(
            tmp[col].astype(int).astype(str))
        rate = tmp.groupby(col)["target"].agg(["mean", "size"]).reset_index()
        ax.bar(rate[col].astype(str), rate["mean"], color="#e8604c")
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Disease rate by {col}")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _save(fig, "eda_categorical_disease_rate.png")


def boxplots(df):
    cols = ["age", "thalach", "oldpeak", "chol"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x="target", y=col, hue="target", ax=ax,
                    palette=PALETTE, legend=False)
        ax.set_xticks([0, 1], ["no", "yes"])
        ax.set_xlabel("disease")
    fig.suptitle("Outlier check - numeric features by diagnosis")
    fig.tight_layout()
    return _save(fig, "eda_boxplots.png")


def missing_values(raw):
    miss = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(miss.index, miss.values, color="#888")
    for i, v in enumerate(miss.values):
        if v:
            ax.text(i, v + 0.1, str(v), ha="center")
    ax.set_title("Missing values in raw file ('?' markers)")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "eda_missing_values.png")


def summary(df, raw) -> dict:
    return {
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(df)),
        "n_features": len(config.INPUT_FEATURES),
        "positive_rate": round(float(df["target"].mean()), 4),
        "missing_raw": {k: int(v) for k, v in raw.isna().sum().items() if v},
        "top_target_correlations": df.corr(numeric_only=True)["target"]
        .drop("target").abs().sort_values(ascending=False).head(6)
        .round(3).to_dict(),
        "describe": df.describe().round(2).to_dict(),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not config.CLEAN_FILE.exists():
        build_clean_dataset()
    df = load_clean()
    raw = load_raw()
    sns.set_theme(style="whitegrid")
    for fn in (class_balance, histograms, correlation, categorical_rates,
               boxplots):
        fn(df)
    missing_values(raw)
    stats = summary(df, raw)
    (config.FIGURES_DIR / "eda_summary.json").write_text(
        json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "describe"},
                     indent=2))
    return stats


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
