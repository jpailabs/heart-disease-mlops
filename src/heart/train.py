"""Model training, tuning, evaluation and MLflow experiment tracking.

Run:  python -m heart.train
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)

from heart import __version__, config  # noqa: E402
from heart.data import build_clean_dataset, load_clean, split_xy  # noqa: E402
from heart.features import build_model_pipeline  # noqa: E402

log = logging.getLogger("heart.train")

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# Search spaces. Kept deliberately small: the dataset has ~300 rows, so wide
# grids mostly overfit the CV folds.
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, solver="liblinear",
                           random_state=config.RANDOM_STATE),
        {
            "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [1, 3, 5],
            "model__max_features": ["sqrt", 0.5],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=config.RANDOM_STATE),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
    ),
}


def test_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred, zero_division=0),
        "test_f1": f1_score(y_true, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_true, y_proba),
    }


def feature_importance(pipe) -> pd.Series | None:
    model = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].named_steps["columns"] \
        .get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    else:
        return None
    return pd.Series(values, index=names).sort_values(ascending=False)


def save_plots(name, pipe, X_test, y_test, grid, out_dir: Path) -> list[Path]:
    paths = []

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, display_labels=["No disease", "Disease"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{name}: confusion matrix (test)")
    fig.tight_layout()
    paths.append(out_dir / f"{name}_confusion_matrix.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title(f"{name}: ROC curve (test)")
    fig.tight_layout()
    paths.append(out_dir / f"{name}_roc_curve.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    imp = feature_importance(pipe)
    if imp is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        imp.head(15)[::-1].plot.barh(ax=ax, color="#3b7dd8")
        ax.set_title(f"{name}: top feature importances")
        fig.tight_layout()
        paths.append(out_dir / f"{name}_feature_importance.png")
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig)

    cv = pd.DataFrame(grid.cv_results_)
    cv_path = out_dir / f"{name}_cv_results.csv"
    cv.to_csv(cv_path, index=False)
    paths.append(cv_path)
    return paths


def train_candidate(name, estimator, grid_params, X_train, y_train, X_test,
                    y_test, cv, artifacts_dir: Path) -> dict:
    pipe = build_model_pipeline(estimator)
    grid = GridSearchCV(pipe, grid_params, scoring="roc_auc", cv=cv,
                        n_jobs=-1, refit=True, return_train_score=True)

    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({"model_family": name, "stage": "tuning",
                         "code_version": __version__})
        grid.fit(X_train, y_train)

        # One nested run per grid point so every configuration is traceable.
        res = grid.cv_results_
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"{name}-{i:02d}", nested=True):
                mlflow.log_params({k.replace("model__", ""): v
                                   for k, v in params.items()})
                mlflow.log_metrics({
                    "cv_roc_auc_mean": res["mean_test_score"][i],
                    "cv_roc_auc_std": res["std_test_score"][i],
                    "train_roc_auc_mean": res["mean_train_score"][i],
                })

        best = grid.best_estimator_
        cv_scores = cross_validate(best, X_train, y_train, cv=cv,
                                   scoring=SCORING, n_jobs=-1)
        cv_summary = {f"cv_{k}_mean": float(np.mean(cv_scores[f"test_{k}"]))
                      for k in SCORING}
        cv_summary.update({f"cv_{k}_std": float(np.std(cv_scores[f"test_{k}"]))
                           for k in SCORING})

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        holdout = test_metrics(y_test, y_pred, y_proba)

        mlflow.log_params({k.replace("model__", "best_"): v
                           for k, v in grid.best_params_.items()})
        mlflow.log_param("n_grid_points", len(res["params"]))
        mlflow.log_metrics({**cv_summary, **holdout})

        model_dir = artifacts_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        for p in save_plots(name, best, X_test, y_test, grid, model_dir):
            mlflow.log_artifact(str(p), artifact_path="plots"
                                if p.suffix == ".png" else "tables")

        # Declare every input as float so optional/missing fields are valid.
        X_sig = X_test.astype(float)
        signature = infer_signature(X_sig, best.predict_proba(X_sig))
        mlflow.sklearn.log_model(best, artifact_path="model",
                                 signature=signature,
                                 input_example=X_sig.head(3))

        log.info("%-20s cv_auc=%.4f test_auc=%.4f test_acc=%.4f",
                 name, cv_summary["cv_roc_auc_mean"], holdout["test_roc_auc"],
                 holdout["test_accuracy"])

        return {
            "name": name,
            "run_id": parent.info.run_id,
            "best_params": grid.best_params_,
            "estimator": best,
            **cv_summary,
            **holdout,
        }


def comparison_plot(results: list[dict], out: Path) -> Path:
    df = pd.DataFrame(results).set_index("name")
    cols = ["cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean",
            "cv_f1_mean", "cv_roc_auc_mean"]
    ax = df[cols].rename(columns=lambda c: c[3:-5]).T.plot.bar(
        figsize=(8, 4.5), rot=0, ylim=(0.5, 1.0))
    ax.set_title("Model comparison - 5-fold CV on training split")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    return out


def _setup_logging():
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("heart", "heart.train"):
        logging.getLogger(name).disabled = False
    logging.getLogger("alembic").setLevel(logging.WARNING)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=config.MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment", default=config.MLFLOW_EXPERIMENT)
    parser.add_argument("--models", nargs="*", default=list(CANDIDATES),
                        choices=list(CANDIDATES))
    parser.add_argument("--model-dir", type=Path, default=config.MODELS_DIR,
                        help="where to write heart_model.joblib + metadata")
    parser.add_argument("--register", action="store_true",
                        help="register the winning model in the MLflow registry")
    args = parser.parse_args(argv)

    _setup_logging()

    if not config.CLEAN_FILE.exists():
        build_clean_dataset()
    df = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    _setup_logging()  # MLflow's sqlite migrations (alembic) reset logging config
    log.info("Tracking URI: %s | train=%d test=%d", args.tracking_uri,
             len(X_train), len(X_test))

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in args.models:
            est, params = CANDIDATES[name]
            results.append(train_candidate(name, est, params, X_train, y_train,
                                           X_test, y_test, cv, tmp))

        # Selection rule: highest mean CV ROC-AUC (never the test score, to
        # keep the hold-out set honest). Ties broken by CV recall, because a
        # missed disease case is costlier than a false alarm.
        winner = max(results, key=lambda r: (round(r["cv_roc_auc_mean"], 3),
                                             r["cv_recall_mean"]))

        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        comp = comparison_plot(results, config.FIGURES_DIR / "model_comparison.png")
        with mlflow.start_run(run_name="model_selection"):
            mlflow.set_tag("stage", "selection")
            mlflow.log_param("selected_model", winner["name"])
            mlflow.log_param("selected_run_id", winner["run_id"])
            mlflow.log_metrics({k: v for k, v in winner.items()
                                if k.startswith(("cv_", "test_"))})
            mlflow.log_artifact(str(comp), artifact_path="plots")
            table = pd.DataFrame([{k: v for k, v in r.items()
                                   if k not in ("estimator",)} for r in results])
            table_path = tmp / "model_comparison.csv"
            table.to_csv(table_path, index=False)
            mlflow.log_artifact(str(table_path), artifact_path="tables")

        # Copy per-model plots of all candidates to reports/figures for the report
        for r in results:
            for png in (tmp / r["name"]).glob("*.png"):
                (config.FIGURES_DIR / png.name).write_bytes(png.read_bytes())

    if args.register:
        mlflow.register_model(f"runs:/{winner['run_id']}/model",
                              "heart-disease-classifier")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_file = args.model_dir / config.MODEL_FILE.name
    joblib.dump(winner["estimator"], model_file)
    metadata = {
        "model_name": winner["name"],
        "model_version": __version__,
        "mlflow_run_id": winner["run_id"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_params": {k.replace("model__", ""): v
                        for k, v in winner["best_params"].items()},
        "metrics": {k: round(float(v), 4) for k, v in winner.items()
                    if k.startswith(("cv_", "test_"))},
        "features": config.INPUT_FEATURES,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "all_models": [{"name": r["name"],
                        "cv_roc_auc_mean": round(r["cv_roc_auc_mean"], 4),
                        "test_roc_auc": round(r["test_roc_auc"], 4),
                        "test_accuracy": round(r["test_accuracy"], 4)}
                       for r in results],
    }
    (args.model_dir / config.METADATA_FILE.name).write_text(
        json.dumps(metadata, indent=2, default=str))
    log.info("Selected %s -> %s", winner["name"], model_file)
    return metadata


if __name__ == "__main__":
    main()
