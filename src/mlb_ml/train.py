"""Train and evaluate home-win-probability models.

Benchmarks every model in ``models.MODEL_REGISTRY`` with walk-forward
``TimeSeriesSplit`` CV, then fits the selected model (default: best by log
loss) on all data with isotonic calibration and saves it.
"""
from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .config import FEATURES_PARQUET, MODEL_PATH, TARGET_COL
from .features import FEATURE_COLS
from .models import MODEL_REGISTRY, get_model

log = logging.getLogger(__name__)

# Models that don't handle NaN natively need imputation at the X-matrix layer.
_NEEDS_IMPUTE = {"logit", "stacked"}


def _load() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PARQUET).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=[TARGET_COL])
    must_have = ["elo_diff", "home_win_pct_30", "away_win_pct_30"]
    df = df.dropna(subset=must_have)
    return df


def _available_features(df: pd.DataFrame) -> list[str]:
    """Use only FEATURE_COLS that actually exist in the parquet."""
    return [c for c in FEATURE_COLS if c in df.columns]


def _prep_matrix(df: pd.DataFrame, cols: list[str], model_name: str) -> np.ndarray:
    X = df[cols].values
    if model_name in _NEEDS_IMPUTE:
        X = SimpleImputer(strategy="median").fit_transform(X)
    return X


def _base_model():
    """Back-compat shim for backtest.py: the old default was XGBoost."""
    return get_model("xgb")


def cross_validate(df: pd.DataFrame, model_name: str, n_splits: int = 5) -> pd.DataFrame:
    """Walk-forward CV. Returns per-fold metrics for one model."""
    cols = _available_features(df)
    X = _prep_matrix(df, cols, model_name)
    y = df[TARGET_COL].values
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X), 1):
        model = get_model(model_name)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        rows.append({
            "model": model_name,
            "fold": fold,
            "accuracy": accuracy_score(y[te], p > 0.5),
            "log_loss": log_loss(y[te], p),
            "brier": brier_score_loss(y[te], p),
            "auc": roc_auc_score(y[te], p),
        })
    return pd.DataFrame(rows)


def benchmark(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Cross-validate every registered model and return a leaderboard."""
    all_rows = []
    for name in MODEL_REGISTRY:
        log.info("Cross-validating %s", name)
        all_rows.append(cross_validate(df, name, n_splits=n_splits))
    per_fold = pd.concat(all_rows, ignore_index=True)
    leaderboard = (
        per_fold.groupby("model")[["accuracy", "log_loss", "brier", "auc"]]
        .mean()
        .sort_values("log_loss")
        .reset_index()
    )
    return leaderboard


def fit_final(df: pd.DataFrame, model_name: str) -> CalibratedClassifierCV:
    """Fit a probability-calibrated version of the chosen model."""
    cols = _available_features(df)
    X = _prep_matrix(df, cols, model_name)
    y = df[TARGET_COL].values
    base = get_model(model_name)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X, y)
    return cal


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="auto",
                        help=f"One of {sorted(MODEL_REGISTRY)} or 'auto' to pick best by log loss")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    df = _load()
    log.info("Training on %s games (%s -> %s)", len(df),
             df["date"].min().date(), df["date"].max().date())

    leaderboard = benchmark(df, n_splits=args.n_splits)
    print("\n== leaderboard ==")
    print(leaderboard.to_string(index=False))

    choice = args.model if args.model != "auto" else leaderboard.iloc[0]["model"]
    log.info("Fitting final model: %s", choice)
    model = fit_final(df, choice)
    cols = _available_features(df)
    joblib.dump(
        {"model": model, "feature_cols": cols, "model_name": choice},
        MODEL_PATH,
    )
    log.info("Saved %s (%d features) to %s", choice, len(cols), MODEL_PATH)


if __name__ == "__main__":
    main()
