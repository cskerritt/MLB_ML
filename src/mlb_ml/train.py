"""Train and evaluate the home-win-probability model.

Uses time-series cross-validation (no future leakage) and a final calibrated
XGBoost classifier saved to ``models/xgb_winprob.joblib``.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from .config import FEATURES_PARQUET, MODEL_PATH, RANDOM_STATE, TARGET_COL
from .features import FEATURE_COLS

log = logging.getLogger(__name__)


def _load() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PARQUET).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=[TARGET_COL])
    # Drop rows where the core team-form features are missing (early-season warmup).
    # Pitcher and bullpen columns are allowed to be NaN; XGBoost handles missing values.
    must_have = ["elo_diff", "home_win_pct_30", "away_win_pct_30"]
    df = df.dropna(subset=must_have)
    return df


def _available_features(df: pd.DataFrame) -> list[str]:
    """Use only FEATURE_COLS that actually exist in the parquet.

    Advanced-pitcher and bullpen columns only show up when Statcast was passed
    into ``build_features``. Silently dropping them here lets the same training
    script work on both base and Statcast-enriched tables.
    """
    return [c for c in FEATURE_COLS if c in df.columns]


def _base_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def cross_validate(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Walk-forward CV. Returns per-fold metrics."""
    cols = _available_features(df)
    X = df[cols].values
    y = df[TARGET_COL].values
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X), 1):
        model = _base_model()
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        rows.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "accuracy": accuracy_score(y[te], p > 0.5),
                "log_loss": log_loss(y[te], p),
                "brier": brier_score_loss(y[te], p),
                "auc": roc_auc_score(y[te], p),
            }
        )
    return pd.DataFrame(rows)


def fit_final(df: pd.DataFrame) -> CalibratedClassifierCV:
    """Fit a probability-calibrated model on all available data."""
    cols = _available_features(df)
    X = df[cols].values
    y = df[TARGET_COL].values
    base = _base_model()
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X, y)
    return cal


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = _load()
    log.info("Training on %s games (%s -> %s)", len(df), df["date"].min().date(), df["date"].max().date())

    cv = cross_validate(df)
    log.info("\n%s", cv.to_string(index=False))
    log.info("Mean accuracy=%.4f log_loss=%.4f brier=%.4f auc=%.4f",
             cv["accuracy"].mean(), cv["log_loss"].mean(), cv["brier"].mean(), cv["auc"].mean())

    model = fit_final(df)
    cols = _available_features(df)
    joblib.dump({"model": model, "feature_cols": cols}, MODEL_PATH)
    log.info("Saved model (%d features) to %s", len(cols), MODEL_PATH)


if __name__ == "__main__":
    main()
