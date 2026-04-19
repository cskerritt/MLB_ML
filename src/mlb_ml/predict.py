"""Predict win probabilities for upcoming games.

Loads the trained model and the latest feature snapshot, then prints a
home-win probability for each game on the requested date.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .config import FEATURES_PARQUET, MODEL_PATH

log = logging.getLogger(__name__)

_NEEDS_IMPUTE = {"logit", "stacked"}


def _load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"], bundle.get("model_name", "xgb")


def predict_for_date(target: date) -> pd.DataFrame:
    model, feature_cols, model_name = _load_model()
    feats = pd.read_parquet(FEATURES_PARQUET)
    feats["date"] = pd.to_datetime(feats["date"]).dt.date
    games = feats[feats["date"] == target].copy()
    if games.empty:
        raise SystemExit(f"No games found in features.parquet for {target}.")

    X = games[feature_cols].values
    if model_name in _NEEDS_IMPUTE:
        X = SimpleImputer(strategy="median").fit_transform(X)
    proba = model.predict_proba(X)[:, 1]
    games["home_win_prob"] = proba
    games["pick"] = ["HOME" if p >= 0.5 else "AWAY" for p in proba]
    return games[["date", "away_team", "home_team", "home_win_prob", "pick"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=date.today().isoformat(),
                   help="YYYY-MM-DD; defaults to today")
    args = p.parse_args()
    target = date.fromisoformat(args.date)
    out = predict_for_date(target)
    pd.set_option("display.max_rows", None)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
