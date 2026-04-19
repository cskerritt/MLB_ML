"""Optuna hyperparameter search for the tree-based models.

Uses walk-forward ``TimeSeriesSplit`` CV on the features parquet and
minimises mean log loss. Best params are written to ``models/best_params.json``
so ``models.build_xgb`` / ``build_lgbm`` pick them up on the next train run.
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from .config import BEST_PARAMS_PATH, FEATURES_PARQUET, RANDOM_STATE, TARGET_COL
from .features import FEATURE_COLS
from .train import _available_features, _load

log = logging.getLogger(__name__)


def _score(model, X, y, n_splits: int) -> float:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    losses = []
    for tr, te in splitter.split(X):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        losses.append(log_loss(y[te], np.clip(p, 1e-4, 1 - 1e-4)))
    return float(np.mean(losses))


def _xgb_objective(trial, X, y, n_splits):
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 7),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
    )
    model = XGBClassifier(
        **params,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return _score(model, X, y, n_splits)


def _lgbm_objective(trial, X, y, n_splits):
    from lightgbm import LGBMClassifier

    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
    )
    model = LGBMClassifier(
        **params,
        objective="binary", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    return _score(model, X, y, n_splits)


_OBJECTIVES = {"xgb": _xgb_objective, "lgbm": _lgbm_objective}


def tune(model_name: str, n_trials: int = 40, n_splits: int = 4) -> dict[str, Any]:
    import optuna

    if model_name not in _OBJECTIVES:
        raise ValueError(
            f"Tuning only supports {sorted(_OBJECTIVES)}; got {model_name!r}"
        )

    df = _load()
    cols = _available_features(df)
    X = df[cols].values
    y = df[TARGET_COL].values
    log.info("Tuning %s over %d trials on %d games", model_name, n_trials, len(df))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda t: _OBJECTIVES[model_name](t, X, y, n_splits),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    log.info("Best log loss for %s: %.4f", model_name, study.best_value)
    return study.best_params


def save_best(params_by_model: dict[str, dict[str, Any]]) -> None:
    existing = {}
    if BEST_PARAMS_PATH.exists():
        try:
            existing = json.loads(BEST_PARAMS_PATH.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(params_by_model)
    BEST_PARAMS_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True))
    log.info("Wrote %s", BEST_PARAMS_PATH)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(_OBJECTIVES), default="lgbm")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--n-splits", type=int, default=4)
    args = p.parse_args()

    best = tune(args.model, n_trials=args.trials, n_splits=args.n_splits)
    save_best({args.model: best})
    print(json.dumps({args.model: best}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
