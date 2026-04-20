"""Model zoo: XGBoost, LightGBM, logistic, and a stacked ensemble.

Each builder returns a fresh, unfit estimator. Optuna-tuned hyperparameters
in ``models/best_params.json`` are picked up automatically when they exist
for the relevant estimator (xgb / lgbm).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import BEST_PARAMS_PATH, RANDOM_STATE

log = logging.getLogger(__name__)


def _load_best_params(name: str) -> dict[str, Any]:
    if not BEST_PARAMS_PATH.exists():
        return {}
    try:
        blob = json.loads(BEST_PARAMS_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("Could not parse %s; ignoring tuned params.", BEST_PARAMS_PATH)
        return {}
    return dict(blob.get(name, {}))


_XGB_DEFAULTS = dict(
    n_estimators=600, max_depth=4, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=5, reg_lambda=1.0,
)


def build_xgb(**overrides):
    from xgboost import XGBClassifier

    params = {**_XGB_DEFAULTS, **_load_best_params("xgb"), **overrides}
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


_LGBM_DEFAULTS = dict(
    n_estimators=800, num_leaves=31, max_depth=-1, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.85, min_child_samples=20, reg_lambda=1.0,
)


def build_lgbm(**overrides):
    from lightgbm import LGBMClassifier

    params = {**_LGBM_DEFAULTS, **_load_best_params("lgbm"), **overrides}
    return LGBMClassifier(
        **params,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def build_logistic():
    """L2 logistic on standardized features with median imputation."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
    ])


def build_stacked():
    """Stacked classifier: xgb + lgbm + logistic -> logistic meta."""
    estimators = [
        ("xgb", build_xgb()),
        ("lgbm", build_lgbm()),
        ("logit", build_logistic()),
    ]
    # StackingClassifier requires its CV to be a partition; TimeSeriesSplit is
    # not. The outer TimeSeriesSplit in train.py still enforces the temporal
    # guard at the benchmark level, so an inner KFold here is acceptable.
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,  # avoid thread-stomping between xgb/lgbm
    )


MODEL_REGISTRY: dict[str, Callable] = {
    "xgb": build_xgb,
    "lgbm": build_lgbm,
    "logit": build_logistic,
    "stacked": build_stacked,
}


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Options: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
