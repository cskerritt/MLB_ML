"""Smoke test for the Optuna tuner persistence layer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlb_ml import config, tune


def test_save_best_merges_existing(tmp_path: Path, monkeypatch):
    target = tmp_path / "best_params.json"
    target.write_text(json.dumps({"xgb": {"n_estimators": 500}}))
    monkeypatch.setattr(config, "BEST_PARAMS_PATH", target)
    monkeypatch.setattr(tune, "BEST_PARAMS_PATH", target)

    tune.save_best({"lgbm": {"num_leaves": 31}})
    blob = json.loads(target.read_text())
    assert blob["xgb"]["n_estimators"] == 500
    assert blob["lgbm"]["num_leaves"] == 31


def test_tune_rejects_unknown_model():
    with pytest.raises(ValueError):
        tune.tune("logit", n_trials=1)
