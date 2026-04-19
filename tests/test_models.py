"""Tests for the model zoo."""
from __future__ import annotations

import numpy as np

from mlb_ml.models import MODEL_REGISTRY, get_model


def test_registry_has_expected_models():
    assert set(MODEL_REGISTRY) == {"xgb", "lgbm", "logit", "stacked"}


def test_each_model_fits_and_predicts_proba():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    for name in MODEL_REGISTRY:
        model = get_model(name)
        model.fit(X, y)
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        # Probabilities should sum to ~1.
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
