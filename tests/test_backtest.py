"""Unit tests for backtest helpers."""
from __future__ import annotations

import math

from mlb_ml.backtest import _american_to_decimal, _american_to_prob


def test_american_to_prob_negative():
    # -150 favorite -> 60% implied
    assert math.isclose(_american_to_prob(-150), 0.60, abs_tol=1e-6)


def test_american_to_prob_positive():
    # +130 dog -> ~43.48% implied
    assert math.isclose(_american_to_prob(130), 100 / 230, abs_tol=1e-6)


def test_american_to_decimal_roundtrip():
    # +200 -> 3.0 decimal
    assert math.isclose(_american_to_decimal(200), 3.0, abs_tol=1e-6)
    # -200 -> 1.5 decimal
    assert math.isclose(_american_to_decimal(-200), 1.5, abs_tol=1e-6)
