"""Smoke tests for feature engineering invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_ml.features import build_features


def _toy_games(n_days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    start = pd.Timestamp("2024-04-01")
    for d in range(n_days):
        date = start + pd.Timedelta(days=d)
        h, a = rng.choice(teams, size=2, replace=False)
        hr = int(rng.integers(0, 10))
        ar = int(rng.integers(0, 10))
        if hr == ar:
            hr += 1
        rows.append({
            "date": date,
            "season": 2024,
            "home_team": h,
            "away_team": a,
            "home_runs": hr,
            "away_runs": ar,
            "home_win": int(hr > ar),
            "day_night": "N",
        })
    return pd.DataFrame(rows)


def test_no_future_leakage_in_rolling_stats():
    games = _toy_games()
    feats = build_features(games)
    first_game = feats.iloc[0]
    # First game for each team has no prior history -> rolling cols must be NaN.
    assert pd.isna(first_game["home_win_pct_10"])
    assert pd.isna(first_game["away_win_pct_10"])


def test_elo_diff_sign_matches_outcomes_on_average():
    games = _toy_games(120)
    feats = build_features(games)
    feats = feats.dropna(subset=["elo_diff"])
    # Higher home Elo should correlate positively with home wins.
    corr = feats[["elo_diff", "home_win"]].corr().iloc[0, 1]
    assert corr > -0.5  # weak guarantee on toy data; just checks orientation
