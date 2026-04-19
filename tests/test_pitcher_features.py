"""Smoke tests for pitcher rolling features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_ml.pitcher_features import build_pitcher_features, merge_into_games


def _toy_games(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    teams = ["AAA", "BBB", "CCC", "DDD"]
    pitchers = [f"P{i:02d}" for i in range(8)]
    rows = []
    start = pd.Timestamp("2024-04-01")
    for d in range(n):
        date = start + pd.Timedelta(days=d)
        h, a = rng.choice(teams, size=2, replace=False)
        hp, ap = rng.choice(pitchers, size=2, replace=False)
        hr = int(rng.integers(0, 10))
        ar = int(rng.integers(0, 10))
        if hr == ar:
            hr += 1
        rows.append({
            "date": date, "season": 2024,
            "home_team": h, "away_team": a,
            "home_runs": hr, "away_runs": ar,
            "home_win": int(hr > ar),
            "home_sp_id": hp, "away_sp_id": ap,
            "day_night": "N",
        })
    return pd.DataFrame(rows)


def test_first_start_per_pitcher_has_nan_rolling():
    games = _toy_games()
    pf = build_pitcher_features(games)
    firsts = pf.groupby("pitcher_id").head(1)
    assert firsts["sp_runs_allowed_3"].isna().all()
    assert firsts["sp_runs_allowed_10"].isna().all()


def test_merge_attaches_both_sides():
    games = _toy_games()
    pf = build_pitcher_features(games)
    merged = merge_into_games(games, pf)
    assert "home_sp_runs_allowed_10" in merged.columns
    assert "away_sp_runs_allowed_10" in merged.columns
    assert "sp_runs_allowed_diff_10" in merged.columns
    assert len(merged) == len(games)
