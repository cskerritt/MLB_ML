"""Tests for park-factor computation and lagged merge."""
from __future__ import annotations

import pandas as pd

from mlb_ml.park_factors import compute_park_factors, merge_park_factors_into_games


def _games() -> pd.DataFrame:
    # Two seasons, two parks. PARK_HIGH averages 10 runs/game; PARK_LOW averages 4.
    rows = []
    for season in (2022, 2023):
        for i in range(10):
            rows.append({"season": season, "park_id": "PARK_HIGH",
                         "home_runs": 6, "away_runs": 4,
                         "date": pd.Timestamp(f"{season}-05-01") + pd.Timedelta(days=i),
                         "home_team": "A", "away_team": "B"})
            rows.append({"season": season, "park_id": "PARK_LOW",
                         "home_runs": 2, "away_runs": 2,
                         "date": pd.Timestamp(f"{season}-05-01") + pd.Timedelta(days=i),
                         "home_team": "C", "away_team": "D"})
    return pd.DataFrame(rows)


def test_park_factor_high_above_one_low_below_one():
    pf = compute_park_factors(_games())
    high_2022 = pf[(pf["season"] == 2022) & (pf["park_id"] == "PARK_HIGH")].iloc[0]
    low_2022 = pf[(pf["season"] == 2022) & (pf["park_id"] == "PARK_LOW")].iloc[0]
    assert high_2022["park_factor"] > 1.0
    assert low_2022["park_factor"] < 1.0


def test_merge_uses_prior_season_factor():
    games = _games()
    pf = compute_park_factors(games)
    merged = merge_park_factors_into_games(games, pf)
    # 2022 games should get neutral 1.0 (no prior season available).
    assert (merged[merged["season"] == 2022]["park_factor_lag1"] == 1.0).all()
    # 2023 games at PARK_HIGH should inherit 2022's >1.0 factor.
    high_23 = merged[(merged["season"] == 2023) & (merged["park_id"] == "PARK_HIGH")]
    assert (high_23["park_factor_lag1"] > 1.0).all()
