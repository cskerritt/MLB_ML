"""Tests for umpire factor computation and lagged merge."""
from __future__ import annotations

import pandas as pd

from mlb_ml.umpire import (
    UMPIRE_FEATURE_COLS,
    compute_umpire_factors,
    merge_umpire_factors_into_games,
)


def _games() -> pd.DataFrame:
    rows = []
    for season in (2022, 2023):
        # Ump HIGH: 12 games averaging 12 runs. Ump LOW: 12 games averaging 4 runs.
        for i in range(12):
            rows.append({"season": season, "ump_id": "HIGH",
                         "home_runs": 7, "away_runs": 5,
                         "date": pd.Timestamp(f"{season}-05-01") + pd.Timedelta(days=i),
                         "home_team": "A", "away_team": "B"})
            rows.append({"season": season, "ump_id": "LOW",
                         "home_runs": 2, "away_runs": 2,
                         "date": pd.Timestamp(f"{season}-05-01") + pd.Timedelta(days=i),
                         "home_team": "C", "away_team": "D"})
    return pd.DataFrame(rows)


def test_factors_respect_league_mean():
    uf = compute_umpire_factors(_games())
    high_22 = uf[(uf["season"] == 2022) & (uf["ump_id"] == "HIGH")].iloc[0]
    low_22 = uf[(uf["season"] == 2022) & (uf["ump_id"] == "LOW")].iloc[0]
    assert high_22["ump_factor"] > 1.0
    assert low_22["ump_factor"] < 1.0


def test_lagged_merge_uses_prior_season_only():
    games = _games()
    uf = compute_umpire_factors(games)
    merged = merge_umpire_factors_into_games(games, uf)
    for col in UMPIRE_FEATURE_COLS:
        assert col in merged.columns
    # 2022 games predate any prior-season data -> neutral 1.0.
    assert (merged[merged["season"] == 2022]["ump_factor_lag1"] == 1.0).all()
    # 2023 games should inherit 2022's factors.
    high_23 = merged[(merged["season"] == 2023) & (merged["ump_id"] == "HIGH")]
    assert (high_23["ump_factor_lag1"] > 1.0).all()


def test_missing_ump_id_column_is_safe():
    games = _games().drop(columns=["ump_id"])
    uf = pd.DataFrame(columns=["season", "ump_id", "ump_factor"])
    merged = merge_umpire_factors_into_games(games, uf)
    assert (merged["ump_factor_lag1"] == 1.0).all()
