"""Tests for injuries/IL feature merge."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlb_ml.injuries import (
    INJURY_FEATURE_COLS,
    load_injuries_csv,
    merge_injuries_into_games,
)


def _write_csv(tmp_path: Path, include_optional: bool = True) -> Path:
    rows = [
        {"date": "2024-04-01", "team": "AAA", "il_count": 3},
        {"date": "2024-04-05", "team": "AAA", "il_count": 5},
        {"date": "2024-04-01", "team": "BBB", "il_count": 1},
    ]
    if include_optional:
        for r, w, war in zip(rows, [4.5, 7.1, 1.2], [0.5, 0.9, 0.1]):
            r["il_wrc_lost"] = w
            r["il_war_lost"] = war
    path = tmp_path / "il.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_injuries_requires_core_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-04-01"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_injuries_csv(bad)


def test_optional_columns_default_to_zero(tmp_path):
    il = load_injuries_csv(_write_csv(tmp_path, include_optional=False))
    assert (il["il_wrc_lost"] == 0).all()
    assert (il["il_war_lost"] == 0).all()


def test_asof_merge_uses_latest_prior_snapshot(tmp_path):
    il = load_injuries_csv(_write_csv(tmp_path))
    games = pd.DataFrame([
        # AAA on 4/3: should see 3 (from 4/1), not 5 (from 4/5).
        {"date": pd.Timestamp("2024-04-03"), "season": 2024,
         "home_team": "AAA", "away_team": "BBB",
         "home_runs": 4, "away_runs": 2, "home_win": 1, "day_night": "N"},
        # AAA on 4/6: should see 5 (from 4/5).
        {"date": pd.Timestamp("2024-04-06"), "season": 2024,
         "home_team": "AAA", "away_team": "BBB",
         "home_runs": 1, "away_runs": 3, "home_win": 0, "day_night": "N"},
    ])
    merged = merge_injuries_into_games(games, il)
    for col in INJURY_FEATURE_COLS:
        assert col in merged.columns
    assert merged.iloc[0]["home_il_count"] == 3
    assert merged.iloc[1]["home_il_count"] == 5
    assert merged.iloc[0]["away_il_count"] == 1
