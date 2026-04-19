"""Smoke tests for bullpen fatigue features."""
from __future__ import annotations

import pandas as pd

from mlb_ml.bullpen import (
    BULLPEN_WINDOWS,
    build_bullpen_features,
    infer_starters,
    merge_bullpen_into_games,
)


def _toy_statcast() -> pd.DataFrame:
    """Two games on consecutive days for team AAA, mix of starter + reliever pitches."""
    rows = []
    # Day 1: AAA starter S1 throws 80, reliever R1 throws 20.
    for _ in range(80):
        rows.append({"game_pk": 1, "game_date": pd.Timestamp("2024-04-01"),
                     "pitcher": "S1", "home_team": "AAA", "away_team": "BBB",
                     "inning_topbot": "Top"})
    for _ in range(20):
        rows.append({"game_pk": 1, "game_date": pd.Timestamp("2024-04-01"),
                     "pitcher": "R1", "home_team": "AAA", "away_team": "BBB",
                     "inning_topbot": "Top"})
    # Day 2: AAA starter S2 throws 90, reliever R1 throws 10.
    for _ in range(90):
        rows.append({"game_pk": 2, "game_date": pd.Timestamp("2024-04-02"),
                     "pitcher": "S2", "home_team": "AAA", "away_team": "CCC",
                     "inning_topbot": "Top"})
    for _ in range(10):
        rows.append({"game_pk": 2, "game_date": pd.Timestamp("2024-04-02"),
                     "pitcher": "R1", "home_team": "AAA", "away_team": "CCC",
                     "inning_topbot": "Top"})
    return pd.DataFrame(rows)


def test_infer_starters_picks_first_pitcher_per_team():
    sc = _toy_statcast()
    starters = infer_starters(sc)
    aaa_day1 = starters[(starters["game_pk"] == 1) & (starters["pitcher_team"] == "AAA")]
    assert aaa_day1["starter_id"].iloc[0] == "S1"


def test_bullpen_rolling_sum_excludes_today():
    sc = _toy_statcast()
    starters = infer_starters(sc)
    bp = build_bullpen_features(sc, starters)
    aaa = bp[bp["team"] == "AAA"].sort_values("date").reset_index(drop=True)
    # Day 1 has no prior reliever usage -> NaN (first row is shifted out).
    day1 = aaa[aaa["date"] == pd.Timestamp("2024-04-01")].iloc[0]
    assert pd.isna(day1["bp_pitches_last_1d"])
    # Day 2's 1-day lookback should equal day-1 reliever count (20).
    day2 = aaa[aaa["date"] == pd.Timestamp("2024-04-02")].iloc[0]
    assert day2["bp_pitches_last_1d"] == 20


def test_merge_bullpen_into_games_adds_both_sides():
    sc = _toy_statcast()
    starters = infer_starters(sc)
    bp = build_bullpen_features(sc, starters)
    games = pd.DataFrame({
        "date": [pd.Timestamp("2024-04-02")],
        "home_team": ["AAA"], "away_team": ["CCC"],
        "home_runs": [4], "away_runs": [1], "home_win": [1], "season": [2024],
    })
    merged = merge_bullpen_into_games(games, bp)
    for w in BULLPEN_WINDOWS:
        assert f"home_bp_pitches_last_{w}d" in merged.columns
        assert f"bp_pitches_diff_{w}d" in merged.columns
