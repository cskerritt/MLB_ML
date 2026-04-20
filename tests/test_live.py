"""Tests for live daily-pipeline helpers."""
from __future__ import annotations

import pandas as pd

from mlb_ml.live import _assemble_row, _latest_feature_snapshot, _normalise_team


def test_normalise_team_aliases():
    assert _normalise_team("LAD") == "LAN"
    assert _normalise_team("AZ") == "ARI"
    # Unknown abbreviations pass through unchanged.
    assert _normalise_team("BOS") == "BOS"


def test_latest_feature_snapshot_picks_most_recent_prior_row():
    feats = pd.DataFrame({
        "date": pd.to_datetime(["2024-04-01", "2024-04-05", "2024-04-10"]),
        "home_team": ["AAA", "BBB", "AAA"],
        "away_team": ["BBB", "AAA", "CCC"],
        "home_elo_pre": [1500.0, 1505.0, 1510.0],
    })
    snap = _latest_feature_snapshot(feats, "AAA", pd.Timestamp("2024-04-08"))
    assert snap is not None
    assert snap["date"] == pd.Timestamp("2024-04-05")


def test_assemble_row_copies_team_features_by_side():
    feats = pd.DataFrame({
        "date": pd.to_datetime(["2024-04-01"]),
        "home_team": ["AAA"],
        "away_team": ["BBB"],
        "home_elo_pre": [1520.0],
        "away_elo_pre": [1490.0],
        "elo_diff": [30.0],
    })
    feature_cols = ["home_elo_pre", "away_elo_pre", "elo_diff"]
    schedule_row = {
        "date": pd.Timestamp("2024-04-02"),
        "home_team": "AAA", "away_team": "BBB",
    }
    row = _assemble_row(schedule_row, feats, feature_cols)
    # AAA was home in the snapshot and is home today -> home_elo_pre carries.
    assert row["home_elo_pre"] == 1520.0
    # BBB was away in the snapshot and is away today -> away_elo_pre carries.
    assert row["away_elo_pre"] == 1490.0
    # Non-sided column is taken from the home snapshot.
    assert row["elo_diff"] == 30.0
