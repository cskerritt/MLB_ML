"""Tests for handedness features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_ml.handedness import (
    HANDEDNESS_FEATURE_COLS,
    merge_handedness_into_games,
    starter_hands,
)


def _toy_statcast() -> pd.DataFrame:
    rows = []
    # Pitcher P01 is a lefty across two starts; P02 is a righty.
    for day, pid, hand in [
        (pd.Timestamp("2024-04-01"), "P01", "L"),
        (pd.Timestamp("2024-04-02"), "P02", "R"),
        (pd.Timestamp("2024-04-05"), "P01", "L"),
        (pd.Timestamp("2024-04-06"), "P02", "R"),
    ]:
        for _ in range(60):
            rows.append({
                "game_date": day, "game_pk": hash((day, pid)) % 1000,
                "pitcher": pid, "p_throws": hand,
                "home_team": "AAA", "away_team": "BBB",
                "inning_topbot": "Top",
                "events": "", "description": "ball",
            })
    return pd.DataFrame(rows)


def _toy_games() -> pd.DataFrame:
    return pd.DataFrame([
        # AAA scored 6 vs LHP on 4/1, 2 vs RHP on 4/2, then plays a LHP on 4/5.
        {"date": pd.Timestamp("2024-04-01"), "season": 2024,
         "home_team": "AAA", "away_team": "BBB",
         "home_sp_id": "P01", "away_sp_id": "P02",
         "home_runs": 3, "away_runs": 6, "home_win": 0, "day_night": "N"},
        {"date": pd.Timestamp("2024-04-02"), "season": 2024,
         "home_team": "BBB", "away_team": "AAA",
         "home_sp_id": "P02", "away_sp_id": "P01",
         "home_runs": 5, "away_runs": 2, "home_win": 1, "day_night": "N"},
        {"date": pd.Timestamp("2024-04-05"), "season": 2024,
         "home_team": "AAA", "away_team": "BBB",
         "home_sp_id": "P01", "away_sp_id": "P02",
         "home_runs": 7, "away_runs": 4, "home_win": 1, "day_night": "N"},
    ])


def test_starter_hands_deduplicated_per_game():
    sc = _toy_statcast()
    hands = starter_hands(sc)
    # Four unique (pitcher, date) combinations in the toy set.
    assert len(hands) == 4
    assert set(hands["p_throws"].unique()) == {"L", "R"}


def test_merge_produces_expected_feature_columns():
    merged = merge_handedness_into_games(_toy_games(), _toy_statcast())
    for col in HANDEDNESS_FEATURE_COLS:
        assert col in merged.columns
    # On 4/2, AAA is the visiting team facing LHP P01 (away_sp_hand)? No --
    # home_sp_id is P02 (R) so away (AAA) faces RHP, so away_faces_lhp=0.
    row = merged[merged["date"] == pd.Timestamp("2024-04-02")].iloc[0]
    assert row["away_faces_lhp"] == 0
    # On 4/5, away team BBB faces LHP P01 (home starter is lefty).
    row_5 = merged[merged["date"] == pd.Timestamp("2024-04-05")].iloc[0]
    assert row_5["away_faces_lhp"] == 1


def test_handedness_split_shifted_no_leakage():
    merged = merge_handedness_into_games(_toy_games(), _toy_statcast())
    # AAA's first game has no prior vs-L history.
    first_aaa = merged[merged["date"] == pd.Timestamp("2024-04-01")].iloc[0]
    assert np.isnan(first_aaa["home_rs_vs_opp_hand_10"]) or first_aaa["home_rs_vs_opp_hand_10"] == 0 \
        or pd.isna(first_aaa["home_rs_vs_opp_hand_10"])
