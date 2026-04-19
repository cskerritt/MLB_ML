"""Smoke tests for Statcast-backed advanced pitcher features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_ml.pitcher_adv import (
    ADV_WINDOWS,
    build_advanced_pitcher_features,
    merge_advanced_into_games,
)


def _toy_statcast(n_starts: int = 20) -> pd.DataFrame:
    """Build a tiny pitch-level dataframe with two pitchers alternating starts."""
    rng = np.random.default_rng(2)
    rows = []
    start = pd.Timestamp("2024-04-01")
    for g in range(n_starts):
        date = start + pd.Timedelta(days=g)
        pitcher = ["P01", "P02"][g % 2]
        for pitch in range(60):
            desc = rng.choice(
                ["ball", "called_strike", "swinging_strike", "foul", "hit_into_play"],
                p=[0.35, 0.25, 0.15, 0.15, 0.10],
            )
            event = ""
            if desc == "hit_into_play":
                event = rng.choice(["field_out", "single", "double", "home_run"],
                                   p=[0.65, 0.22, 0.08, 0.05])
            elif desc == "swinging_strike" and pitch % 6 == 0:
                event = "strikeout"
            rows.append({
                "game_date": date,
                "game_pk": g,
                "pitcher": pitcher,
                "pitch_type": "FF",
                "events": event,
                "description": desc,
                "type": "X",
                "estimated_woba_using_speedangle": rng.random() if desc == "hit_into_play" else np.nan,
                "woba_value": np.nan,
                "woba_denom": np.nan,
                "home_team": "AAA",
                "away_team": "BBB",
                "inning_topbot": "Top" if g % 2 == 0 else "Bot",
            })
    return pd.DataFrame(rows)


def test_advanced_features_shifted_no_leakage():
    sc = _toy_statcast()
    adv = build_advanced_pitcher_features(sc)
    firsts = adv.groupby("pitcher_id").head(1)
    for w in ADV_WINDOWS:
        assert firsts[f"sp_xwoba_against_{w}"].isna().all()
        assert firsts[f"sp_k_pct_{w}"].isna().all()


def test_merge_advanced_attaches_both_sides():
    sc = _toy_statcast()
    adv = build_advanced_pitcher_features(sc)
    games = pd.DataFrame({
        "date": [pd.Timestamp("2024-04-10")],
        "home_team": ["AAA"], "away_team": ["BBB"],
        "home_sp_id": ["P01"], "away_sp_id": ["P02"],
        "home_runs": [3], "away_runs": [2], "home_win": [1],
        "season": [2024],
    })
    merged = merge_advanced_into_games(games, adv)
    assert "home_sp_xwoba_against_3" in merged.columns
    assert "sp_k_pct_diff_10" in merged.columns
