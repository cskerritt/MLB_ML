"""Statcast-backed advanced pitcher metrics.

Aggregates pitch-level data into one row per (pitcher, game) and then builds
trailing rolling averages per pitcher. Uses ``shift(1)`` so the current start
never leaks into its own features.

Metrics:
  - ``xwoba_against``   mean estimated wOBA using speed+angle (contact quality)
  - ``k_pct``           strikeouts / batters faced
  - ``bb_pct``          walks / batters faced
  - ``whiff_pct``       swinging strikes / swings
  - ``pitches``         total pitches thrown
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ADV_WINDOWS = (3, 10)

_PA_END_EVENTS = {
    "strikeout", "walk", "hit_by_pitch", "single", "double", "triple",
    "home_run", "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play", "sac_fly", "sac_bunt", "field_error",
    "fielders_choice", "fielders_choice_out", "strikeout_double_play",
    "catcher_interf", "other_out",
}
_SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
    "missed_bunt", "foul_bunt",
}
_WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}


def _per_start(statcast: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level rows to one row per (pitcher, game_date)."""
    df = statcast.copy()
    df["is_pa_end"] = df["events"].isin(_PA_END_EVENTS)
    df["is_k"] = df["events"].eq("strikeout")
    df["is_bb"] = df["events"].eq("walk")
    df["is_swing"] = df["description"].isin(_SWING_DESCRIPTIONS)
    df["is_whiff"] = df["description"].isin(_WHIFF_DESCRIPTIONS)
    df["xwoba"] = pd.to_numeric(df.get("estimated_woba_using_speedangle"), errors="coerce")

    grp = df.groupby(["pitcher", "game_date", "game_pk"], as_index=False)
    agg = grp.agg(
        pitches=("pitch_type", "size"),
        pa=("is_pa_end", "sum"),
        k=("is_k", "sum"),
        bb=("is_bb", "sum"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        xwoba_sum=("xwoba", "sum"),
        xwoba_n=("xwoba", "count"),
    )
    agg["k_pct"] = np.where(agg["pa"] > 0, agg["k"] / agg["pa"], np.nan)
    agg["bb_pct"] = np.where(agg["pa"] > 0, agg["bb"] / agg["pa"], np.nan)
    agg["whiff_pct"] = np.where(agg["swings"] > 0, agg["whiffs"] / agg["swings"], np.nan)
    agg["xwoba_against"] = np.where(
        agg["xwoba_n"] > 0, agg["xwoba_sum"] / agg["xwoba_n"], np.nan
    )
    agg = agg.rename(columns={"pitcher": "pitcher_id", "game_date": "date"})
    agg["pitcher_id"] = agg["pitcher_id"].astype("string")
    agg["date"] = pd.to_datetime(agg["date"])
    return agg[["pitcher_id", "date", "pitches", "pa", "k_pct", "bb_pct",
                "whiff_pct", "xwoba_against"]]


def build_advanced_pitcher_features(statcast: pd.DataFrame) -> pd.DataFrame:
    """Rolling advanced metrics per pitcher, shifted to exclude current start."""
    starts = _per_start(statcast).sort_values(["pitcher_id", "date"]).reset_index(drop=True)
    # Keep only true starts (>= ~40 pitches) so rolling form reflects starters.
    starts = starts[starts["pitches"] >= 40].copy()

    grp = starts.groupby("pitcher_id", group_keys=False)
    for w in ADV_WINDOWS:
        for col in ("xwoba_against", "k_pct", "bb_pct", "whiff_pct"):
            starts[f"sp_{col}_{w}"] = grp[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return starts


def merge_advanced_into_games(games: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    """Attach home/away advanced pitcher features to each game row."""
    need = ["pitcher_id", "date"] + [
        f"sp_{col}_{w}" for w in ADV_WINDOWS
        for col in ("xwoba_against", "k_pct", "bb_pct", "whiff_pct")
    ]
    adv = adv[need].copy()

    g = games.copy()
    for c in ("home_sp_id", "away_sp_id"):
        if c in g.columns:
            g[c] = g[c].astype("string")

    home = adv.add_prefix("home_").rename(
        columns={"home_date": "date", "home_pitcher_id": "home_sp_id"}
    )
    away = adv.add_prefix("away_").rename(
        columns={"away_date": "date", "away_pitcher_id": "away_sp_id"}
    )
    g = g.merge(home, on=["date", "home_sp_id"], how="left")
    g = g.merge(away, on=["date", "away_sp_id"], how="left")

    for w in ADV_WINDOWS:
        for col in ("xwoba_against", "k_pct", "bb_pct", "whiff_pct"):
            g[f"sp_{col}_diff_{w}"] = g[f"away_sp_{col}_{w}"] - g[f"home_sp_{col}_{w}"]
    return g


ADVANCED_PITCHER_FEATURE_COLS = [
    f"sp_{col}_diff_{w}"
    for w in ADV_WINDOWS
    for col in ("xwoba_against", "k_pct", "bb_pct", "whiff_pct")
] + [
    f"home_sp_{col}_{w}"
    for w in ADV_WINDOWS
    for col in ("xwoba_against", "k_pct", "bb_pct", "whiff_pct")
] + [
    f"away_sp_{col}_{w}"
    for w in ADV_WINDOWS
    for col in ("xwoba_against", "k_pct", "bb_pct", "whiff_pct")
]
