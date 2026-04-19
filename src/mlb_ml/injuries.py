"""Injured-list features.

Free IL feeds are scraped from rotoworld / baseball-ref and churn often, so
we keep the same pattern as weather: the caller supplies a CSV snapshot
capturing each team's IL load per date. The model sees how many players
each team had on the IL going into the game.

Expected CSV schema (one row per team per date the snapshot was taken):

    date,team,il_count,il_wrc_lost,il_war_lost

``il_wrc_lost`` and ``il_war_lost`` are optional; if missing, we fill with
zero so the column still exists but carries no signal.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

INJURY_FEATURE_COLS = [
    "home_il_count", "away_il_count", "il_count_diff",
    "home_il_wrc_lost", "away_il_wrc_lost", "il_wrc_lost_diff",
    "home_il_war_lost", "away_il_war_lost", "il_war_lost_diff",
]


def load_injuries_csv(path: str | Path) -> pd.DataFrame:
    """Load an injuries snapshot CSV. Adds optional columns as zeros."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "team", "il_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Injuries CSV is missing columns: {sorted(missing)}")
    for c in ("il_wrc_lost", "il_war_lost"):
        if c not in df.columns:
            df[c] = 0.0
    df["team"] = df["team"].astype(str)
    df["il_count"] = df["il_count"].fillna(0).astype(int)
    return df


def _asof_team(games_side: pd.DataFrame, il: pd.DataFrame) -> pd.DataFrame:
    """Merge the most-recent-prior IL snapshot onto each (team, date)."""
    g = games_side.sort_values("date").copy()
    il_sorted = il.sort_values("date")
    out = pd.merge_asof(
        g, il_sorted,
        on="date", by="team",
        direction="backward",
        allow_exact_matches=True,
    )
    for c in ("il_count", "il_wrc_lost", "il_war_lost"):
        out[c] = out[c].fillna(0)
    return out


def merge_injuries_into_games(games: pd.DataFrame, il: pd.DataFrame) -> pd.DataFrame:
    """Attach home/away IL counts using an as-of merge to avoid leakage."""
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"])
    il = il.copy()
    il["date"] = pd.to_datetime(il["date"])

    home = g[["date", "home_team"]].rename(columns={"home_team": "team"}).assign(row=range(len(g)))
    away = g[["date", "away_team"]].rename(columns={"away_team": "team"}).assign(row=range(len(g)))
    home = _asof_team(home, il).sort_values("row")
    away = _asof_team(away, il).sort_values("row")

    g["home_il_count"] = home["il_count"].values
    g["away_il_count"] = away["il_count"].values
    g["home_il_wrc_lost"] = home["il_wrc_lost"].values
    g["away_il_wrc_lost"] = away["il_wrc_lost"].values
    g["home_il_war_lost"] = home["il_war_lost"].values
    g["away_il_war_lost"] = away["il_war_lost"].values

    g["il_count_diff"] = g["away_il_count"] - g["home_il_count"]
    g["il_wrc_lost_diff"] = g["away_il_wrc_lost"] - g["home_il_wrc_lost"]
    g["il_war_lost_diff"] = g["away_il_war_lost"] - g["home_il_war_lost"]
    return g
