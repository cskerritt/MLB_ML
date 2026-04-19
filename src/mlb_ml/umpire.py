"""Umpire tendency features, analogous to park factors.

For each season we compute the average total runs per game under each home
plate umpire, then shift the table forward one season so the current game
only sees the prior season's tendency. Umpires with few games fall back to
the neutral 1.0 factor.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_GAMES_PER_SEASON = 10

UMPIRE_FEATURE_COLS = ["ump_factor_lag1"]


def compute_umpire_factors(games: pd.DataFrame) -> pd.DataFrame:
    """Return (season, ump_id, ump_factor) normalised to the league mean."""
    if "ump_id" not in games.columns:
        raise ValueError("games is missing ump_id; umpire factors cannot be computed")

    df = games.dropna(subset=["ump_id"]).copy()
    df["total_runs"] = df["home_runs"].fillna(0) + df["away_runs"].fillna(0)
    df["ump_id"] = df["ump_id"].astype(str)

    league = df.groupby("season", as_index=False)["total_runs"].mean().rename(
        columns={"total_runs": "league_rpg"}
    )
    ump = df.groupby(["season", "ump_id"], as_index=False).agg(
        ump_rpg=("total_runs", "mean"),
        games=("total_runs", "size"),
    )
    ump = ump[ump["games"] >= MIN_GAMES_PER_SEASON]
    merged = ump.merge(league, on="season", how="left")
    merged["ump_factor"] = np.where(
        merged["league_rpg"] > 0, merged["ump_rpg"] / merged["league_rpg"], 1.0
    )
    return merged[["season", "ump_id", "ump_factor"]]


def merge_umpire_factors_into_games(
    games: pd.DataFrame, ump_factors: pd.DataFrame
) -> pd.DataFrame:
    """Attach ``ump_factor_lag1`` using the prior season's factor."""
    g = games.copy()
    if "ump_id" not in g.columns:
        g["ump_factor_lag1"] = 1.0
        return g
    g["ump_id"] = g["ump_id"].astype(str)

    lagged = ump_factors.copy()
    lagged["season"] = lagged["season"] + 1
    lagged = lagged.rename(columns={"ump_factor": "ump_factor_lag1"})
    out = g.merge(lagged, on=["season", "ump_id"], how="left")
    out["ump_factor_lag1"] = out["ump_factor_lag1"].fillna(1.0)
    return out
