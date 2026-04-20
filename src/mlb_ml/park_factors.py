"""Park factors computed from historical scoring.

Park factor = (runs per game at park X) / (runs per game at all parks),
computed per season and then LAGGED one season so current-season games only
see prior-season park behavior. Neutral park = 1.0; hitter-friendly > 1.0.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_park_factors(games: pd.DataFrame) -> pd.DataFrame:
    """Return a (season, park_id, park_factor) table.

    ``games`` must contain ``season``, ``park_id``, ``home_runs``,
    ``away_runs``. Rows where park_id is missing are dropped.
    """
    if "park_id" not in games.columns:
        raise ValueError("games is missing park_id; park factors cannot be computed")

    df = games.dropna(subset=["park_id"]).copy()
    df["total_runs"] = df["home_runs"].fillna(0) + df["away_runs"].fillna(0)

    league = df.groupby("season", as_index=False)["total_runs"].mean().rename(
        columns={"total_runs": "league_rpg"}
    )
    park = df.groupby(["season", "park_id"], as_index=False)["total_runs"].mean().rename(
        columns={"total_runs": "park_rpg"}
    )
    merged = park.merge(league, on="season", how="left")
    merged["park_factor"] = np.where(
        merged["league_rpg"] > 0, merged["park_rpg"] / merged["league_rpg"], 1.0
    )
    return merged[["season", "park_id", "park_factor"]]


def merge_park_factors_into_games(
    games: pd.DataFrame, park_factors: pd.DataFrame
) -> pd.DataFrame:
    """Attach a ``park_factor_lag1`` column using the PRIOR season's factor."""
    lagged = park_factors.copy()
    lagged["season"] = lagged["season"] + 1  # shift forward so season S sees S-1
    lagged = lagged.rename(columns={"park_factor": "park_factor_lag1"})

    out = games.merge(lagged, on=["season", "park_id"], how="left")
    # Fill missing (new parks, first season in data) with neutral 1.0.
    out["park_factor_lag1"] = out["park_factor_lag1"].fillna(1.0)
    return out


PARK_FEATURE_COLS = ["park_factor_lag1"]
