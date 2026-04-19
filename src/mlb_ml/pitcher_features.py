"""Rolling starting-pitcher features.

Builds a per-(pitcher, start) table from Retrosheet game logs and computes
trailing windows of FIP-like indicators. We use simplified estimators because
Retrosheet game logs don't include batters faced or HR allowed at game level
without the more expensive Statcast pull -- this module is structured so a
Statcast upgrade slots in cleanly later.

Inputs:  games dataframe (output of data._normalize)
Outputs: long dataframe keyed by (date, pitcher_id) with rolling features.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PITCHER_WINDOWS = (3, 10)


def _starts_long(games: pd.DataFrame) -> pd.DataFrame:
    """Explode each game into two pitcher-start rows."""
    home = pd.DataFrame(
        {
            "date": games["date"],
            "season": games["season"],
            "pitcher_id": games.get("home_sp_id"),
            "team": games["home_team"],
            "is_home": 1,
            "runs_allowed": games["away_runs"],
            "won": games["home_win"],
        }
    )
    away = pd.DataFrame(
        {
            "date": games["date"],
            "season": games["season"],
            "pitcher_id": games.get("away_sp_id"),
            "team": games["away_team"],
            "is_home": 0,
            "runs_allowed": games["home_runs"],
            "won": 1 - games["home_win"],
        }
    )
    starts = pd.concat([home, away], ignore_index=True)
    starts = starts.dropna(subset=["pitcher_id"])
    starts["pitcher_id"] = starts["pitcher_id"].astype(str)
    return starts.sort_values(["pitcher_id", "date"]).reset_index(drop=True)


def build_pitcher_features(games: pd.DataFrame) -> pd.DataFrame:
    """Trailing pitcher form, shifted to exclude the current start."""
    starts = _starts_long(games)
    grp = starts.groupby("pitcher_id", group_keys=False)

    out = starts.copy()
    for w in PITCHER_WINDOWS:
        out[f"sp_runs_allowed_{w}"] = grp["runs_allowed"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        out[f"sp_win_pct_{w}"] = grp["won"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )

    out["sp_days_rest"] = grp["date"].transform(lambda s: s.diff().dt.days).clip(0, 14)
    out["sp_career_starts"] = grp.cumcount()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def merge_into_games(games: pd.DataFrame, pitcher_feats: pd.DataFrame) -> pd.DataFrame:
    """Attach home/away pitcher rolling features to each game row."""
    pf = pitcher_feats.drop(columns=["team", "is_home", "runs_allowed", "won", "season"])

    home_pf = pf.add_prefix("home_").rename(
        columns={"home_date": "date", "home_pitcher_id": "home_sp_id"}
    )
    away_pf = pf.add_prefix("away_").rename(
        columns={"away_date": "date", "away_pitcher_id": "away_sp_id"}
    )

    g = games.copy()
    if "home_sp_id" in g.columns:
        g["home_sp_id"] = g["home_sp_id"].astype("string")
    if "away_sp_id" in g.columns:
        g["away_sp_id"] = g["away_sp_id"].astype("string")
    home_pf["home_sp_id"] = home_pf["home_sp_id"].astype("string")
    away_pf["away_sp_id"] = away_pf["away_sp_id"].astype("string")

    g = g.merge(home_pf, on=["date", "home_sp_id"], how="left")
    g = g.merge(away_pf, on=["date", "away_sp_id"], how="left")

    for w in PITCHER_WINDOWS:
        g[f"sp_runs_allowed_diff_{w}"] = (
            g[f"away_sp_runs_allowed_{w}"] - g[f"home_sp_runs_allowed_{w}"]
        )
    g["sp_rest_diff"] = g["home_sp_days_rest"] - g["away_sp_days_rest"]
    g["sp_experience_diff"] = g["home_sp_career_starts"] - g["away_sp_career_starts"]
    return g


PITCHER_FEATURE_COLS = [
    *[f"sp_runs_allowed_diff_{w}" for w in PITCHER_WINDOWS],
    *[f"home_sp_runs_allowed_{w}" for w in PITCHER_WINDOWS],
    *[f"away_sp_runs_allowed_{w}" for w in PITCHER_WINDOWS],
    "sp_rest_diff",
    "sp_experience_diff",
]
