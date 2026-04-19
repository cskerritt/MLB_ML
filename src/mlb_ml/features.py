"""Feature engineering for MLB win-probability prediction.

All features are computed using only information available BEFORE first pitch
to avoid target leakage. Rolling stats use ``shift(1)`` then ``rolling()``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import FEATURES_PARQUET, GAMES_PARQUET, TARGET_COL

log = logging.getLogger(__name__)

ROLL_WINDOWS = (10, 30)


def _team_long_format(games: pd.DataFrame) -> pd.DataFrame:
    """Explode each game into two rows (one per team) for rolling stats."""
    home = games.assign(
        team=games["home_team"],
        opp=games["away_team"],
        is_home=1,
        runs_for=games["home_runs"],
        runs_against=games["away_runs"],
        win=games["home_win"],
    )
    away = games.assign(
        team=games["away_team"],
        opp=games["home_team"],
        is_home=0,
        runs_for=games["away_runs"],
        runs_against=games["home_runs"],
        win=1 - games["home_win"],
    )
    cols = ["date", "season", "team", "opp", "is_home", "runs_for", "runs_against", "win"]
    return pd.concat([home[cols], away[cols]], ignore_index=True).sort_values(
        ["team", "date"]
    )


def _rolling_team_stats(team_games: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling team form, shifted to exclude the current game."""
    out = team_games.copy()
    grp = out.groupby("team", group_keys=False)

    for w in ROLL_WINDOWS:
        out[f"win_pct_{w}"] = grp["win"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=3).mean()
        )
        out[f"rs_per_g_{w}"] = grp["runs_for"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=3).mean()
        )
        out[f"ra_per_g_{w}"] = grp["runs_against"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=3).mean()
        )
        out[f"run_diff_{w}"] = out[f"rs_per_g_{w}"] - out[f"ra_per_g_{w}"]

    out["rest_days"] = grp["date"].transform(lambda s: s.diff().dt.days).fillna(3).clip(0, 10)
    return out


def _elo(games: pd.DataFrame, k: float = 4.0, hfa: float = 24.0) -> pd.DataFrame:
    """Vanilla Elo with home-field advantage. Returns pre-game ratings per game."""
    ratings: dict[str, float] = {}
    home_pre, away_pre = [], []
    for _, g in games.iterrows():
        rh = ratings.get(g["home_team"], 1500.0)
        ra = ratings.get(g["away_team"], 1500.0)
        home_pre.append(rh)
        away_pre.append(ra)
        exp_home = 1.0 / (1.0 + 10 ** (-((rh + hfa) - ra) / 400))
        result = g[TARGET_COL]
        ratings[g["home_team"]] = rh + k * (result - exp_home)
        ratings[g["away_team"]] = ra + k * ((1 - result) - (1 - exp_home))
    return pd.DataFrame(
        {"home_elo_pre": home_pre, "away_elo_pre": away_pre}, index=games.index
    )


def build_features(games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Construct the modeling table from a games dataframe."""
    if games is None:
        games = pd.read_parquet(GAMES_PARQUET)
    games = games.sort_values("date").reset_index(drop=True)

    long = _team_long_format(games)
    long = _rolling_team_stats(long)

    home_feats = long[long["is_home"] == 1].add_prefix("home_")
    away_feats = long[long["is_home"] == 0].add_prefix("away_")
    home_feats = home_feats.rename(columns={"home_date": "date", "home_team": "home_team"})
    away_feats = away_feats.rename(columns={"away_date": "date", "away_team": "away_team"})

    merged = games.merge(
        home_feats.drop(columns=["home_opp", "home_is_home", "home_season"]),
        on=["date", "home_team"],
        how="left",
    ).merge(
        away_feats.drop(columns=["away_opp", "away_is_home", "away_season"]),
        on=["date", "away_team"],
        how="left",
    )

    elo = _elo(games)
    merged = pd.concat([merged, elo], axis=1)
    merged["elo_diff"] = merged["home_elo_pre"] - merged["away_elo_pre"]

    for w in ROLL_WINDOWS:
        merged[f"win_pct_diff_{w}"] = merged[f"home_win_pct_{w}"] - merged[f"away_win_pct_{w}"]
        merged[f"run_diff_diff_{w}"] = merged[f"home_run_diff_{w}"] - merged[f"away_run_diff_{w}"]

    merged["rest_diff"] = merged["home_rest_days"] - merged["away_rest_days"]
    merged["is_day"] = (merged.get("day_night", "N").astype(str).str.upper() == "D").astype(int)

    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged.to_parquet(FEATURES_PARQUET, index=False)
    log.info("Wrote %s feature rows to %s", len(merged), FEATURES_PARQUET)
    return merged


FEATURE_COLS = [
    "elo_diff",
    "home_elo_pre",
    "away_elo_pre",
    "rest_diff",
    "is_day",
    *[f"win_pct_diff_{w}" for w in ROLL_WINDOWS],
    *[f"run_diff_diff_{w}" for w in ROLL_WINDOWS],
    *[f"home_win_pct_{w}" for w in ROLL_WINDOWS],
    *[f"away_win_pct_{w}" for w in ROLL_WINDOWS],
    *[f"home_run_diff_{w}" for w in ROLL_WINDOWS],
    *[f"away_run_diff_{w}" for w in ROLL_WINDOWS],
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_features()
