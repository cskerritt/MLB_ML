"""Handedness-aware features.

We don't have game-by-game lineup data in Retrosheet game logs, so we proxy
batter-handedness exposure with two signals per team:

  1. Starter's throwing hand (R / L) from Statcast ``p_throws``.
  2. Rolling runs scored by each team in its prior games against L-handed
     starters vs. R-handed starters (``team_rs_vs_L_10``, ``team_rs_vs_R_10``).

Combined, the model sees how the home team has hit lefties / righties
recently AND what hand today's opposing starter throws.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HANDEDNESS_WINDOWS = (10, 30)


def starter_hands(statcast: pd.DataFrame) -> pd.DataFrame:
    """Map each (pitcher_id, game_date) -> throwing hand.

    Resolves to the hand the pitcher used in that game's first pitch (very
    rare flips are ignored).
    """
    if "p_throws" not in statcast.columns:
        return pd.DataFrame(columns=["pitcher_id", "date", "p_throws"])
    df = statcast[["pitcher", "game_date", "p_throws"]].dropna()
    df = df.sort_values(["pitcher", "game_date"]).drop_duplicates(
        subset=["pitcher", "game_date"], keep="first"
    )
    df = df.rename(columns={"pitcher": "pitcher_id", "game_date": "date"})
    df["pitcher_id"] = df["pitcher_id"].astype("string")
    df["date"] = pd.to_datetime(df["date"])
    df["p_throws"] = df["p_throws"].str.upper().where(df["p_throws"].isin(["L", "R"]))
    return df[["pitcher_id", "date", "p_throws"]]


def _attach_opp_hand(games: pd.DataFrame, hands: pd.DataFrame) -> pd.DataFrame:
    """Add home_sp_hand / away_sp_hand to the games frame."""
    g = games.copy()
    for c in ("home_sp_id", "away_sp_id"):
        if c in g.columns:
            g[c] = g[c].astype("string")
    hands = hands.copy()
    hands["pitcher_id"] = hands["pitcher_id"].astype("string")
    hands["date"] = pd.to_datetime(hands["date"])

    g = g.merge(
        hands.rename(columns={"pitcher_id": "home_sp_id", "p_throws": "home_sp_hand"}),
        on=["date", "home_sp_id"], how="left",
    )
    g = g.merge(
        hands.rename(columns={"pitcher_id": "away_sp_id", "p_throws": "away_sp_hand"}),
        on=["date", "away_sp_id"], how="left",
    )
    return g


def build_team_split_history(games_with_hands: pd.DataFrame) -> pd.DataFrame:
    """Rolling team runs-scored split by opposing starter's hand.

    Returns a long (team, date) table with ``rs_vs_L_{w}`` and ``rs_vs_R_{w}``
    columns, shifted so the current game is excluded.
    """
    df = games_with_hands.copy()
    # Explode into per-team rows with opponent starter hand.
    home = pd.DataFrame({
        "date": df["date"], "team": df["home_team"],
        "runs_for": df["home_runs"], "opp_sp_hand": df["away_sp_hand"],
    })
    away = pd.DataFrame({
        "date": df["date"], "team": df["away_team"],
        "runs_for": df["away_runs"], "opp_sp_hand": df["home_sp_hand"],
    })
    long = pd.concat([home, away], ignore_index=True)
    long = long.dropna(subset=["opp_sp_hand"])
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    # For each team, separate series of vs-L and vs-R runs.
    long["rs_vs_L"] = np.where(long["opp_sp_hand"] == "L", long["runs_for"], np.nan)
    long["rs_vs_R"] = np.where(long["opp_sp_hand"] == "R", long["runs_for"], np.nan)

    grp = long.groupby("team", group_keys=False)
    for w in HANDEDNESS_WINDOWS:
        long[f"rs_vs_L_{w}"] = grp["rs_vs_L"].transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
        )
        long[f"rs_vs_R_{w}"] = grp["rs_vs_R"].transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
        )
    cols = ["team", "date"] + [
        f"rs_vs_{h}_{w}" for h in ("L", "R") for w in HANDEDNESS_WINDOWS
    ]
    return long[cols]


def merge_handedness_into_games(
    games: pd.DataFrame, statcast: pd.DataFrame | None
) -> pd.DataFrame:
    """Attach opposing-starter hand + team-vs-L/R scoring history."""
    if statcast is None or statcast.empty:
        return games

    hands = starter_hands(statcast)
    if hands.empty:
        return games

    g = _attach_opp_hand(games, hands)
    splits = build_team_split_history(g)

    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        side_splits = splits.rename(columns={"team": team_col}).add_prefix(f"{side}_")
        side_splits = side_splits.rename(columns={
            f"{side}_{team_col}": team_col,
            f"{side}_date": "date",
        })
        g = g.merge(side_splits, on=["date", team_col], how="left")

    # Pick the relevant split based on the opposing starter's hand.
    for w in HANDEDNESS_WINDOWS:
        g[f"home_rs_vs_opp_hand_{w}"] = np.where(
            g["away_sp_hand"] == "L", g[f"home_rs_vs_L_{w}"], g[f"home_rs_vs_R_{w}"]
        )
        g[f"away_rs_vs_opp_hand_{w}"] = np.where(
            g["home_sp_hand"] == "L", g[f"away_rs_vs_L_{w}"], g[f"away_rs_vs_R_{w}"]
        )
        g[f"rs_vs_opp_hand_diff_{w}"] = (
            g[f"home_rs_vs_opp_hand_{w}"] - g[f"away_rs_vs_opp_hand_{w}"]
        )

    # Binary indicators for the opposing starter's hand.
    g["home_faces_lhp"] = (g["away_sp_hand"] == "L").astype(int)
    g["away_faces_lhp"] = (g["home_sp_hand"] == "L").astype(int)
    return g


HANDEDNESS_FEATURE_COLS = (
    [f"rs_vs_opp_hand_diff_{w}" for w in HANDEDNESS_WINDOWS]
    + [f"home_rs_vs_opp_hand_{w}" for w in HANDEDNESS_WINDOWS]
    + [f"away_rs_vs_opp_hand_{w}" for w in HANDEDNESS_WINDOWS]
    + ["home_faces_lhp", "away_faces_lhp"]
)
