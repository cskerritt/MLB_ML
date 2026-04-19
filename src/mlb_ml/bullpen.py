"""Bullpen fatigue features from Statcast pitch-level data.

For each (team, game_date) we compute how many pitches that team's relievers
threw over the trailing N days (excluding today, so we can use it as an
input). Relievers are defined as any pitcher on the team whose pitcher_id is
not equal to the team's starter that day. We approximate "team's starter" as
the pitcher who threw the first pitch of their team's half-inning.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BULLPEN_WINDOWS = (1, 3)


def _team_for_pitcher(statcast: pd.DataFrame) -> pd.DataFrame:
    """Pitcher team = the team batting the OTHER half-inning (inning_topbot)."""
    df = statcast[["game_pk", "game_date", "pitcher", "home_team", "away_team",
                   "inning_topbot"]].copy()
    # Top = away batting, so pitcher is on home team. Bot = home batting -> away pitcher.
    df["pitcher_team"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    return df


def _per_pitcher_game(statcast: pd.DataFrame) -> pd.DataFrame:
    base = _team_for_pitcher(statcast)
    agg = base.groupby(
        ["game_pk", "game_date", "pitcher_team", "pitcher"], as_index=False
    ).size().rename(columns={"size": "pitches"})
    agg["game_date"] = pd.to_datetime(agg["game_date"])
    return agg


def _flag_starters(per_game: pd.DataFrame, starters_by_game: pd.DataFrame) -> pd.DataFrame:
    """starters_by_game: columns [game_pk, pitcher_team, starter_id]."""
    merged = per_game.merge(starters_by_game, on=["game_pk", "pitcher_team"], how="left")
    merged["is_reliever"] = merged["pitcher"].astype("string") != merged["starter_id"].astype("string")
    return merged


def build_bullpen_features(
    statcast: pd.DataFrame, starters_by_game: pd.DataFrame
) -> pd.DataFrame:
    """Return a (team, date) dataframe with rolling reliever pitch counts.

    Parameters
    ----------
    statcast : pitch-level dataframe with columns used by ``_team_for_pitcher``.
    starters_by_game : dataframe with ``game_pk``, ``pitcher_team``, ``starter_id``.
    """
    per_game = _per_pitcher_game(statcast)
    flagged = _flag_starters(per_game, starters_by_game)
    bullpen = (
        flagged[flagged["is_reliever"]]
        .groupby(["pitcher_team", "game_date"], as_index=False)["pitches"]
        .sum()
        .rename(columns={"pitcher_team": "team", "game_date": "date", "pitches": "bp_pitches"})
    )
    bullpen = bullpen.sort_values(["team", "date"]).reset_index(drop=True)

    # Build a dense (team, date) grid so rolling sums honor actual calendar gaps.
    out_rows = []
    for team, sub in bullpen.groupby("team", group_keys=False):
        sub = sub.set_index("date").sort_index()
        dense = sub.reindex(pd.date_range(sub.index.min(), sub.index.max(), freq="D"),
                            fill_value=0)
        dense.index.name = "date"
        for w in BULLPEN_WINDOWS:
            dense[f"bp_pitches_last_{w}d"] = (
                dense["bp_pitches"].shift(1).rolling(w, min_periods=1).sum()
            )
        dense["team"] = team
        out_rows.append(dense.reset_index())
    out = pd.concat(out_rows, ignore_index=True)
    return out[["team", "date"] + [f"bp_pitches_last_{w}d" for w in BULLPEN_WINDOWS]]


def infer_starters(statcast: pd.DataFrame) -> pd.DataFrame:
    """Guess each team's starter per game as the first pitcher in their half."""
    base = _team_for_pitcher(statcast)
    # The first pitch of each (game, team-as-pitcher) half belongs to the starter.
    base = base.sort_values(["game_pk", "pitcher_team"])
    first = base.groupby(["game_pk", "pitcher_team"], as_index=False).first()
    first = first.rename(columns={"pitcher": "starter_id"})
    first["starter_id"] = first["starter_id"].astype("string")
    return first[["game_pk", "pitcher_team", "starter_id"]]


def merge_bullpen_into_games(games: pd.DataFrame, bullpen: pd.DataFrame) -> pd.DataFrame:
    """Attach home/away bullpen fatigue features to each game row."""
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"])
    bp = bullpen.copy()
    bp["date"] = pd.to_datetime(bp["date"])

    home = bp.add_prefix("home_").rename(columns={"home_team": "home_team", "home_date": "date"})
    away = bp.add_prefix("away_").rename(columns={"away_team": "away_team", "away_date": "date"})
    g = g.merge(home, on=["date", "home_team"], how="left")
    g = g.merge(away, on=["date", "away_team"], how="left")

    for w in BULLPEN_WINDOWS:
        g[f"bp_pitches_diff_{w}d"] = (
            g[f"away_bp_pitches_last_{w}d"] - g[f"home_bp_pitches_last_{w}d"]
        )
    return g


BULLPEN_FEATURE_COLS = (
    [f"bp_pitches_diff_{w}d" for w in BULLPEN_WINDOWS]
    + [f"home_bp_pitches_last_{w}d" for w in BULLPEN_WINDOWS]
    + [f"away_bp_pitches_last_{w}d" for w in BULLPEN_WINDOWS]
)
