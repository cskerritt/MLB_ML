"""Data ingestion from pybaseball into a local parquet store.

Pulls Retrosheet game logs (final scores, starting pitchers, parks) for a
range of seasons and writes them to ``data/processed/games.parquet``.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from .config import GAMES_PARQUET, RAW_DIR

log = logging.getLogger(__name__)

GAME_LOG_COLS = [
    "Date",
    "HomeTeam",
    "VisitingTeam",
    "HomeRunsScore",
    "VisitorRunsScored",
    "HomeStartingPitcherID",
    "VisitorStartingPitcherID",
    "HomeStartingPitcherName",
    "VisitorStartingPitcherName",
    "ParkID",
    "DayNight",
    "HomeManagerName",
    "VisitorManagerName",
    "HomePlateUmpireID",
    "HomePlateUmpireName",
]


def _import_pybaseball():
    try:
        import pybaseball  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "pybaseball is required. Install with `pip install -r requirements.txt`."
        ) from e
    return pybaseball


def fetch_season_games(season: int) -> pd.DataFrame:
    """Fetch one season of Retrosheet game logs."""
    pyb = _import_pybaseball()
    log.info("Fetching Retrosheet game logs for %s", season)
    df = pyb.retrosheet.season_game_logs(season)
    df = df.copy()
    df["season"] = season
    return df


def build_games_dataset(seasons: Iterable[int], cache: bool = True) -> pd.DataFrame:
    """Concatenate game logs across seasons and persist to parquet."""
    frames = []
    for s in seasons:
        cache_path = RAW_DIR / f"gamelogs_{s}.parquet"
        if cache and cache_path.exists():
            log.info("Loading cached %s", cache_path.name)
            frames.append(pd.read_parquet(cache_path))
            continue
        df = fetch_season_games(s)
        df.to_parquet(cache_path, index=False)
        frames.append(df)

    games = pd.concat(frames, ignore_index=True)
    games = _normalize(games)
    games.to_parquet(GAMES_PARQUET, index=False)
    log.info("Wrote %s rows to %s", len(games), GAMES_PARQUET)
    return games


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Trim to the columns we need and add the binary target."""
    keep = [c for c in GAME_LOG_COLS if c in df.columns] + ["season"]
    out = df[keep].copy()
    out = out.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "VisitingTeam": "away_team",
            "HomeRunsScore": "home_runs",
            "VisitorRunsScored": "away_runs",
            "HomeStartingPitcherID": "home_sp_id",
            "VisitorStartingPitcherID": "away_sp_id",
            "HomeStartingPitcherName": "home_sp_name",
            "VisitorStartingPitcherName": "away_sp_name",
            "ParkID": "park_id",
            "DayNight": "day_night",
            "HomePlateUmpireID": "ump_id",
            "HomePlateUmpireName": "ump_name",
        }
    )
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d", errors="coerce")
    out = out.dropna(subset=["date", "home_team", "away_team"])
    out["home_runs"] = pd.to_numeric(out["home_runs"], errors="coerce")
    out["away_runs"] = pd.to_numeric(out["away_runs"], errors="coerce")
    out = out.dropna(subset=["home_runs", "away_runs"])
    out["home_win"] = (out["home_runs"] > out["away_runs"]).astype(int)
    out = out.sort_values("date").reset_index(drop=True)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2018)
    p.add_argument("--end", type=int, default=2024)
    args = p.parse_args()

    build_games_dataset(range(args.start, args.end + 1))
