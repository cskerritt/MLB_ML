"""Statcast pitch-level ingestion with on-disk caching.

One season is ~700k rows; we cache each season as parquet so feature builds
are fast on repeat runs. Call ``load_statcast(seasons)`` to get a single
concatenated dataframe.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from .config import RAW_DIR

log = logging.getLogger(__name__)

# Columns we actually use downstream. Keeping the projection narrow keeps the
# parquet cache small and the merge steps fast.
KEEP_COLS = [
    "game_date",
    "game_pk",
    "pitcher",
    "player_name",
    "p_throws",
    "stand",
    "pitch_type",
    "events",
    "description",
    "type",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "home_team",
    "away_team",
    "inning_topbot",
]


def _import_pybaseball():
    try:
        import pybaseball  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "pybaseball is required. Install with `pip install -r requirements.txt`."
        ) from e
    return pybaseball


def fetch_statcast_season(season: int) -> pd.DataFrame:
    """Pull one regular season of Statcast pitches via pybaseball."""
    pyb = _import_pybaseball()
    start = f"{season}-03-15"
    end = f"{season}-11-15"
    log.info("Pulling Statcast %s -> %s", start, end)
    df = pyb.statcast(start_dt=start, end_dt=end)
    cols = [c for c in KEEP_COLS if c in df.columns]
    return df[cols].copy()


def load_statcast(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """Load (and cache) Statcast pitches for the requested seasons."""
    frames = []
    for s in seasons:
        cache = RAW_DIR / f"statcast_{s}.parquet"
        if cache.exists() and not refresh:
            log.info("Loading cached %s", cache.name)
            frames.append(pd.read_parquet(cache))
            continue
        df = fetch_statcast_season(s)
        df.to_parquet(cache, index=False)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out
