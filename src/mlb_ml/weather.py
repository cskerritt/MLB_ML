"""Weather feature merge.

Free MLB weather data is scattered across paid APIs and rate-limited scrapers.
Rather than shipping a brittle scraper, this module merges a user-supplied CSV
into the games table. Fetching the data is the caller's responsibility.

Expected CSV schema (one row per game):

    date,park_id,temp_f,wind_mph,wind_dir,precip_pct,is_dome

``wind_dir`` is one of {``out``, ``in``, ``cross``, ``calm``}. We one-hot encode
it. Dome games get ``is_dome=1`` and the numeric columns are ignored (treated
as NaN so the model learns to lean on the flag).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WIND_DIRS = ("out", "in", "cross", "calm")

WEATHER_FEATURE_COLS = [
    "temp_f",
    "wind_mph",
    "precip_pct",
    "is_dome",
    *[f"wind_dir_{d}" for d in WIND_DIRS],
]


def load_weather_csv(path: str | Path) -> pd.DataFrame:
    """Load a user-supplied weather CSV and normalize column types."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "park_id", "temp_f", "wind_mph", "wind_dir", "precip_pct", "is_dome"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Weather CSV is missing columns: {sorted(missing)}")

    df["park_id"] = df["park_id"].astype(str)
    df["is_dome"] = df["is_dome"].astype(int)
    df["wind_dir"] = df["wind_dir"].fillna("calm").str.lower()
    df.loc[df["is_dome"] == 1, ["temp_f", "wind_mph", "precip_pct"]] = np.nan
    return df


def merge_weather_into_games(
    games: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """Attach weather features to games by (date, park_id). Missing -> NaN."""
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"])
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"])

    if "park_id" not in g.columns:
        # No park -> skip weather merge cleanly.
        for c in WEATHER_FEATURE_COLS:
            g[c] = np.nan
        return g

    g["park_id"] = g["park_id"].astype(str)
    dummies = pd.get_dummies(w["wind_dir"], prefix="wind_dir").astype(int)
    for d in WIND_DIRS:
        col = f"wind_dir_{d}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"wind_dir_{d}" for d in WIND_DIRS]]
    w = pd.concat([w.drop(columns=["wind_dir"]), dummies], axis=1)

    merged = g.merge(w, on=["date", "park_id"], how="left")
    return merged
