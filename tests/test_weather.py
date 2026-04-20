"""Tests for weather CSV loading and merge."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_ml.weather import (
    WEATHER_FEATURE_COLS,
    load_weather_csv,
    merge_weather_into_games,
)


def _write_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame([
        {"date": "2024-04-01", "park_id": "BOS01", "temp_f": 52, "wind_mph": 10,
         "wind_dir": "out", "precip_pct": 10, "is_dome": 0},
        {"date": "2024-04-01", "park_id": "TOR02", "temp_f": 70, "wind_mph": 0,
         "wind_dir": "calm", "precip_pct": 0, "is_dome": 1},
    ])
    path = tmp_path / "weather.csv"
    df.to_csv(path, index=False)
    return path


def test_load_weather_csv_requires_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-04-01"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_weather_csv(bad)


def test_dome_blanks_numeric_weather(tmp_path):
    w = load_weather_csv(_write_csv(tmp_path))
    tor = w[w["park_id"] == "TOR02"].iloc[0]
    assert np.isnan(tor["temp_f"])
    assert np.isnan(tor["wind_mph"])
    assert tor["is_dome"] == 1


def test_merge_adds_one_hot_wind_columns(tmp_path):
    w = load_weather_csv(_write_csv(tmp_path))
    games = pd.DataFrame({
        "date": [pd.Timestamp("2024-04-01")],
        "home_team": ["BOS"], "away_team": ["NYY"],
        "park_id": ["BOS01"], "home_runs": [4], "away_runs": [2],
        "home_win": [1], "season": [2024],
    })
    merged = merge_weather_into_games(games, w)
    for col in WEATHER_FEATURE_COLS:
        assert col in merged.columns
    assert merged["wind_dir_out"].iloc[0] == 1
    assert merged["wind_dir_in"].iloc[0] == 0
