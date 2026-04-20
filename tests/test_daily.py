"""Smoke tests for the daily orchestrator helpers."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from mlb_ml import daily


def test_write_picks_creates_dated_and_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "PICKS_DIR", tmp_path)
    monkeypatch.setattr(daily, "LATEST_PATH", tmp_path / "latest.csv")

    picks = pd.DataFrame({
        "date": [pd.Timestamp("2026-04-19")] * 2,
        "away_team": ["BOS", "NYY"],
        "home_team": ["TOR", "BAL"],
        "home_win_prob": [0.42, 0.58],
        "pick": ["AWAY", "HOME"],
    })

    out = daily.write_picks(picks, date(2026, 4, 19))

    assert Path(out).exists()
    assert (tmp_path / "latest.csv").exists()

    written = pd.read_csv(out)
    assert "generated_at_utc" in written.columns
    assert list(written["pick"]) == ["AWAY", "HOME"]


def test_write_picks_latest_mirrors_dated(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "PICKS_DIR", tmp_path)
    monkeypatch.setattr(daily, "LATEST_PATH", tmp_path / "latest.csv")

    picks = pd.DataFrame({"home_team": ["TOR"], "home_win_prob": [0.55], "pick": ["HOME"]})
    daily.write_picks(picks, date(2026, 4, 19))

    dated = pd.read_csv(tmp_path / "2026-04-19.csv")
    latest = pd.read_csv(tmp_path / "latest.csv")
    pd.testing.assert_frame_equal(dated, latest)
