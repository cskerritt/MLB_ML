"""Tests for the scoring / track-record helpers."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from mlb_ml import score


def _picks() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-04-18", "2026-04-18", "2026-04-18"]),
        "away_team": ["BOS", "NYY", "LAN"],
        "home_team": ["TOR", "BAL", "SFN"],
        "home_win_prob": [0.40, 0.62, 0.55],
        "pick": ["AWAY", "HOME", "HOME"],
    })


def _finals() -> pd.DataFrame:
    # TOR wins (home_win=1), BAL loses (home_win=0), LAN/SFN not played.
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-04-18", "2026-04-18"]),
        "away_team": ["BOS", "NYY"],
        "home_team": ["TOR", "BAL"],
        "home_runs": [6, 2],
        "away_runs": [3, 5],
        "home_win": [1, 0],
    })


def test_grade_marks_correctness_and_brier():
    graded = score.grade(_picks(), _finals())
    tor = graded[graded["home_team"] == "TOR"].iloc[0]
    bal = graded[graded["home_team"] == "BAL"].iloc[0]
    sfn = graded[graded["home_team"] == "SFN"].iloc[0]

    # Model picked AWAY for TOR (prob 0.40 < 0.5), but home won -> incorrect.
    assert tor["model_pick_home"] == 0
    assert tor["correct"] == 0
    # Model picked HOME for BAL (prob 0.62), home lost -> incorrect.
    assert bal["correct"] == 0
    # SFN not played yet: home_win NaN -> correct/brier also NaN-equivalent.
    assert pd.isna(sfn["home_win"])


def test_summarise_ignores_unplayed_games():
    graded = score.grade(_picks(), _finals())
    summary = score.summarise(graded)
    assert summary["n_games"] == 2
    # Both BAL and TOR graded as incorrect -> accuracy 0.
    assert summary["accuracy"] == 0.0


def test_upsert_track_record_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "PICKS_DIR", tmp_path)
    monkeypatch.setattr(score, "TRACK_RECORD_PATH", tmp_path / "track_record.csv")

    graded = score.grade(_picks(), _finals())
    score._upsert_track_record(graded)
    # Re-run with the same graded rows -- should not duplicate.
    combined = score._upsert_track_record(graded)
    # Only the two played games should appear, exactly once each.
    assert len(combined) == 2

    # Now append a second date and confirm both dates persist.
    later_picks = _picks().assign(date=pd.to_datetime(["2026-04-19"] * 3))
    later_finals = _finals().assign(date=pd.to_datetime(["2026-04-19"] * 2))
    later_graded = score.grade(later_picks, later_finals)
    combined = score._upsert_track_record(later_graded)
    assert len(combined) == 4
    assert set(combined["date"].dt.date) == {date(2026, 4, 18), date(2026, 4, 19)}
