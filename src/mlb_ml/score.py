"""Grade prior-day picks against final scores from the MLB Stats API.

Given a date, this module:

  1. Loads ``daily_picks/<date>.csv`` (produced by ``daily.py``).
  2. Fetches that date's final scores from ``statsapi.mlb.com``.
  3. Writes ``daily_picks/<date>_scored.csv`` with outcome + correct columns.
  4. Appends per-game results to ``daily_picks/track_record.csv`` for a
     running win rate / log-loss / Brier tally.

The scorer is idempotent: re-running for the same date replaces the scored
CSV but de-dupes rows in ``track_record.csv`` keyed on (date, home_team,
away_team) so the running tally stays accurate.
"""
from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ROOT
from .live import _normalise_team

log = logging.getLogger(__name__)

PICKS_DIR = ROOT / "daily_picks"
TRACK_RECORD_PATH = PICKS_DIR / "track_record.csv"
SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&date={date}"
)

FINAL_CODES = {"F", "FT", "FR"}  # Final, Final-tiebreaker, Final-rain-shortened


def fetch_final_scores(target: date) -> pd.DataFrame:
    """Return a dataframe of completed games with final scores for ``target``."""
    url = SCHEDULE_URL.format(date=target.isoformat())
    log.info("GET %s", url)
    with urllib.request.urlopen(url, timeout=15) as resp:
        payload = json.load(resp)

    rows: list[dict[str, Any]] = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            status = g.get("status", {})
            code = status.get("statusCode", "")
            if code not in FINAL_CODES:
                continue
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            rows.append({
                "date": pd.Timestamp(target),
                "home_team": _normalise_team(home.get("team", {}).get("abbreviation", "")),
                "away_team": _normalise_team(away.get("team", {}).get("abbreviation", "")),
                "home_runs": home.get("score"),
                "away_runs": away.get("score"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["home_win"] = (df["home_runs"] > df["away_runs"]).astype(int)
    return df


def grade(picks: pd.DataFrame, finals: pd.DataFrame) -> pd.DataFrame:
    """Attach home_runs, away_runs, home_win, correct, brier to each pick."""
    if finals.empty:
        raise SystemExit("No completed games found for that date.")

    merged = picks.merge(
        finals, on=["home_team", "away_team"], how="left", suffixes=("", "_final"),
    )
    merged["home_win"] = merged["home_win"].astype("Int64")
    merged["model_pick_home"] = (merged["home_win_prob"] >= 0.5).astype("Int64")
    merged["correct"] = (merged["model_pick_home"] == merged["home_win"]).astype("Int64")
    merged["brier"] = (merged["home_win_prob"] - merged["home_win"]).pow(2)
    return merged


def summarise(graded: pd.DataFrame) -> dict[str, float]:
    """Summary metrics over a graded dataframe; ignores un-played games."""
    played = graded.dropna(subset=["home_win"])
    if played.empty:
        return {"n_games": 0}
    p = np.clip(played["home_win_prob"].values, 1e-4, 1 - 1e-4)
    y = played["home_win"].astype(int).values
    return {
        "n_games": int(len(played)),
        "accuracy": float(played["correct"].mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(played["brier"].mean()),
    }


def _upsert_track_record(graded: pd.DataFrame) -> pd.DataFrame:
    """Merge new graded rows into the running track record, de-duping on key."""
    key = ["date", "home_team", "away_team"]
    new = graded.dropna(subset=["home_win"]).copy()
    if TRACK_RECORD_PATH.exists():
        prior = pd.read_csv(TRACK_RECORD_PATH, parse_dates=["date"])
        combined = pd.concat([prior, new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=key, keep="last").sort_values("date")
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TRACK_RECORD_PATH, index=False)
    return combined


def score_date(target: date) -> Path:
    picks_path = PICKS_DIR / f"{target.isoformat()}.csv"
    if not picks_path.exists():
        raise SystemExit(f"No picks file for {target} at {picks_path}.")
    picks = pd.read_csv(picks_path, parse_dates=["date"])
    finals = fetch_final_scores(target)
    graded = grade(picks, finals)

    out = PICKS_DIR / f"{target.isoformat()}_scored.csv"
    graded.to_csv(out, index=False)
    log.info("Wrote %s", out)

    combined = _upsert_track_record(graded)
    summary_today = summarise(graded)
    summary_all = summarise(combined)
    log.info("Today:   %s", summary_today)
    log.info("Overall: %s", summary_all)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str,
                   default=(date.today() - timedelta(days=1)).isoformat(),
                   help="YYYY-MM-DD; defaults to yesterday (UTC)")
    args = p.parse_args()
    score_date(date.fromisoformat(args.date))


if __name__ == "__main__":
    main()
