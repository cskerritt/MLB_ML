"""Today's schedule + live predictions.

Pulls today's MLB schedule and probable pitchers from the public MLB Stats
API, looks up each starter's most-recent rolling features from the cached
feature table, then runs the trained model to emit home-win probabilities.

The public endpoint used::

    https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher

No API key is required. If the API is unreachable we fail loudly instead of
fabricating games.
"""
from __future__ import annotations

import argparse
import logging
import urllib.request
import json
from datetime import date
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .config import FEATURES_PARQUET, MODEL_PATH

log = logging.getLogger(__name__)

SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&date={date}&hydrate=probablePitcher(note)"
)

# MLB Stats API abbreviations sometimes differ from Retrosheet ones we use
# internally. Extend as needed. Unknown teams pass through unchanged.
TEAM_ALIAS = {
    "AZ": "ARI",
    "WSH": "WAS",
    "CWS": "CHW",
    "SF": "SFN",
    "SD": "SDN",
    "LAD": "LAN",
    "LAA": "ANA",
    "TB": "TBA",
    "KC": "KCA",
    "NYM": "NYN",
    "NYY": "NYA",
    "STL": "SLN",
    "CHC": "CHN",
}


_NEEDS_IMPUTE = {"logit", "stacked"}


def _normalise_team(abbr: str) -> str:
    return TEAM_ALIAS.get(abbr, abbr)


def fetch_schedule(target: date) -> list[dict[str, Any]]:
    url = SCHEDULE_URL.format(date=target.isoformat())
    log.info("GET %s", url)
    with urllib.request.urlopen(url, timeout=15) as resp:
        payload = json.load(resp)

    games: list[dict[str, Any]] = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {})
            home_team = teams.get("home", {}).get("team", {}).get("abbreviation", "")
            away_team = teams.get("away", {}).get("team", {}).get("abbreviation", "")
            home_pp = teams.get("home", {}).get("probablePitcher", {})
            away_pp = teams.get("away", {}).get("probablePitcher", {})
            games.append({
                "game_pk": g.get("gamePk"),
                "date": pd.Timestamp(target),
                "home_team": _normalise_team(home_team),
                "away_team": _normalise_team(away_team),
                "home_sp_id": str(home_pp.get("id")) if home_pp else None,
                "home_sp_name": home_pp.get("fullName") if home_pp else None,
                "away_sp_id": str(away_pp.get("id")) if away_pp else None,
                "away_sp_name": away_pp.get("fullName") if away_pp else None,
            })
    return games


def _latest_feature_snapshot(
    features: pd.DataFrame, team: str, as_of: pd.Timestamp
) -> pd.Series | None:
    """Most-recent feature row for ``team`` (as home or away) strictly before ``as_of``."""
    mask = (features["date"] < as_of) & (
        (features["home_team"] == team) | (features["away_team"] == team)
    )
    prior = features[mask]
    if prior.empty:
        return None
    return prior.sort_values("date").iloc[-1]


def _assemble_row(
    schedule_row: dict[str, Any],
    features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Build a single feature vector for a scheduled game.

    Uses each team's most-recent feature snapshot (as home or away side) as a
    proxy for today's inputs. Team-specific "home_*" / "away_*" columns are
    re-labeled to match today's side assignment.
    """
    home_snap = _latest_feature_snapshot(features, schedule_row["home_team"],
                                         schedule_row["date"])
    away_snap = _latest_feature_snapshot(features, schedule_row["away_team"],
                                         schedule_row["date"])
    row = pd.Series({c: np.nan for c in feature_cols})

    def _copy_side(snap: pd.Series, team: str, prefix: str) -> None:
        if snap is None:
            return
        # Figure out which side the team played in the snapshot.
        snap_side = "home" if snap["home_team"] == team else "away"
        for col in feature_cols:
            if not col.startswith(f"{prefix}_"):
                continue
            source_col = col.replace(f"{prefix}_", f"{snap_side}_", 1)
            if source_col in snap.index and pd.notna(snap[source_col]):
                row[col] = snap[source_col]

    _copy_side(home_snap, schedule_row["home_team"], "home")
    _copy_side(away_snap, schedule_row["away_team"], "away")

    # Symmetric diff / shared columns: use home_snap by default.
    if home_snap is not None:
        for col in feature_cols:
            if col.startswith("home_") or col.startswith("away_"):
                continue
            if col in home_snap.index and pd.notna(home_snap[col]):
                row[col] = home_snap[col]
    return row


def _load_bundle():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"], bundle.get("model_name", "xgb")


def predict_live(target: date) -> pd.DataFrame:
    model, feature_cols, model_name = _load_bundle()
    features = pd.read_parquet(FEATURES_PARQUET)
    features["date"] = pd.to_datetime(features["date"])

    schedule = fetch_schedule(target)
    if not schedule:
        raise SystemExit(f"MLB Stats API returned no games for {target}.")

    rows = [_assemble_row(g, features, feature_cols) for g in schedule]
    X = pd.DataFrame(rows, columns=feature_cols).values
    if model_name in _NEEDS_IMPUTE:
        X = SimpleImputer(strategy="median").fit_transform(X)

    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame(schedule)
    out["home_win_prob"] = proba
    out["pick"] = np.where(proba >= 0.5, "HOME", "AWAY")
    return out[[
        "date", "away_team", "away_sp_name",
        "home_team", "home_sp_name",
        "home_win_prob", "pick",
    ]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=date.today().isoformat(),
                   help="YYYY-MM-DD; defaults to today")
    args = p.parse_args()
    target = date.fromisoformat(args.date)
    out = predict_live(target)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
