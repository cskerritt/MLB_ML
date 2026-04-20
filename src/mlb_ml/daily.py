"""End-to-end daily refresh: data → features → train → live predictions.

Intended for CI use (see ``.github/workflows/daily_picks.yml``) but also
runnable locally. Writes picks to ``daily_picks/YYYY-MM-DD.csv`` and mirrors
the latest file to ``daily_picks/latest.csv``.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .config import FEATURES_PARQUET, GAMES_PARQUET, MODEL_PATH, ROOT
from .data import build_games_dataset
from .features import build_features
from .live import predict_live
from .train import _load, benchmark, fit_final, _available_features

log = logging.getLogger(__name__)

PICKS_DIR = ROOT / "daily_picks"
LATEST_PATH = PICKS_DIR / "latest.csv"


def ensure_games(seasons: range, refresh: bool) -> pd.DataFrame:
    if GAMES_PARQUET.exists() and not refresh:
        log.info("Using cached games at %s", GAMES_PARQUET)
        return pd.read_parquet(GAMES_PARQUET)
    log.info("Pulling Retrosheet game logs for %s..%s", seasons.start, seasons.stop - 1)
    return build_games_dataset(seasons)


def ensure_features(games: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    if FEATURES_PARQUET.exists() and not refresh:
        log.info("Using cached features at %s", FEATURES_PARQUET)
        return pd.read_parquet(FEATURES_PARQUET)
    log.info("Building features")
    return build_features(games)


def ensure_model() -> None:
    if MODEL_PATH.exists():
        log.info("Using cached model at %s", MODEL_PATH)
        return
    log.info("No cached model; benchmarking and fitting best by log loss")
    df = _load()
    leaderboard = benchmark(df, n_splits=4)
    choice = leaderboard.iloc[0]["model"]
    log.info("Chosen model: %s", choice)
    model = fit_final(df, choice)

    import joblib

    cols = _available_features(df)
    joblib.dump(
        {"model": model, "feature_cols": cols, "model_name": choice},
        MODEL_PATH,
    )


def write_picks(picks: pd.DataFrame, target: date) -> Path:
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    picks = picks.copy()
    picks["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = PICKS_DIR / f"{target.isoformat()}.csv"
    picks.to_csv(out, index=False)
    picks.to_csv(LATEST_PATH, index=False)
    log.info("Wrote %s and %s", out, LATEST_PATH)
    return out


def run(target: date, start_season: int, end_season: int, refresh: bool) -> Path:
    ensure_games(range(start_season, end_season + 1), refresh=refresh)
    # features rebuild is cheap relative to the data pull; do it every run so
    # today's rolling windows include last night's games.
    games = pd.read_parquet(GAMES_PARQUET)
    ensure_features(games, refresh=True)
    ensure_model()
    picks = predict_live(target)
    return write_picks(picks, target)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=date.today().isoformat(),
                   help="YYYY-MM-DD; defaults to today (UTC)")
    p.add_argument("--start-season", type=int, default=date.today().year - 3)
    p.add_argument("--end-season", type=int, default=date.today().year)
    p.add_argument("--refresh", action="store_true",
                   help="Ignore cached games/features parquet and rebuild")
    args = p.parse_args()
    run(date.fromisoformat(args.date), args.start_season, args.end_season, args.refresh)


if __name__ == "__main__":
    main()
