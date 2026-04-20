"""Walk-forward backtest with optional moneyline ROI.

Two modes:

  1. Pure-prediction backtest: refit on an expanding window each season and
     compare model accuracy/log-loss to two baselines (always-home,
     home-form-only).

  2. Betting backtest: if the user supplies an odds CSV with columns
     ``date,home_team,away_team,home_ml,away_ml`` (American odds), simulate
     flat unit bets when the model's edge over the implied probability
     exceeds a threshold and report yield.

We deliberately do NOT ship synthetic odds. The betting numbers are only
meaningful against real historical lines.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .config import FEATURES_PARQUET, TARGET_COL
from .features import FEATURE_COLS
from .train import _available_features, _base_model

log = logging.getLogger(__name__)


def _american_to_prob(odds: float) -> float:
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def _american_to_decimal(odds: float) -> float:
    if odds < 0:
        return 1 + 100 / -odds
    return 1 + odds / 100


def walk_forward(df: pd.DataFrame, train_seasons: int = 3) -> pd.DataFrame:
    """Refit per season using the prior ``train_seasons`` years."""
    df = df.sort_values("date").reset_index(drop=True)
    cols = _available_features(df)
    seasons = sorted(df["season"].unique())
    out = []
    for s in seasons[train_seasons:]:
        train_mask = df["season"].between(s - train_seasons, s - 1)
        test_mask = df["season"] == s
        if not train_mask.any() or not test_mask.any():
            continue
        model = _base_model()
        model.fit(df.loc[train_mask, cols].values, df.loc[train_mask, TARGET_COL].values)
        proba = model.predict_proba(df.loc[test_mask, cols].values)[:, 1]
        block = df.loc[test_mask, ["date", "season", "home_team", "away_team", TARGET_COL]].copy()
        block["model_prob"] = proba
        out.append(block)
    if not out:
        raise SystemExit("Not enough seasons for walk-forward backtest.")
    return pd.concat(out, ignore_index=True)


def baseline_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Compare the model to two trivial baselines."""
    y = preds[TARGET_COL].values
    rows = []

    for name, p in (
        ("always_home", np.full(len(y), 0.54)),     # MLB home win rate ~0.54
        ("model", preds["model_prob"].values),
    ):
        rows.append({
            "strategy": name,
            "accuracy": accuracy_score(y, p > 0.5),
            "log_loss": log_loss(y, np.clip(p, 1e-4, 1 - 1e-4)),
            "brier": brier_score_loss(y, p),
        })
    return pd.DataFrame(rows)


def roi_against_odds(preds: pd.DataFrame, odds_csv: Path, edge: float = 0.04) -> pd.DataFrame:
    """Flat-unit ROI when model edge over implied prob exceeds ``edge``."""
    odds = pd.read_csv(odds_csv, parse_dates=["date"])
    odds["date"] = odds["date"].dt.date
    p = preds.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.date

    merged = p.merge(odds, on=["date", "home_team", "away_team"], how="inner")
    if merged.empty:
        raise SystemExit("Odds CSV did not match any backtest games on (date, home, away).")

    merged["home_implied"] = merged["home_ml"].apply(_american_to_prob)
    merged["away_implied"] = merged["away_ml"].apply(_american_to_prob)
    merged["home_dec"] = merged["home_ml"].apply(_american_to_decimal)
    merged["away_dec"] = merged["away_ml"].apply(_american_to_decimal)

    merged["edge_home"] = merged["model_prob"] - merged["home_implied"]
    merged["edge_away"] = (1 - merged["model_prob"]) - merged["away_implied"]

    bets = []
    for _, r in merged.iterrows():
        if r["edge_home"] >= edge:
            won = r[TARGET_COL] == 1
            pnl = (r["home_dec"] - 1) if won else -1
            bets.append({"side": "HOME", "edge": r["edge_home"], "pnl": pnl, "won": won})
        elif r["edge_away"] >= edge:
            won = r[TARGET_COL] == 0
            pnl = (r["away_dec"] - 1) if won else -1
            bets.append({"side": "AWAY", "edge": r["edge_away"], "pnl": pnl, "won": won})

    if not bets:
        raise SystemExit(f"No bets cleared the edge threshold of {edge:.2%}.")
    bets_df = pd.DataFrame(bets)
    summary = {
        "n_bets": len(bets_df),
        "win_rate": bets_df["won"].mean(),
        "total_pnl_units": bets_df["pnl"].sum(),
        "roi": bets_df["pnl"].mean(),
        "edge_threshold": edge,
    }
    log.info("Betting summary: %s", summary)
    return bets_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--train-seasons", type=int, default=3)
    p.add_argument("--odds-csv", type=Path, default=None,
                   help="Optional CSV: date,home_team,away_team,home_ml,away_ml")
    p.add_argument("--edge", type=float, default=0.04)
    args = p.parse_args()

    df = pd.read_parquet(FEATURES_PARQUET).dropna(subset=[TARGET_COL, "elo_diff"])
    preds = walk_forward(df, train_seasons=args.train_seasons)
    log.info("Backtest covers %s games over %s seasons",
             len(preds), preds["season"].nunique())

    print("\n== prediction quality ==")
    print(baseline_metrics(preds).to_string(index=False))

    if args.odds_csv:
        print("\n== betting yield ==")
        bets = roi_against_odds(preds, args.odds_csv, edge=args.edge)
        print(bets.describe().to_string())


if __name__ == "__main__":
    main()
