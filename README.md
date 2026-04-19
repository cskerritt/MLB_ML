# MLB_ML

Machine-learning pipeline that predicts the probability the home team wins an
MLB game. Pulls free Retrosheet/Statcast data via `pybaseball`, builds rolling
team-form + Elo features, and trains a calibrated XGBoost classifier.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Pull seasons of game logs (cached per-season as parquet)
python -m mlb_ml.data --start 2018 --end 2024

# 2. Build the modeling table
python -m mlb_ml.features

# 3. Train + walk-forward cross-validate
python -m mlb_ml.train

# 4. Predict for a date already present in features.parquet
python -m mlb_ml.predict --date 2024-08-01

# 5. Walk-forward backtest (refits per season, compares vs baselines)
python -m mlb_ml.backtest --train-seasons 3

# 6. Optional: ROI vs your own historical odds CSV
#    columns: date,home_team,away_team,home_ml,away_ml  (American odds)
python -m mlb_ml.backtest --odds-csv data/raw/odds_2024.csv --edge 0.04
```

Run modules from the `src/` directory or add it to `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

## Project layout

```
src/mlb_ml/
  config.py            paths + constants
  data.py              Retrosheet game-log ingestion (pybaseball)
  features.py          rolling team form, Elo, rest, park
  pitcher_features.py  rolling starting-pitcher form
  train.py             time-series CV + calibrated XGBoost
  predict.py           daily win-probability CLI
  backtest.py          walk-forward backtest + optional ROI
data/           raw + processed parquet (gitignored)
models/         saved model artifacts (gitignored)
```

## Modeling notes

- **Target:** binary `home_win` (regular-season games only after dropna).
- **Leakage guards:** rolling stats use `shift(1).rolling(...)`, Elo updates
  *after* each game, splits are time-ordered (`TimeSeriesSplit`).
- **Calibration:** isotonic via `CalibratedClassifierCV` so probabilities can
  be compared to implied moneyline odds (`p_implied = 1 / decimal_odds`).
- **Realistic ceiling:** Vegas hits ~57% on MLB sides; expect 54-57% from this
  baseline. Beating the closing line is harder than beating accuracy.

## Roadmap

- [x] Starting-pitcher rolling features (runs allowed, rest, experience)
- [x] Walk-forward backtest with optional moneyline ROI
- [ ] Statcast upgrade: pitcher FIP, K%, BB%, xwOBA against
- [ ] Bullpen fatigue (last-3-day pitch counts)
- [ ] Park factors and weather
- [ ] Lineup handedness vs. starter splits
- [ ] LightGBM benchmark + stacked ensemble
