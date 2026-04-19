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

# 2a. Build the modeling table (team form + Elo + simple pitcher)
python -m mlb_ml.features

# 2b. Optional: enrich with Statcast advanced pitcher metrics + bullpen fatigue
#     (first run downloads ~700k pitches per season; cached to data/raw/)
python -m mlb_ml.features --statcast-start 2022 --statcast-end 2024

# 2c. Optional: merge weather data from your own CSV
#     (schema: date,park_id,temp_f,wind_mph,wind_dir,precip_pct,is_dome)
python -m mlb_ml.features --weather-csv data/raw/weather.csv

# 3. Benchmark XGBoost vs LightGBM vs logistic vs stacked, save the best
python -m mlb_ml.train                  # auto-picks best by log loss
python -m mlb_ml.train --model stacked  # or force a specific model

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
  statcast.py          Statcast pitch-level pulls (cached per season)
  features.py          feature orchestrator: team form + Elo + all layers
  pitcher_features.py  rolling simple starting-pitcher form
  pitcher_adv.py       Statcast advanced metrics (xwOBA, K%, BB%, whiff%)
  bullpen.py           bullpen fatigue (last-1d/3d reliever pitch counts)
  park_factors.py      lagged per-park runs/game factor
  weather.py           user-supplied weather CSV merge
  models.py            model zoo: xgb, lgbm, logistic, stacked ensemble
  train.py             benchmark + calibrated fit of the selected model
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
- [x] Statcast advanced pitcher metrics (xwOBA against, K%, BB%, whiff%)
- [x] Bullpen fatigue (last-1d / last-3d reliever pitch counts)
- [x] Park factors (lagged one season) and user-supplied weather merge
- [x] LightGBM benchmark + stacked ensemble
- [ ] Lineup handedness vs. starter splits
- [ ] Hyperparameter search (Optuna) on the top model
