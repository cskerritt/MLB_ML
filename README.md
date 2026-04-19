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
```

Run modules from the `src/` directory or add it to `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

## Project layout

```
src/mlb_ml/
  config.py     paths + constants
  data.py       Retrosheet game-log ingestion (pybaseball)
  features.py   rolling team form, Elo, rest, park
  train.py      time-series CV + calibrated XGBoost
  predict.py    daily win-probability CLI
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

- [ ] Pitcher-specific features (rolling FIP, K%, BB%, TBF) from Statcast
- [ ] Bullpen fatigue (last-3-day pitch counts)
- [ ] Park factors and weather
- [ ] Lineup handedness vs. starter splits
- [ ] LightGBM benchmark + stacked ensemble
- [ ] Backtest betting yield against historical closing lines
