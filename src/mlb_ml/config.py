"""Project paths and configuration constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

GAMES_PARQUET = PROCESSED_DIR / "games.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
MODEL_PATH = MODELS_DIR / "xgb_winprob.joblib"

TARGET_COL = "home_win"
RANDOM_STATE = 42
