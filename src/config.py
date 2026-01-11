from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"
FIGURES_DIR = ASSETS_DIR / "figures"


RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


@dataclass
class Config:
    nifty_symbols: List[str] = (
        "^NSEI",
        "^NSEI.NS",
        "^NIFTY50",
        "^CNX500",
    )
    default_symbol: str = "^NSEI"
    start_date: str = "2018-01-01"
    end_date: str = None  # defaults to today in loader
    random_seed: int = 42
    lstm_window: int = 30
    lstm_features: List[str] = ("Close",)

    @property
    def raw_dir(self) -> Path:
        return RAW_DATA_DIR

    @property
    def processed_dir(self) -> Path:
        return PROCESSED_DATA_DIR

    @property
    def figures_dir(self) -> Path:
        return FIGURES_DIR


CONFIG = Config()
