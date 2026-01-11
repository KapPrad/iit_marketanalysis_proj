from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

from .config import CONFIG, Config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe conforms to the unified schema and is sorted by date."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate raw market data for downstream processing."""
    df = standardize_schema(df)
    df["Open"] = df["Open"].ffill()
    df["High"] = df["High"].ffill()
    df["Low"] = df["Low"].ffill()
    df["Close"] = df["Close"].ffill()
    df["Adj Close"] = df["Adj Close"].ffill()
    df["Volume"] = df["Volume"].fillna(0)
    df = df.drop_duplicates(subset=["Date"]).reset_index(drop=True)

    if df["Close"].isna().any():
        raise ValueError("Close contains nulls after cleaning.")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Dates are not strictly increasing.")
    if len(df) < 2500:
        logger.warning("Dataset length is %d (<2500). Likely synthetic or short history.", len(df))
    return df


def time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def preprocess_and_save(
    df: pd.DataFrame,
    filename: str = "processed.csv",
    config: Config = CONFIG,
) -> Path:
    processed_df = clean_data(df)
    output_path = Path(config.processed_dir) / filename
    processed_df.to_csv(output_path, index=False)
    logger.info("Saved processed data to %s", output_path)
    return output_path


def load_processed(config: Config = CONFIG) -> Optional[pd.DataFrame]:
    candidates: Iterable[Path] = sorted(Path(config.processed_dir).glob("*.csv"), reverse=True)
    for path in candidates:
        if path.name.startswith("predictions") or path.name.startswith("anomalies"):
            continue
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            logger.info("Loaded processed data from %s", path)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", path, exc)
    logger.info("No processed data found.")
    return None
