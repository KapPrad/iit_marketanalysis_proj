from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import CONFIG, Config

logger = logging.getLogger(__name__)


def _fetch_from_yfinance(
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        logger.warning("yfinance not installed; falling back to synthetic data.")
        return None

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        if df is None or df.empty:
            logger.warning("Empty dataframe from yfinance for symbol %s", symbol)
            return None
        df = df.reset_index()
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Download failed for %s: %s", symbol, exc)
        return None


def _synthetic_data(n_days: int = 2600) -> pd.DataFrame:
    logger.warning("Generating synthetic dataset with %d days (offline fallback).", n_days)
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq="B")
    base = np.linspace(100, 120, num=n_days) + np.random.normal(0, 1, n_days).cumsum()
    volatility = np.abs(np.random.normal(0.5, 0.1, n_days))
    noise = np.random.normal(0, volatility)
    close = base + noise
    open_ = close + np.random.normal(0, 0.5, n_days)
    high = np.maximum(open_, close) + np.abs(np.random.normal(0.5, 0.2, n_days))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0.5, 0.2, n_days))
    adj_close = close * (1 - np.random.normal(0.001, 0.0005, n_days))
    volume = np.random.randint(1_000_000, 3_000_000, n_days)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adj_close,
            "Volume": volume,
        }
    )


def download_data(
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Config = CONFIG,
) -> pd.DataFrame:
    symbol = symbol or config.default_symbol
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
    end_date = end_date or config.end_date

    df = _fetch_from_yfinance(symbol, start_date, end_date)
    if df is None:
        logger.warning("Falling back to synthetic data for symbol %s.", symbol)
        df = _synthetic_data(n_days=2600)
    else:
        logger.info("Downloaded %d rows for %s.", len(df), symbol)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = config.raw_dir / f"nifty50_raw_{timestamp}.csv"
    df.to_csv(raw_path, index=False)
    logger.info("Saved raw data to %s", raw_path)
    return df


def load_cached_or_download(
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Config = CONFIG,
) -> pd.DataFrame:
    recent_files = sorted(Path(config.raw_dir).glob("*.csv"), reverse=True)
    if recent_files:
        latest = recent_files[0]
        logger.info("Loading cached data from %s", latest)
        return pd.read_csv(latest)
    return download_data(symbol=symbol, start_date=start_date, end_date=end_date, config=config)
