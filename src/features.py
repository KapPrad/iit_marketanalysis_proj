from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .config import CONFIG, Config

logger = logging.getLogger(__name__)


def add_return_and_volatility(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["Volatility"] = out["Return"].rolling(window=window).std()
    out = out.fillna(method="bfill")
    return out


def create_sliding_windows(
    df: pd.DataFrame,
    target_col: str = "Close",
    window_size: int = CONFIG.lstm_window,
    config: Config = CONFIG,
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(config.random_seed)
    values = df[target_col].values
    X, y = [], []
    for idx in range(len(values) - window_size):
        X.append(values[idx : idx + window_size])
        y.append(values[idx + window_size])
    X_arr = np.array(X).reshape(-1, window_size, 1)
    y_arr = np.array(y)
    logger.info("Created sliding windows: X=%s, y=%s", X_arr.shape, y_arr.shape)
    return X_arr, y_arr


def create_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Daily_Return"] = out["Close"].pct_change()
    out["Log_Return"] = np.log(out["Close"]).diff()
    out["Roll_Mean_7"] = out["Close"].rolling(window=7).mean()
    out["Roll_Mean_14"] = out["Close"].rolling(window=14).mean()
    out["Roll_STD_7"] = out["Close"].rolling(window=7).std()
    out["Roll_STD_14"] = out["Close"].rolling(window=14).std()
    out["Close_Lag_1"] = out["Close"].shift(1)
    out["Close_Lag_7"] = out["Close"].shift(7)
    out["Close_Lag_14"] = out["Close"].shift(14)
    out = out.fillna(method="bfill")
    logger.info("Created EDA features with %d rows.", len(out))
    return out
