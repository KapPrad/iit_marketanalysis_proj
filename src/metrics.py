from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    metrics = {
        "Model": model_name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
    logger.info("Metrics for %s: %s", model_name, metrics)
    return pd.DataFrame([metrics])


def infer_prediction_column(df: pd.DataFrame) -> str:
    """Infer prediction column name from a dataframe with possible schema differences."""
    lowered = {c: c.lower() for c in df.columns}
    candidates = []
    for col, low in lowered.items():
        if "date" in low or "close" in low or "actual" in low:
            continue
        if "pred" in low or "forecast" in low:
            candidates.append(col)
    if candidates:
        return candidates[0]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if "close" not in lowered[c] and "actual" not in lowered[c]]
    if numeric_cols:
        return numeric_cols[-1]
    raise ValueError("No prediction column could be inferred.")


def absolute_error_series(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    return (y_true - y_pred).abs()
