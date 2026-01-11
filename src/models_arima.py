from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def naive_forecast(series: pd.Series) -> pd.Series:
    preds = series.shift(1)
    if preds.isna().any():
        preds.iloc[0] = series.iloc[0]
    return preds


def _naive_test_forecast(train_series: pd.Series, test_series: Optional[pd.Series], test_length: int) -> pd.Series:
    if test_series is not None and len(test_series) > 0:
        preds = []
        last_val = train_series.iloc[-1]
        for i in range(test_length):
            if i == 0:
                preds.append(last_val)
            else:
                preds.append(float(test_series.iloc[i - 1]))
        return pd.Series(preds)
    last_val = train_series.iloc[-1]
    return pd.Series([last_val] * test_length)


def arima_forecast(
    train_series: pd.Series,
    test_length: int,
    order: Tuple[int, int, int] = (5, 1, 5),
    test_series: Optional[pd.Series] = None,
) -> pd.Series:
    """Fit ARIMA on train series and forecast test length. Falls back to naive."""
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    except ImportError:
        logger.warning("statsmodels not installed; using naive forecast.")
        return _naive_test_forecast(train_series, test_series, test_length)

    try:
        model = ARIMA(train_series, order=order, trend="t")
        fitted = model.fit()
        if test_series is not None and len(test_series) == test_length:
            preds = []
            current = fitted
            for i in range(test_length):
                pred = current.forecast(steps=1)
                preds.append(float(pred.iloc[0]))
                current = current.append(pd.Series([test_series.iloc[i]]), refit=False)
            return pd.Series(preds)
        forecast = fitted.forecast(steps=test_length)
        forecast = pd.Series(forecast).reset_index(drop=True)
        return forecast
    except Exception as exc:  # noqa: BLE001
        logger.warning("ARIMA failed (%s); using naive forecast.", exc)
        return _naive_test_forecast(train_series, test_series, test_length)
