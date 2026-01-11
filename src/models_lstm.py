from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .config import CONFIG, Config
except ImportError:  # pragma: no cover - fallback for running as script
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from src.config import CONFIG, Config  # type: ignore

logger = logging.getLogger(__name__)


def _make_windows(values: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i : i + window_size])
        y.append(values[i + window_size])
    X_arr = np.array(X).reshape(-1, window_size, 1)
    y_arr = np.array(y)
    return X_arr, y_arr


def _make_windows_with_context(
    context: np.ndarray, target: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build windows for a target segment using prior context for the first window."""
    if len(target) == 0:
        return np.empty((0, window_size, 1)), np.array([])
    series = np.concatenate([context, target])
    X, y = [], []
    for i in range(len(target)):
        start = i
        end = i + window_size
        if end >= len(series):
            break
        X.append(series[start:end])
        y.append(series[end])
    X_arr = np.array(X).reshape(-1, window_size, 1)
    y_arr = np.array(y)
    return X_arr, y_arr


def naive_lstm_fallback(
    train_series: pd.Series,
    test_series: Optional[pd.Series],
    test_length: int,
) -> pd.Series:
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


def lstm_train_and_forecast(
    train_series: pd.Series,
    val_series: pd.Series,
    test_series: pd.Series,
    window_size: int = CONFIG.lstm_window,
    config: Config = CONFIG,
) -> pd.Series:
    try:
        import tensorflow as tf  # type: ignore
        from sklearn.preprocessing import MinMaxScaler  # type: ignore
    except ImportError:
        logger.warning("TensorFlow or sklearn not installed; using naive fallback.")
        return naive_lstm_fallback(train_series, test_series, len(test_series))

    try:
        tf.random.set_seed(config.random_seed)
        np.random.seed(config.random_seed)

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).ravel()
        val_scaled = scaler.transform(val_series.values.reshape(-1, 1)).ravel()
        test_scaled = scaler.transform(test_series.values.reshape(-1, 1)).ravel()

        if len(train_scaled) <= window_size or len(test_scaled) <= window_size:
            logger.warning("Series too short for window size; using naive fallback.")
            return naive_lstm_fallback(train_series, test_series, len(test_series))

        X_train, y_train = _make_windows(train_scaled, window_size)
        X_val, y_val = _make_windows(val_scaled, window_size)
        X_test, _ = _make_windows(test_scaled, window_size)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(window_size, 1)),
                tf.keras.layers.LSTM(64, activation="tanh"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=callbacks,
            shuffle=False,
        )

        # One-step-ahead rolling predictions using actual history (teacher forcing)
        history = np.concatenate([train_scaled, val_scaled]).tolist()
        preds_scaled = []
        for i in range(len(test_scaled)):
            window = np.array(history[-window_size:]).reshape(1, window_size, 1)
            yhat = model.predict(window, verbose=0).reshape(-1)[0]
            preds_scaled.append(yhat)
            history.append(float(test_scaled[i]))

        y_pred = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()
        return pd.Series(y_pred)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LSTM training failed (%s); using naive fallback.", exc)
        return naive_lstm_fallback(train_series, test_series, len(test_series))
