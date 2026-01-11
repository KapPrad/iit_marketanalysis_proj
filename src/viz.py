from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from .config import CONFIG, Config

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8")


def plot_prices(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Date"], df["Close"], label="Close", color="steelblue")
    ax.set_title("NIFTY 50 Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_forecasts(
    df: pd.DataFrame,
    arima_pred: pd.Series,
    lstm_pred: pd.Series,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Date"], df["Close"], label="Actual", color="black", linewidth=1.5)
    ax.plot(df["Date"], arima_pred, label="ARIMA/Naive", linestyle="--", color="orange")
    ax.plot(df["Date"], lstm_pred, label="LSTM Stub", linestyle=":", color="green")
    ax.set_title("Forecast Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_anomalies(df: pd.DataFrame, anomalies: Optional[pd.DataFrame]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Date"], df["Close"], label="Close", color="steelblue")
    if anomalies is not None and not anomalies.empty:
        ax.scatter(
            anomalies["Date"],
            df.set_index("Date").loc[anomalies["Date"], "Close"],
            color="red",
            label="Anomalies",
        )
    ax.set_title("Detected Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def save_figure(fig: plt.Figure, filename: str, config: Config = CONFIG) -> None:
    path = config.figures_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    logger.info("Saved figure to %s", path)


def plot_close_price(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["Date"], df["Close"], color="black", linewidth=1.2)
    ax.set_title("Close Price Over Full Range")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    fig.autofmt_xdate()
    return fig


def plot_close_with_ma(df: pd.DataFrame, window: int = 30) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["Date"], df["Close"], label="Close", color="steelblue", linewidth=1.0)
    ma = df["Close"].rolling(window=window).mean()
    ax.plot(df["Date"], ma, label=f"{window}-Day MA", color="orange")
    ax.set_title("Close Price with Moving Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_returns_hist(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["Daily_Return"], bins=50, color="slategray", alpha=0.8)
    ax.set_title("Daily Returns Histogram")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    return fig


def plot_rolling_volatility(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["Date"], df["Roll_STD_7"], label="7-Day Volatility", color="purple")
    ax.plot(df["Date"], df["Roll_STD_14"], label="14-Day Volatility", color="teal")
    ax.set_title("Rolling Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Std Dev")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_monthly_avg_returns(df: pd.DataFrame) -> plt.Figure:
    monthly = df.set_index("Date")["Daily_Return"].resample("M").mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.index, monthly.values, color="darkgreen")
    ax.set_title("Monthly Average Returns")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Return")
    fig.autofmt_xdate()
    return fig


def plot_yearly_trends(df: pd.DataFrame) -> plt.Figure:
    yearly = df.set_index("Date")["Close"].resample("Y").mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(yearly.index, yearly.values, marker="o", color="brown")
    ax.set_title("Year-wise Price Trends (Avg Close)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg Close")
    fig.autofmt_xdate()
    return fig


def plot_acf_pacf(df: pd.DataFrame, lags: int = 30) -> tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore
    except ImportError:
        logger.warning("statsmodels not available; skipping ACF/PACF plots.")
        return None, None

    fig_acf = plot_acf(df["Daily_Return"], lags=lags)
    fig_pacf = plot_pacf(df["Daily_Return"], lags=lags)
    fig_acf.suptitle("ACF of Daily Returns")
    fig_pacf.suptitle("PACF of Daily Returns")
    return fig_acf, fig_pacf


def plot_actual_vs_pred(
    dates: pd.Series,
    actual: pd.Series,
    pred: pd.Series,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, actual, label="Actual", color="black", linewidth=1.2)
    ax.plot(dates, pred, label="Forecast", color="orange", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_forecast_overlay(
    dates: pd.Series,
    actual: pd.Series,
    arima_pred: pd.Series,
    lstm_pred: pd.Series,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, actual, label="Actual", color="black", linewidth=1.2)
    ax.plot(dates, arima_pred, label="ARIMA", color="orange", linestyle="--")
    ax.plot(dates, lstm_pred, label="LSTM", color="green", linestyle=":")
    ax.set_title("Forecast Comparison on Test Set")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_test_forecast_comparison(
    dates: pd.Series,
    actual: pd.Series,
    naive: pd.Series,
    arima_pred: pd.Series,
    lstm_pred: pd.Series,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(dates.min(), dates.max(), color="#f2f2f2", alpha=0.8, label="Unseen test data")
    ax.plot(dates, actual, label="Actual Close", color="black", linewidth=1.5)
    ax.plot(dates, naive, label="Naive Baseline (t-1)", color="gray", linestyle="--")
    ax.plot(dates, arima_pred, label="ARIMA", color="blue")
    ax.plot(dates, lstm_pred, label="LSTM", color="green")
    ax.set_title("Forecast comparison on unseen test data (model never trained here)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_error_over_time(
    dates: pd.Series,
    arima_err: pd.Series,
    lstm_err: pd.Series,
    arima_mae: float,
    lstm_mae: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, arima_err, label="ARIMA Abs Error", color="blue", alpha=0.8)
    ax.plot(dates, lstm_err, label="LSTM Abs Error", color="green", alpha=0.8)
    ax.axhline(arima_mae, color="blue", linestyle="--", linewidth=1.2, label=f"ARIMA MAE ({arima_mae:.2f})")
    ax.axhline(lstm_mae, color="green", linestyle="--", linewidth=1.2, label=f"LSTM MAE ({lstm_mae:.2f})")
    ax.set_title("Absolute Error Over Test Period (Index Points)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Absolute Error")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_anomaly_error_overlay(
    dates_full: pd.Series,
    close_full: pd.Series,
    anomalies_full: pd.DataFrame,
    dates_test: pd.Series,
    test_error: pd.Series,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=False)
    ax1.plot(dates_full, close_full, label="Close", color="black")
    if "is_anomaly" in anomalies_full.columns:
        anoms = anomalies_full[anomalies_full["is_anomaly"] == 1]
        ax1.scatter(anoms["Date"], anoms["Close"], color="red", label="Anomaly", s=18)
    ax1.set_title("Close Price with Anomalies (Full Period)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close")
    ax1.legend()

    ax2.plot(dates_test, test_error, label="LSTM Abs Error", color="green")
    test_anoms = anomalies_full[anomalies_full["Date"].isin(pd.to_datetime(dates_test))]
    if not test_anoms.empty:
        for dt in test_anoms["Date"]:
            ax2.axvline(dt, color="red", linestyle=":", linewidth=1.0)
        ax2.text(
            0.01,
            0.9,
            "Red markers = anomaly dates in test window",
            transform=ax2.transAxes,
            fontsize=9,
            color="red",
        )
    else:
        ax2.text(
            0.01,
            0.9,
            "No anomaly dates in test window",
            transform=ax2.transAxes,
            fontsize=9,
            color="gray",
        )
    ax2.set_title("Test-Period Error with Anomaly Markers")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Absolute Error")
    ax2.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_anomalies_on_price(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["Date"], df["Close"], label="Close", color="black", linewidth=1.2)
    if "is_anomaly" in df.columns:
        anoms = df[df["is_anomaly"] == 1]
        ax.scatter(anoms["Date"], anoms["Close"], color="red", label="Anomaly", s=18)
    ax.set_title("Close Price with Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_anomalies_on_returns(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["Date"], df["Daily_Return"], label="Daily Return", color="steelblue")
    if "is_anomaly" in df.columns:
        anoms = df[df["is_anomaly"] == 1]
        ax.scatter(anoms["Date"], anoms["Daily_Return"], color="red", label="Anomaly", s=18)
    ax.set_title("Daily Returns with Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_anomalies_on_volatility(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["Date"], df["Roll_STD_7"], label="7-Day Volatility", color="purple")
    ax.plot(df["Date"], df["Roll_STD_14"], label="14-Day Volatility", color="teal")
    if "is_anomaly" in df.columns:
        anoms = df[df["is_anomaly"] == 1]
        ax.scatter(anoms["Date"], anoms["Roll_STD_7"], color="red", label="Anomaly", s=18)
    ax.set_title("Rolling Volatility with Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Std Dev")
    ax.legend()
    fig.autofmt_xdate()
    return fig
