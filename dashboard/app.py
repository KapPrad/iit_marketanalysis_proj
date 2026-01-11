import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import CONFIG  # noqa: E402


st.set_page_config(page_title="Market Trend AI", layout="wide")
st.title("AI-Driven Market Trend Analysis & Forecasting (Prototype)")
st.caption("Decision-support analytics for NIFTY 50. Not financial advice.")

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Market Overview & EDA", "Forecast Comparison", "Anomaly Alerts", "Export & Reports"],
)


@st.cache_data(show_spinner=False)
def safe_read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return None


def warn_missing(name: str, path: Path) -> None:
    st.warning(f"Missing {name}: `{path}`. Run the notebook to generate it.")


def plot_close_and_volatility(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["Date"], df["Close"], label="Close", color="black", linewidth=1.2)
    ax.set_title("Close Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(df["Date"], df["Roll_Mean_7"], label="7D Mean", color="orange")
    ax2.plot(df["Date"], df["Roll_Mean_14"], label="14D Mean", color="green")
    ax2.plot(df["Date"], df["Roll_STD_7"], label="7D Volatility", color="purple")
    ax2.plot(df["Date"], df["Roll_STD_14"], label="14D Volatility", color="teal")
    ax2.set_title("Rolling Mean & Volatility")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value")
    ax2.legend()
    fig2.autofmt_xdate()
    st.pyplot(fig2)


def plot_forecasts(actual_df: pd.DataFrame, arima_df: pd.DataFrame, lstm_df: pd.DataFrame) -> None:
    fig_a, ax_a = plt.subplots(figsize=(9, 4))
    ax_a.plot(actual_df["Date"], actual_df["Close"], label="Actual", color="black")
    ax_a.plot(arima_df["Date"], arima_df["ARIMA_Pred"], label="ARIMA", color="orange", linestyle="--")
    ax_a.set_title("Actual vs ARIMA Forecast")
    ax_a.set_xlabel("Date")
    ax_a.set_ylabel("Close")
    ax_a.legend()
    fig_a.autofmt_xdate()
    st.pyplot(fig_a)

    fig_l, ax_l = plt.subplots(figsize=(9, 4))
    ax_l.plot(actual_df["Date"], actual_df["Close"], label="Actual", color="black")
    ax_l.plot(lstm_df["Date"], lstm_df["LSTM_Pred"], label="LSTM", color="green", linestyle="--")
    ax_l.set_title("Actual vs LSTM Forecast")
    ax_l.set_xlabel("Date")
    ax_l.set_ylabel("Close")
    ax_l.legend()
    fig_l.autofmt_xdate()
    st.pyplot(fig_l)

    fig_o, ax_o = plt.subplots(figsize=(9, 4))
    ax_o.plot(actual_df["Date"], actual_df["Close"], label="Actual", color="black")
    ax_o.plot(arima_df["Date"], arima_df["ARIMA_Pred"], label="ARIMA", color="orange", linestyle="--")
    ax_o.plot(lstm_df["Date"], lstm_df["LSTM_Pred"], label="LSTM", color="green", linestyle=":")
    ax_o.set_title("Overlay Forecast Comparison")
    ax_o.set_xlabel("Date")
    ax_o.set_ylabel("Close")
    ax_o.legend()
    fig_o.autofmt_xdate()
    st.pyplot(fig_o)


def plot_anomalies(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["Date"], df["Close"], label="Close", color="black")
    if "is_anomaly" in df.columns:
        anoms = df[df["is_anomaly"] == 1]
        ax.scatter(anoms["Date"], anoms["Close"], color="red", label="Anomaly", s=20)
    ax.set_title("Close Price with Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)


market_features_path = CONFIG.processed_dir / "market_features.csv"
train_path = CONFIG.processed_dir / "train.csv"
val_path = CONFIG.processed_dir / "val.csv"
test_path = CONFIG.processed_dir / "test.csv"
arima_path = CONFIG.processed_dir / "arima_predictions.csv"
lstm_path = CONFIG.processed_dir / "lstm_predictions.csv"
metrics_path = CONFIG.processed_dir / "metrics.csv"
anomalies_path = CONFIG.processed_dir / "anomalies.csv"


if section == "Market Overview & EDA":
    st.subheader("Market Overview & EDA")
    df = safe_read_csv(market_features_path, parse_dates=["Date"])
    if df is None or df.empty:
        warn_missing("market_features.csv", market_features_path)
    else:
        st.write(
            "This section summarizes market behavior using close prices, rolling mean, and volatility to "
            "highlight trends and regime changes."
        )
        plot_close_and_volatility(df)
        st.dataframe(df.tail(10))


if section == "Forecast Comparison":
    st.subheader("Forecast Comparison")
    st.info(
        "How to read: the naive baseline (yesterday's price) is the minimum bar to beat. "
        "All plots use test-only data to avoid training leakage."
    )
    test_df = safe_read_csv(test_path, parse_dates=["Date"])
    arima_df = safe_read_csv(arima_path, parse_dates=["Date"])
    lstm_df = safe_read_csv(lstm_path, parse_dates=["Date"])
    metrics_df = safe_read_csv(metrics_path)

    missing_any = False
    if test_df is None or test_df.empty:
        warn_missing("test.csv", test_path)
        missing_any = True
    if arima_df is None or arima_df.empty:
        warn_missing("arima_predictions.csv", arima_path)
        missing_any = True
    if lstm_df is None or lstm_df.empty:
        warn_missing("lstm_predictions.csv", lstm_path)
        missing_any = True

    if not missing_any:
        plot_forecasts(test_df, arima_df, lstm_df)

    if metrics_df is None or metrics_df.empty:
        warn_missing("metrics.csv", metrics_path)
    else:
        st.dataframe(metrics_df)


if section == "Anomaly Alerts":
    st.subheader("Anomaly Alerts")
    st.info(
        "Anomalies flag dates with unusual returns or volatility. "
        "They indicate potential regime shifts, not trading signals."
    )
    anomalies_df = safe_read_csv(anomalies_path, parse_dates=["Date"])
    if anomalies_df is None or anomalies_df.empty:
        warn_missing("anomalies.csv", anomalies_path)
    else:
        min_date = anomalies_df["Date"].min()
        max_date = anomalies_df["Date"].max()
        date_range = st.date_input("Filter by date range", (min_date, max_date))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered = anomalies_df[(anomalies_df["Date"] >= start) & (anomalies_df["Date"] <= end)]
        else:
            filtered = anomalies_df
        st.dataframe(filtered[["Date", "Close", "Daily_Return", "Roll_STD_7", "Roll_STD_14", "anomaly_score"]])
        plot_anomalies(anomalies_df)


if section == "Export & Reports":
    st.subheader("Export & Reports")
    arima_df = safe_read_csv(arima_path, parse_dates=["Date"])
    lstm_df = safe_read_csv(lstm_path, parse_dates=["Date"])
    metrics_df = safe_read_csv(metrics_path)
    anomalies_df = safe_read_csv(anomalies_path, parse_dates=["Date"])

    if arima_df is not None and not arima_df.empty:
        st.download_button("Download ARIMA Predictions", arima_df.to_csv(index=False), file_name="arima_predictions.csv")
    else:
        warn_missing("arima_predictions.csv", arima_path)

    if lstm_df is not None and not lstm_df.empty:
        st.download_button("Download LSTM Predictions", lstm_df.to_csv(index=False), file_name="lstm_predictions.csv")
    else:
        warn_missing("lstm_predictions.csv", lstm_path)

    if metrics_df is not None and not metrics_df.empty:
        st.download_button("Download Metrics", metrics_df.to_csv(index=False), file_name="metrics.csv")
    else:
        warn_missing("metrics.csv", metrics_path)

    if anomalies_df is not None and not anomalies_df.empty:
        st.download_button("Download Anomalies", anomalies_df.to_csv(index=False), file_name="anomalies.csv")
    else:
        warn_missing("anomalies.csv", anomalies_path)
