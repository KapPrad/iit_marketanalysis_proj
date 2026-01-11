# AI-Driven Market Trend Analysis & Forecasting System (Indian NSE)
# Project Summary: Prototype for decision-support analytics on the NIFTY 50 index combining classical (ARIMA) and deep (LSTM) forecasting with anomaly detection on returns/volatility. Not intended for live trading.

# 1. Problem Definition & Objective
# - Forecast next-day Close price for NIFTY 50.
# - Compare ARIMA baseline vs. LSTM stub (Phase 1 placeholder).
# - Detect anomalies via Isolation Forest on returns and volatility.
# - Deliver evaluator-friendly, reproducible workflow.

# 2. Data Understanding & Preparation
# - Source: yfinance (preferred) with cached CSV; synthetic fallback ensures offline runs.
# - Schema enforced: [Date, Open, High, Low, Close, Adj Close, Volume].
# - Deterministic seeds for reproducibility.

import sys
from pathlib import Path
import warnings
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import CONFIG
from src import data_loader, preprocess, features, models_arima, models_lstm, anomaly, metrics, viz, utils

warnings.filterwarnings("ignore")
utils.set_seeds(CONFIG.random_seed)

# Load or download data
raw_df = data_loader.load_cached_or_download(
    symbol=CONFIG.default_symbol,
    start_date=CONFIG.start_date,
    end_date=CONFIG.end_date,
)
processed_df = preprocess.standardize_schema(raw_df)
preprocess.preprocess_and_save(processed_df, filename="processed.csv")
df = processed_df.copy()
print(df.head())

# 3. Exploratory Data Analysis
fig_price = viz.plot_prices(df)
plt.show()
print(df.tail())

# 4. Feature Engineering
# TODO: Add richer macro/sector signals (Phase 2).
df_feat = features.add_return_and_volatility(df)
print(df_feat[["Date", "Close", "Return", "Volatility"]].tail())

# 5. Baseline Forecasting Model (ARIMA)
# TODO: Tune orders, diagnostics, backtesting (Phase 2/3).
arima_pred = models_arima.arima_forecast(df_feat["Close"])
arima_pred = arima_pred.fillna(method="bfill")
arima_pred.name = "ARIMA_Pred"
print(arima_pred.head())

# 6. Deep Learning Forecasting Model (LSTM)
# TODO: Implement proper train/val split, scaling, and checkpointing (Phase 2+).
lstm_pred = models_lstm.lstm_forecast(df_feat["Close"], window_size=CONFIG.lstm_window)
lstm_pred = lstm_pred.fillna(method="bfill")
lstm_pred.name = "LSTM_Pred"
print(lstm_pred.head())

# 7. Anomaly Detection (Isolation Forest)
# TODO: Add explainability (e.g., SHAP on features) in later phase.
anomalies = anomaly.detect_anomalies(df_feat)
anomaly.save_anomalies(anomalies)
print(anomalies.head())

# 8. Model Evaluation & Comparison
metrics_arima = metrics.evaluate_predictions(df_feat["Close"], arima_pred, "ARIMA/Naive")
metrics_lstm = metrics.evaluate_predictions(df_feat["Close"], lstm_pred, "LSTM Stub")
metrics_table = pd.concat([metrics_arima, metrics_lstm], ignore_index=True)
print(metrics_table)

# Save predictions for dashboard
from src.utils import save_predictions

preds_df = pd.DataFrame(
    {
        "Date": df_feat["Date"],
        "Actual_Close": df_feat["Close"],
        "ARIMA_Pred": arima_pred,
        "LSTM_Pred": lstm_pred,
    }
)
save_predictions(preds_df)

fig_forecasts = viz.plot_forecasts(df_feat, arima_pred, lstm_pred)
plt.show()

fig_anoms = viz.plot_anomalies(df_feat, anomalies)
plt.show()

# 9. Dashboard (Streamlit) Overview
# - Run with: streamlit run dashboard/app.py
# - Notebook-generated processed/predictions/anomalies feed the UI.

# 10. Ethical Considerations & Responsible AI
# - This is an analytics prototype; do not use for live trading.
# - TODO: Add drift monitoring, model cards, and bias checks (Phase 3).

# 11. Conclusion & Future Scope
# - Established reproducible scaffold with runnable placeholders.
# - Next steps: robust backtesting, hyperparameter tuning, feature expansion, richer anomaly narratives.

# Appendix: Reproducibility Notes
# - Seeds fixed via CONFIG.random_seed.
# - Synthetic fallback ensures offline execution.
# - Cached raw downloads stored in data/raw/ with timestamps.
# - Processed schema standardized for downstream modules.
