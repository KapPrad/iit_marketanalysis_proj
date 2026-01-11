# AI-Driven Market Trend Analysis & Forecasting System (Indian NSE)

This repository is a decision-support analytics prototype for forecasting the next-day close of the NIFTY 50 index, comparing a classical ARIMA baseline against a lightweight LSTM stub, and detecting anomalies on returns/volatility via Isolation Forest. It is **not** a trading system; it is intended for exploratory analysis and research evaluation.

## Problem statement
Provide a reproducible workflow to fetch NIFTY 50 daily data, prepare standardized time series, engineer features, benchmark forecasts (ARIMA vs. LSTM), detect anomalies, and surface results through a Streamlit dashboard.

## Architecture overview
- `src/config.py`: central configuration (symbols, dates, seeds, paths).
- `src/data_loader.py`: yfinance downloader with cached CSV + synthetic fallback.
- `src/preprocess.py`: schema enforcement, date parsing, sorting, and cleaning.
- `src/features.py`: returns/volatility features, sliding windows for LSTM.
- `src/models_arima.py`: ARIMA wrapper with naive fallback.
- `src/models_lstm.py`: stubbed LSTM wrapper with deterministic dummy preds.
- `src/anomaly.py`: Isolation Forest on returns/volatility with guards.
- `src/metrics.py`: RMSE/MAE/MAPE utilities and tabulation.
- `src/viz.py`: plotting helpers for EDA/forecast comparison.
- `dashboard/app.py`: Streamlit UI (EDA, Forecasts, Anomalies, Export).
- `notebooks/01_market_trend_forecasting.ipynb`: evaluator-facing notebook.

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

## How to run the notebook
```
cd market-trend-ai
jupyter nbconvert --to notebook --execute notebooks/01_market_trend_forecasting.ipynb --output executed.ipynb
# or open in Jupyter/VS Code and run all cells
```

## How to run Streamlit
```
cd market-trend-ai
streamlit run dashboard/app.py
```

## Folder structure
- `data/raw/`: cached downloads (timestamped CSV).
- `data/processed/`: cleaned dataset, predictions, anomalies.
- `notebooks/`: evaluation notebook.
- `src/`: modular pipeline code.
- `dashboard/`: Streamlit app.
- `assets/`: figures/screenshots for docs.

## Reproducibility notes
- Deterministic seeds set in config and used where applicable.
- Date ranges constrained in config to avoid large downloads.
- Fallback synthetic data enables fully offline runs.
- Cached raw downloads stored with timestamps; processed schema is standardized (`[Date, Open, High, Low, Close, Adj Close, Volume]`).
- Guarded imports for optional deps (statsmodels, tensorflow/keras).

## Disclaimer
This codebase is for research and decision-support analytics only. It is **not** financial advice and should not be used for live trading.
