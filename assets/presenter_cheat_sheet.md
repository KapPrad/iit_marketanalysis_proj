# Presenter Cheat Sheet (Market Trend AI)

## 20s opening
"We built a decision-support prototype that forecasts next-day NIFTY 50 close, compares ARIMA vs LSTM, and flags unusual market days. This is not trading advice; it is a transparent analytics baseline."

## ARIMA vs LSTM one-liners
- ARIMA: linear, interpretable baseline that captures autocorrelation.
- LSTM: nonlinear model that can learn longer temporal patterns when data supports it.

## What errors mean (index points)
- "MAE in points" = average absolute miss in NIFTY index points on unseen test days.
- Lower MAE means tighter day-to-day tracking, not guaranteed profits.

## Why anomalies matter
- Anomalies align with volatility shocks or regime breaks.
- These are contexts where forecasts are less reliable.

## IS / IS NOT
IS:
- A reproducible analytics benchmark
- Useful for understanding market structure and risk regimes
IS NOT:
- A trading system
- A guarantee of future performance
- A causal explanation engine
