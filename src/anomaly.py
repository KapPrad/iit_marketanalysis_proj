from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.02,
    config: Config = CONFIG,
) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest on return/volatility features.
    Falls back to Z-score on Daily_Return if sklearn is unavailable.
    """
    required_cols = {"Date", "Close", "Daily_Return", "Roll_STD_7", "Roll_STD_14"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for anomaly detection: {sorted(missing)}")

    df_feat = df.copy()
    df_feat["Abs_Return"] = df_feat["Daily_Return"].abs()
    feature_cols = ["Daily_Return", "Roll_STD_7", "Roll_STD_14", "Abs_Return"]
    X = df_feat[feature_cols].fillna(0.0)

    try:
        from sklearn.ensemble import IsolationForest  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ImportError:
        logger.warning("scikit-learn not available; falling back to Z-score detection.")
        z = (df_feat["Daily_Return"] - df_feat["Daily_Return"].mean()) / (df_feat["Daily_Return"].std() + 1e-8)
        df_feat["anomaly_score"] = z
        df_feat["is_anomaly"] = (z.abs() > 3).astype(int)
        return df_feat

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        contamination=contamination,
        random_state=config.random_seed,
    )
    scores = model.fit_predict(X_scaled)
    anomaly_score = model.decision_function(X_scaled)
    df_feat["anomaly_score"] = anomaly_score
    df_feat["is_anomaly"] = (scores == -1).astype(int)
    logger.info("Detected %d anomalies.", int(df_feat["is_anomaly"].sum()))
    return df_feat


def save_anomalies(anomalies: pd.DataFrame, config: Config = CONFIG) -> Path:
    path = Path(config.processed_dir) / "anomalies.csv"
    anomalies.to_csv(path, index=False)
    logger.info("Saved anomalies to %s", path)
    return path
