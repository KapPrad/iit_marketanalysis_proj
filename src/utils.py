from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import CONFIG, Config

logger = logging.getLogger(__name__)


def set_seeds(seed: int = CONFIG.random_seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        logger.debug("TensorFlow not available; skipping TF seed.")


def save_predictions(
    df: pd.DataFrame,
    filename: str = "predictions.csv",
    config: Config = CONFIG,
) -> Path:
    path = Path(config.processed_dir) / filename
    df.to_csv(path, index=False)
    logger.info("Saved predictions to %s", path)
    return path


def safe_read_csv(path: Path, parse_dates: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed reading %s: %s", path, exc)
        return None
