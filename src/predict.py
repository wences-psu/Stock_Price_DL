"""
predict.py - CRISP-DM phase 6: Deployment (inference).

Loads a trained model plus everything needed to use it (scalers, metadata)
and predicts the next trading day's closing price from the latest data.

Key deployment consideration: the model alone is not enough. You must apply
exactly the same preprocessing as in training with same features, same
lookback, same fitted scalers. Training saves them next to
the model (see src/train.py).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_loader import load_stock_data
from src.features import build_inference_frame


def find_model_file(artifact_dir):
    artifact_dir = Path(artifact_dir)

    for file in artifact_dir.iterdir():
        if file.suffix in {".keras", ".h5"}:
            return file

    return None


def load_artifacts(artifact_dir: Union[str, Path]) -> dict:
    """
    Load a trained run: Keras model, both scalers and metadata.

    compile=False: we only need the model for predictions, so the optimizer
    and loss do not need to be restored (faster, and avoids version issues).
    """

    artifact_dir = Path(artifact_dir)
    model_file = find_model_file(artifact_dir)
    if model_file is None:
        raise FileNotFoundError(
            f"No model.keras or model.h5 in {artifact_dir}. Train a model first (python -m src.train)."
        )

    with open(artifact_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "model": keras.models.load_model(model_file, compile=False),
        "feature_scaler": joblib.load(artifact_dir / "feature_scaler.joblib"),
        "target_scaler": joblib.load(artifact_dir / "target_scaler.joblib"),
        "metadata": metadata,
        "artifact_dir": artifact_dir,
    }


def list_artifact_dirs(artifacts_dir: Union[str, Path] = config.ARTIFACTS_DIR) -> list:
    """Return every sub-folder of `artifacts/` that contains a trained model."""
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        return []
    return sorted(
        p for p in artifacts_dir.iterdir() if p.is_dir() and find_model_file(p) is not None
    )


def predict_next_day(
    artifacts: Union[str, Path, dict],
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Predict the close of the trading day after the last row of `df`.

    Parameters
    ----------
    artifacts : artifact folder path, or the dict returned by load_artifacts().
    df        : clean OHLCV DataFrame (from load_stock_data). Defaults to config.DATA_PATH.

    Returns
    -------
    dict with last_date, last_close, next_date (estimated), predicted_return,
    predicted_close and predicted_change_pct.
    """
    if not isinstance(artifacts, dict):
        artifacts = load_artifacts(artifacts)
    meta = artifacts["metadata"]
    lookback = meta["lookback"]
    feature_columns = meta["feature_columns"]
    target_mode = meta["target_mode"]

    if df is None:
        # Use the full file: only the most recent `lookback` rows matter here.
        df = load_stock_data(config.DATA_PATH, start_date=None, verbose=False)

    # Same feature engineering as training (keeps the latest row).
    frame = build_inference_frame(df)
    if len(frame) < lookback:
        raise ValueError(f"Need at least {lookback} rows of features, got {len(frame)}.")

    # Take the most recent window and scale it with the TRAINING scaler.
    window = frame[feature_columns].to_numpy()[-lookback:]
    window_scaled = artifacts["feature_scaler"].transform(window)

    # Add the batch dimension: (lookback, n_features) -> (1, lookback, n_features).
    x = window_scaled[np.newaxis, ...].astype("float32")

    # Predict (scaled) and convert back to original units.
    pred_scaled = artifacts["model"].predict(x, verbose=0)
    pred = float(artifacts["target_scaler"].inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])

    # Rebuild the price.
    last_close = float(frame["Close"].iloc[-1])
    if target_mode == "log_return":
        predicted_return = pred
        predicted_close = last_close * float(np.exp(pred))
    else:
        predicted_close = pred
        predicted_return = float(np.log(pred / last_close))

    last_date = frame.index[-1]
    # Next business day. Note: this ignores market holidays; use an exchange
    # calendar (e.g. pandas_market_calendars) for an exact date.
    next_date = last_date + pd.offsets.BDay(1)

    return {
        "run": meta.get("run_name", Path(artifacts["artifact_dir"]).name),
        "last_date": last_date.date().isoformat(),
        "last_close": last_close,
        "next_date": next_date.date().isoformat(),
        "predicted_return": predicted_return,
        "predicted_close": predicted_close,
        "predicted_change_pct": (predicted_close / last_close - 1) * 100,
    }

