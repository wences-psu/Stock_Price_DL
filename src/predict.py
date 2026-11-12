"""
predict.py - CRISP-DM phase 6: Deployment (inference).

Loads a trained model plus everything needed to use it (scalers, metadata)
and predicts the next trading day's closing price from the latest data.

Key deployment consideration: the model alone is not enough. You must apply
exactly the same preprocessing as in training with same features, same
lookback, same fitted scalers. Training saves them next to
the model (see src/train.py).

The math matches the deployment section of the single notebook.
"""

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_loader import load_stock_data
from src.evaluate import to_price
from src.features import build_inference_frame
from src.train import MODEL_FILENAME, load_run


def load_artifacts(artifact_dir: Union[str, Path]) -> dict:
    """
    Load a trained run: Keras model, both scalers and metadata.

    The model is loaded with compile=False: we only need it for predictions, so the
    optimizer and loss do not need to be restored (faster, and avoids version issues).
    """
    artifact_dir = Path(artifact_dir)
    if not (artifact_dir / MODEL_FILENAME).exists():
        raise FileNotFoundError(f"No {MODEL_FILENAME} in {artifact_dir}. Train the model first (notebook 03).")
    saved = load_run(artifact_dir.name, artifact_dir.parent)
    return {
        "model": saved.model,
        "feature_scaler": saved.feature_scaler,
        "target_scaler": saved.target_scaler,
        "metadata": saved.metadata,
        "artifact_dir": artifact_dir,
    }


def list_artifact_dirs(artifacts_dir: Union[str, Path] = config.ARTIFACTS_DIR) -> list:
    """Return every sub-folder of `artifacts/` that contains a trained model."""
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        return []
    return sorted(p for p in artifacts_dir.iterdir() if p.is_dir() and (p / MODEL_FILENAME).exists())


def predict_next_day(
    artifacts: Union[str, Path, dict],
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Predict the close of the trading day after the last row of `df`.

    Parameters
    ----------
    artifacts : artifact folder path, or the dict returned by load_artifacts().
    df        : clean Close/Volume DataFrame (from load_stock_data). Defaults to config.DATA_PATH.

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
    # Add the batch dimension: (lookback, n_features) -> (1, lookback, n_features).
    x = artifacts["feature_scaler"].transform(window)[np.newaxis, ...].astype("float32")

    # Predict (scaled) and convert back to original units, then to a price.
    pred_scaled = artifacts["model"].predict(x, verbose=0).reshape(-1, 1)
    pred_target = float(artifacts["target_scaler"].inverse_transform(pred_scaled)[0, 0])
    last_close = float(frame["Close"].iloc[-1])
    predicted_close = float(to_price(pred_target, last_close, target_mode))
    predicted_return = float(np.log(predicted_close / last_close)) if predicted_close > 0 else float("nan")

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


def simulate_recent(artifacts: Union[str, Path, dict], df: pd.DataFrame, days: int = 20) -> pd.DataFrame:
    """Replay the last `days` trading days, using only the data known on each day."""
    if not isinstance(artifacts, dict):
        artifacts = load_artifacts(artifacts)
    rows = []
    for as_of in df.index[-(days + 1):-1]:
        p = predict_next_day(artifacts, df.loc[:as_of])
        actual_next = df["Close"].iloc[df.index.get_loc(as_of) + 1]
        rows.append({"as of": as_of.date(), "last close": p["last_close"], "predicted next": p["predicted_close"],
                     "actual next": actual_next, "error ($)": p["predicted_close"] - actual_next})
    return pd.DataFrame(rows).set_index("as of")
