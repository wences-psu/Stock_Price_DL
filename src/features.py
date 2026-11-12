"""
features.py - CRISP-DM phase 3: Data Preparation.

Responsibilities
----------------
1. Prepare historical closing prices and trading volume.
2. Create the next-day prediction targets (closing price and log return).
3. Support six deep-learning configurations:
   LSTM, GRU, and 1D-CNN, each with two feature sets.

The models use the previous 60 trading days to predict
the next trading day's closing price.
"""

import numpy as np
import pandas as pd

import config

# Price only features.
PRICE_FEATURES = ["Close"]

# Extra inputs added when volume is included.
VOLUME_FEATURES = ["Volume"]

# All engineered features.
ENGINEERED_FEATURES = ["log_return", "log_volume", "volume_change"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered feature columns.

    Columns added
    -------------
    log_return    : ln(Close_t / Close_{t-1})
    log_volume    : ln(1 + Volume_t)
    volume_change : log_volume_t - log_volume_{t-1}
    """
    out = df.copy()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["log_volume"] = np.log1p(out["Volume"])
    out["volume_change"] = out["log_volume"].diff()
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the prediction targets for each row t, which describe day t+1.

    shift(-1) moves every value one row UP, so row t receives tomorrow's value.
    The last row gets NaN because its tomorrow is not in the data yet;
    that is the day to predict in deployment.

    Columns added
    -------------
    next_close        : tomorrow's actual close (used to evaluate in dollars).
    target_date       : tomorrow's date (used for plots).
    target_price      : target of the six main models (tomorrow's close).
    target_log_return : target of the log-return experiment, ln(Close_{t+1} / Close_t).
    """
    out = df.copy()
    out["next_close"] = out["Close"].shift(-1)
    out["target_date"] = out.index.to_series().shift(-1)
    out["target_price"] = out["next_close"]
    out["target_log_return"] = np.log(out["next_close"] / out["Close"])
    return out


def add_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features plus both targets (same function as in the single notebook)."""
    return add_target(add_features(df))


def get_feature_columns(use_volume: bool = config.USE_VOLUME) -> list:
    """
    Return the list of input columns.

    use_volume=False -> ["Close"]            (price only)
    use_volume=True  -> ["Close", "Volume"]  (price + volume)
    """
    return list(config.FEATURE_SETS["close_volume" if use_volume else "close"])


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-engineering pipeline for TRAINING/EVALUATION:
    features + targets, with incomplete rows removed.

    Removes the first row (no previous close -> no return) and the last row
    (no next day -> no target).
    """
    return add_features_and_target(df).dropna()


def build_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature pipeline for PREDICTION: same features, but keep the most recent
    row (it has no target yet - that is what we want to predict).
    """
    return add_features_and_target(df).dropna(subset=ENGINEERED_FEATURES)
