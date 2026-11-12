"""
features.py - CRISP-DM phase 3: Data Preparation.

Responsibilities
----------------
1. Prepare historical closing prices and trading volume.
2. Create the next-day prediction target (log return by default, see config.TARGET_MODE).
3. Support six deep-learning configurations:
   LSTM, GRU, and 1D-CNN, each with two feature sets.

The models use the previous 60 trading days to predict the next trading day's
log return, which is converted back to the next day's closing price.
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


def add_features(df: pd.DataFrame, ticker: str = config.TICKER) -> pd.DataFrame:
    """
    Add engineered feature columns.

    ticker: used for selecting the MultiIndex column.

    Columns added
    -------------
    log_return    : ln(Close_t / Close_{t-1})
    log_volume    : ln(1 + Volume_t)
    volume_change : log_volume_t - log_volume_{t-1}
    """
    out = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        out["log_return", ticker] = np.log(out["Close", ticker] / out["Close", ticker].shift(1))
        out["log_volume", ticker] = np.log1p(out["Volume", ticker])
        out["volume_change", ticker] = out["log_volume", ticker].diff()
    else:
        out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
        out["log_volume"] = np.log1p(out["Volume"])
        out["volume_change"] = out["log_volume"].diff()
    return out


def get_feature_columns(use_volume: bool) -> list:
    """
    Return the list of input columns.

    use_volume=False -> ["Close"]                                  (price only)
    use_volume=True  -> ["Close", "Volume"]   (price + volume)
    """
    columns = PRICE_FEATURES.copy()
    if use_volume:
        columns += VOLUME_FEATURES
    return columns


def add_target(
    df: pd.DataFrame, ticker: str = config.TICKER, target_mode: str = config.TARGET_MODE
) -> pd.DataFrame:
    """
    Add the prediction target for each row t, which describes day t+1.

    shift(-1) moves every value one row UP, so row t receives tomorrow's value.
    The last row gets NaN because its tomorrow is not in the data yet;
    that is the day to predict in deployment.

    Columns added
    -------------
    target      : what the network learns.
                  target_mode="log_return": ln(Close_{t+1} / Close_t)
                  target_mode="price"     : Close_{t+1}
    next_close  : tomorrow's actual close (used to evaluate in dollars).
    target_date : tomorrow's date (used for plots).
    """
    if target_mode not in ("log_return", "price"):
        raise ValueError(f"Unknown target_mode '{target_mode}'. Use 'log_return' or 'price'.")
    out = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        close = out["Close", ticker]
        out["next_close", ticker] = close.shift(-1)
        out["target_date"] = out.index.to_series().shift(-1)
        out["target", ticker] = np.log(close.shift(-1) / close) if target_mode == "log_return" else close.shift(-1)
    else:
        close = out["Close"]
        out["next_close"] = close.shift(-1)
        out["target_date"] = out.index.to_series().shift(-1)
        out["target"] = np.log(close.shift(-1) / close) if target_mode == "log_return" else close.shift(-1)
    return out


def build_modeling_frame(
    df: pd.DataFrame, ticker: str = config.TICKER, target_mode: str = config.TARGET_MODE
) -> pd.DataFrame:
    """
    Full feature-engineering pipeline for TRAINING/EVALUATION:
    features + target, with incomplete rows removed.

    Removes the first row (no previous close -> no return) and the last row
    (no next day -> no target).
    """
    out = add_target(add_features(df, ticker), ticker, target_mode)
    return out.dropna(subset=ENGINEERED_FEATURES + ["target"])


def build_inference_frame(df: pd.DataFrame, ticker: str = config.TICKER) -> pd.DataFrame:
    """
    Feature pipeline for PREDICTION: same features, but keep the most recent
    row (it has no target yet - that is what we want to predict).
    """
    return add_features(df, ticker).dropna(subset=ENGINEERED_FEATURES)
