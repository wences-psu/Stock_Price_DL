"""
features.py - CRISP-DM phase 3: Data Preparation (feature engineering).

Responsibilities
----------------
1. Create model inputs (features) from raw prices and volume.
2. Create the target: the value we want to predict for the NEXT trading day.

The model predicts tomorrow's log return:

    Log returns, r_t = ln(Close_t / Close_{t-1}) 

 Then turn it back into a price:

    predicted_close_{t+1} = Close_t * exp(predicted_return_{t+1})

"""

import numpy as np
import pandas as pd
import config

# Extra inputs added when volume is included.
VOLUME_FEATURES = ["log_volume", "volume_change"]

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

    use_volume=False -> ["log_return"]                                  (price only)
    use_volume=True  -> ["log_return", "log_volume", "volume_change"]   (price + volume)
    """
    if use_volume:
        columns += VOLUME_FEATURES
    return columns


def add_target(df: pd.DataFrame, ticker: str = config.TICKER) -> pd.DataFrame:
    """
    Add the prediction target for each row t, which describes day t+1.

    shift(-1) moves every value one row UP, so row t receives tomorrow's value.
    The last row gets NaN because its "tomorrow" is not in the data yet -
    that is exactly the day we will predict in deployment.

    Columns added
    -------------
    target      : what the network learns (next log return).
    next_close  : tomorrow's actual close (used to evaluate in dollars).
    target_date : tomorrow's date (used for plots).
    """
    out = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        out["next_close", ticker] = out["Close", ticker].shift(-1)
        out["target_date"] = out.index.to_series().shift(-1)
        out["target", ticker] = out["log_return", ticker]
    else:
        out["next_close"] = out["Close"].shift(-1)
        out["target_date"] = out.index.to_series().shift(-1)
        out["target"] = out["log_return"]
    return out


def build_modeling_frame(df: pd.DataFrame, ticker: str = config.TICKER) -> pd.DataFrame:
    """
    Full feature-engineering pipeline for TRAINING/EVALUATION:
    features + target, with incomplete rows removed.

    Removes the first row (no previous close -> no return) and the last row
    (no next day -> no target).
    """
    out = add_target(add_features(df), ticker)
    return out.dropna(subset=ENGINEERED_FEATURES + ["target"])


def build_inference_frame(df: pd.DataFrame, ticker: str = config.TICKER) -> pd.DataFrame:
    """
    Feature pipeline for PREDICTION: same features, but keep the most recent
    row (it has no target yet - that is what we want to predict).
    """
    return add_features(df, ticker).dropna(subset=ENGINEERED_FEATURES)
