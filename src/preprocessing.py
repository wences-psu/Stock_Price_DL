"""
preprocessing.py - CRISP-DM phase 3: Data Preparation (split, scale, window).

Preprocessing for time series:

1. CHRONOLOGICAL SPLIT (no shuffling)
   In ordinary ML we shuffle rows before splitting. With time series that
   would let the model train learn from the future. We always split by time:

       |------------- train -------------|--- val ---|--- test ---|
       oldest                                               newest

2. SCALERS FIT ON TRAIN ONLY
   Neural networks train better when inputs are on a similar scale (about
   mean 0, std 1). The scaler learns the mean and std, and it must learn them
   only from training data. Fitting on everything would leak information
   about the future into training.

3. SLIDING WINDOWS
   Recurrent/convolutional networks take sequences shaped
   (samples, timesteps, features). For each day t we build a window of the
   last LOOKBACK days of features and pair it with the target for day t+1:

       X[i] = features[t-LOOKBACK+1 ... t]     y[i] = target[t]  (= value of day t+1)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from src.data_loader import load_stock_data
from src.features import build_modeling_frame, get_feature_columns


@dataclass
class SplitWindows:
    """All arrays for one split (train, val or test), aligned sample by sample."""

    name: str
    X: np.ndarray                    # (samples, lookback, n_features), scaled inputs
    y: np.ndarray                    # (samples,), scaled target
    row_idx: np.ndarray              # position of day t in the modeling frame
    dates: pd.DatetimeIndex          # day t ("today", last day inside the window)
    target_dates: pd.DatetimeIndex   # day t+1 (the day being predicted)
    close_t: np.ndarray              # close of day t (needed to rebuild prices from returns)
    next_close: np.ndarray           # actual close of day t+1 (ground truth in dollars)

    @property
    def n_samples(self) -> int:
        return len(self.y)


@dataclass
class PreparedData:
    """Everything the modeling and evaluation steps need."""

    frame: pd.DataFrame              # modeling frame (features + target) used to build windows
    feature_columns: list
    target_mode: str
    lookback: int
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    splits: Dict[str, SplitWindows] = field(default_factory=dict)
    boundaries: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # row ranges per split

    def summary(self) -> pd.DataFrame:
        """Table with sample counts and date ranges of each split."""
        rows = []
        for name, s in self.splits.items():
            rows.append(
                {
                    "split": name,
                    "samples": s.n_samples,
                    "first_target_date": s.target_dates.min().date(),
                    "last_target_date": s.target_dates.max().date(),
                    "X shape": s.X.shape,
                }
            )
        return pd.DataFrame(rows).set_index("split")


def chronological_split_indices(
    n_rows: int, train_ratio: float = config.TRAIN_RATIO, val_ratio: float = config.VAL_RATIO
) -> Tuple[int, int]:
    """
    Return (train_end, val_end) row positions for a time-ordered split:
        train = rows [0, train_end)
        val   = rows [train_end, val_end)
        test  = rows [val_end, n_rows)
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Ratios must be in (0, 1) and train_ratio + val_ratio must be < 1.")
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return train_end, val_end


def split_frame(
    frame: pd.DataFrame, train_ratio: float = config.TRAIN_RATIO, val_ratio: float = config.VAL_RATIO
) -> Dict[str, pd.DataFrame]:
    """Split a DataFrame by time into train/val/test pieces (handy for exploration and plots)."""
    train_end, val_end = chronological_split_indices(len(frame), train_ratio, val_ratio)
    return {
        "train": frame.iloc[:train_end],
        "val": frame.iloc[train_end:val_end],
        "test": frame.iloc[val_end:],
    }


def fit_scalers(
    frame: pd.DataFrame, feature_columns: list, train_end: int
) -> Tuple[StandardScaler, StandardScaler]:
    """
    Fit one scaler for the inputs and one for the target, using TRAINING rows only.

    Two separate scalers are used because the target must be converted back
    to its original units later (inverse_transform), independent of the inputs.
    """
    train_rows = frame.iloc[:train_end]
    feature_scaler = StandardScaler().fit(train_rows[feature_columns].to_numpy())
    target_scaler = StandardScaler().fit(train_rows[["target"]].to_numpy())
    return feature_scaler, target_scaler


def make_windows(
    features_scaled: np.ndarray, lookback: int, start: int, end: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows for the days t in [start, end).

    A window for day t may reach back before `start` (into the previous split).
    That is fine: those are *past inputs* that would be known on day t in real
    life. What must never cross into the future is the target, and each target
    here belongs to day t+1 of the same split.

    Returns
    -------
    X     : float32 array (samples, lookback, n_features)
    t_idx : the day-t row position for each sample
    """
    first_t = max(start, lookback - 1)  # need `lookback` rows of history
    t_idx = np.arange(first_t, end)
    X = np.stack([features_scaled[t - lookback + 1 : t + 1] for t in t_idx]).astype("float32")
    return X, t_idx


def prepare_data(
    frame: pd.DataFrame,
    feature_columns: list,
    target_mode: str = config.TARGET_MODE,
    lookback: int = config.LOOKBACK,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    feature_scaler: Optional[StandardScaler] = None,
    target_scaler: Optional[StandardScaler] = None,
) -> PreparedData:
    """
    Turn a modeling frame into scaled, windowed train/val/test sets.

    If scalers are passed (e.g. loaded from a saved model), they are reused
    instead of being fit again, which guarantees evaluation uses the exact
    same transformation as training.
    """
    n_rows = len(frame)
    train_end, val_end = chronological_split_indices(n_rows, train_ratio, val_ratio)

    if feature_scaler is None or target_scaler is None:
        feature_scaler, target_scaler = fit_scalers(frame, feature_columns, train_end)

    # Transform ALL rows with the train-fitted scalers.
    features_scaled = feature_scaler.transform(frame[feature_columns].to_numpy())
    target_scaled = target_scaler.transform(frame[["target"]].to_numpy()).ravel()

    boundaries = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n_rows)}
    close = frame["Close"].to_numpy()
    next_close = frame["next_close"].to_numpy()

    splits = {}
    for name, (start, end) in boundaries.items():
        X, t_idx = make_windows(features_scaled, lookback, start, end)
        if len(t_idx) == 0:
            raise ValueError(f"The '{name}' split has no samples; reduce LOOKBACK or use more data.")
        splits[name] = SplitWindows(
            name=name,
            X=X,
            y=target_scaled[t_idx].astype("float32"),
            row_idx=t_idx,
            dates=frame.index[t_idx],
            target_dates=pd.DatetimeIndex(frame["target_date"].iloc[t_idx]),
            close_t=close[t_idx],
            next_close=next_close[t_idx],
        )

    return PreparedData(
        frame=frame,
        feature_columns=list(feature_columns),
        target_mode=target_mode,
        lookback=lookback,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        splits=splits,
        boundaries=boundaries,
    )


def load_prepared_data(
    use_volume: bool = config.USE_VOLUME,
    target_mode: str = config.TARGET_MODE,
    lookback: int = config.LOOKBACK,
    start_date: Optional[str] = config.START_DATE,
    data_path: Union[str, Path] = config.DATA_PATH,
    feature_scaler: Optional[StandardScaler] = None,
    target_scaler: Optional[StandardScaler] = None,
    verbose: bool = True,
) -> PreparedData:
    """
    Convenience pipeline: load CSV -> engineer features/target -> split, scale, window.
    """
    df = load_stock_data(data_path, start_date=start_date, verbose=verbose)
    frame = build_modeling_frame(df, target_mode)
    feature_columns = get_feature_columns(use_volume, target_mode)
    return prepare_data(
        frame,
        feature_columns,
        target_mode=target_mode,
        lookback=lookback,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )
