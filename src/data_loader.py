"""
data_loader.py - 

Handles the reading, cleaning, and validation of daily OHLCV stock data.

It is based on the CRISP-DM phase 2: Data Understanding findings.

Responsibilities
----------------
- Read the raw CSV.
- Validate the data (no missing values, positive prices, unique dates...).
- Optionally select a subset of the data from a ticker symbol.
- Return a clean DataFrame indexed by date, sorted from oldest to newest.

"""

from pathlib import Path
from typing import IO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import config

# Columns every downstream step expects to find.
REQUIRED_COLUMNS = ['Close', 'High', 'Low', 'Open', 'Volume']
PRICE_COLUMNS = ['Close', 'High', 'Low', 'Open']


def validate_stock_data(df: pd.DataFrame) -> List[str]:
    """
    Raise a ValueError if the data breaks basic assumptions.

    Returns a list of warnings for issues that are suspicious but not fatal,
    so the caller can decide whether to show them.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    if null_counts.any():
        raise ValueError(f"Found missing values:\n{null_counts[null_counts > 0]}")

    if (df[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("Found non-positive prices; log returns would be undefined.")

    if (df["Volume"] < 0).any():
        raise ValueError("Found negative volume values.")

    if df.index.has_duplicates:
        raise ValueError("Found duplicated dates.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted in ascending order.")

    # Report warnings.
    warnings = []
    bad_high_low = df.index[df["High"] < df["Low"]]
    if len(bad_high_low):
        warnings.append(f"{len(bad_high_low)} rows have High < Low.")
    zero_volume = df.index[df["Volume"] == 0]
    if len(zero_volume):
        dates = ", ".join(str(d.date()) for d in zero_volume[:5])
        warnings.append(f"{len(zero_volume)} rows have zero volume ({dates}).")
    return warnings


def load_stock_data(
    source: Union[str, Path, IO] = config.DATA_PATH,
    ticker: Optional[str] = config.TICKER,
    start_date: Optional[str] = config.START_DATE,
    end_date: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load, clean and validate daily OHLCV stock data.

    Parameters
    ----------
    source     : path to a CSV file.
    ticker     : stock ticker symbol.
    start_date : keep rows on/after this date; None keeps everything.
    end_date   : keep rows on/before this date; None keeps everything.
    verbose    : print a short summary (and any data warnings).

    Returns
    -------
    DataFrame indexed by Date with columns: Close, High, Low, Open, Volume.
    sorted oldest -> newest.
    """
    df = pd.read_csv(
    source,
    header=[0, 1],      # first two rows are headers
    skiprows=[2],       # skip the empty third row
    index_col=0,        # first column is the Date
    parse_dates=True    # convert Date to datetime
)

    if ticker is not None:
        if isinstance(df.columns, pd.MultiIndex):
            if ticker not in df.columns.get_level_values(1):
                raise ValueError(f"Ticker '{ticker}' not found in the data.")
            df = df.xs(ticker, axis=1, level=1)
        else:
            raise ValueError("Ticker filtering requested but the data has no MultiIndex.")

    # sort chronologically.
    df = df.sort_index()

    # Volume as float so later math (log, diff) behaves uniformly.
    df["Volume"] = df["Volume"].astype(float)

    # Validate BEFORE filtering by date so problems anywhere in the file surface.
    warnings = validate_stock_data(df)

    # Keep only the requested period.
    if start_date is not None:
        df = df.loc[pd.Timestamp(start_date):]
    if end_date is not None:
        df = df.loc[: pd.Timestamp(end_date)]

    if df.empty:
        raise ValueError("No rows left after applying the date filter.")

    if verbose:
        for warning in warnings:
            print(f"[data] warning (whole file): {warning}")
        print(
            f"[data] Loaded {len(df)} rows from {df.index.min().date()} "
            f"to {df.index.max().date()}."
        )
    return df
