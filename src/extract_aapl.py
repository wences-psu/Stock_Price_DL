"""
Extract Apple (AAPL) stock data from the original Kaggle dataset.

Source: S&P500 All Assets – Daily Updated Dataset (Kaggle, yash16jr)
Original file: SnP_daily_update.csv

The script looks for the original file in data/ first, then in ~/Downloads,
unless a path is given with --source. The Kaggle file is updated daily, so a
newer download contains more days than the project dataset; --end-date keeps
only the days up to the project's last trading day (2026-09-29 by default).

It writes a separate verification file, data/AAPL_only_verified.csv, and
compares it with the project dataset. It does not overwrite data/AAPL_only.csv.

Usage (from the project root):
    python src/extract_aapl.py
    python src/extract_aapl.py --source path/to/SnP_daily_update.csv
    python src/extract_aapl.py --end-date none      # keep every day in the file
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_NAME = "SnP_daily_update.csv"
DEFAULT_SOURCES = [DATA_DIR / SOURCE_NAME, Path.home() / "Downloads" / SOURCE_NAME]
PROJECT_FILE = DATA_DIR / "AAPL_only.csv"
OUTPUT_FILE = DATA_DIR / "AAPL_only_verified.csv"
PROJECT_END_DATE = "2026-09-29"  # last trading day in data/AAPL_only.csv


def find_source(source=None) -> Path:
    """Return the path of the original Kaggle file."""
    if source is not None:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        return path
    for path in DEFAULT_SOURCES:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{SOURCE_NAME} not found in data/ or ~/Downloads. Download it from Kaggle "
        "(S&P500 All Assets - Daily Updated Dataset) or pass its path with --source."
    )


def extract(source: Path, end_date=PROJECT_END_DATE) -> pd.DataFrame:
    """Read the two-row header and keep AAPL's Close and Volume, up to end_date."""
    df = pd.read_csv(source, header=[0, 1], index_col=0)
    aapl = df[[("Close", "AAPL"), ("Volume", "AAPL")]]
    aapl.index.name = None  # same layout as data/AAPL_only.csv: two header rows, then the data
    if end_date is not None:
        # The dates are ISO strings (YYYY-MM-DD), so string comparison is chronological.
        aapl = aapl[aapl.index <= end_date]
    return aapl


def compare_with_project(output: Path, project: Path = PROJECT_FILE) -> None:
    """Compare the recreated file with the project dataset, as loaded by pandas."""
    def load(path):
        return pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)

    recreated, existing = load(output), load(project)
    print(f"Recreated dataset: {len(recreated)} AAPL records "
          f"({recreated.index.min().date()} to {recreated.index.max().date()})")
    print(f"Project dataset:   {len(existing)} AAPL records "
          f"({existing.index.min().date()} to {existing.index.max().date()})")
    print("Data values match:", recreated.equals(existing))
    if recreated.equals(existing):
        return

    # Explain the difference: missing days, or different values on shared days.
    common = recreated.index.intersection(existing.index)
    print(f"Shared days: {len(common)}; only in the project dataset: {len(existing.index.difference(recreated.index))}; "
          f"only in the recreated dataset: {len(recreated.index.difference(existing.index))}")
    close_diff = (recreated.loc[common, ("Close", "AAPL")] - existing.loc[common, ("Close", "AAPL")]).abs().max()
    volume_days = int((recreated.loc[common, ("Volume", "AAPL")] != existing.loc[common, ("Volume", "AAPL")]).sum())
    print(f"On shared days: largest Close difference ${close_diff:.6f}; days with a different Volume: {volume_days}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate the AAPL dataset from the original Kaggle file.")
    parser.add_argument("--source", help=f"path to {SOURCE_NAME} (default: data/, then ~/Downloads)")
    parser.add_argument("--end-date", default=PROJECT_END_DATE,
                        help=f"last date to keep, YYYY-MM-DD, or 'none' for every day (default: {PROJECT_END_DATE})")
    args = parser.parse_args()
    end_date = None if args.end_date.lower() == "none" else args.end_date

    source = find_source(args.source)
    aapl = extract(source, end_date)
    aapl.to_csv(OUTPUT_FILE)

    print("AAPL extraction completed successfully.")
    print("Source file:", source)
    print("Number of records:", len(aapl))
    print("Verification file:", OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix())
    compare_with_project(OUTPUT_FILE)


if __name__ == "__main__":
    main()
