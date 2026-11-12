"""
Extract Apple (AAPL) stock data from the original Kaggle dataset.

Source: S&P500 All Assets – Daily Updated Dataset
Original file: ~/Downloads/SnP_daily_update.csv

This script creates a separate verification file.
It does not overwrite the project's existing AAPL_only.csv.
"""

from pathlib import Path
import pandas as pd

# Identify the original Kaggle dataset
source = Path.home() / "Downloads" / "SnP_daily_update.csv"

# Identify the project data directory
project_root = Path(__file__).resolve().parent.parent
output = project_root / "data" / "AAPL_only_verified.csv"

# Read the original dataset with its two header rows
df = pd.read_csv(source, header=[0, 1], index_col=0)

# Extract Apple's closing price and trading volume
aapl = df[[("Close", "AAPL"), ("Volume", "AAPL")]]

# Save the extracted data without overwriting the existing file
aapl.to_csv(output)

print("AAPL extraction completed successfully.")
print("Number of records:", len(aapl))
print("Verification file:", output)