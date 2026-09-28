"""
config.py - Central configuration for the whole project.

"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "SnP_daily_update.csv"   # raw input data
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"                   # trained models, scalers, metadata
PLOTS_DIR = ARTIFACTS_DIR / "plots"                          # figures saved by the evaluation step
METRICS_PATH = ARTIFACTS_DIR / "metrics.csv"                 # model comparison table

TICKER = "AAPL"  # only used for titles/labels

# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------
# The file starts in 1/4/2010.
# Set to None to use the full history.
START_DATE = "2010-01-04"

# ---------------------------------------------------------------------------
# Time-series windowing and splitting
# ---------------------------------------------------------------------------
# LOOKBACK = how many past trading days the network sees to make ONE prediction.
# 60 trading days is roughly 3 calendar months.
LOOKBACK = 60

# Chronological split ratios (must add up to 1.0).
# Oldest 70% -> train, next 15% -> validation, most recent 15% -> test.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Features and target
# ---------------------------------------------------------------------------
# USE_VOLUME = False -> the model only sees price information.
# USE_VOLUME = True  -> the model also sees trading-volume features.
USE_VOLUME = False

# What the network learns to predict:
#   "log_return": tomorrow's log return, ln(Close_{t+1} / Close_t). The price is
#                 reconstructed afterwards as Close_t * exp(predicted_return).
#                 (Recommended - see docs/02_time_series_concepts.md.)
#   "price":      tomorrow's closing price directly (scaled). Kept so you can
#                 compare and see why this is usually a worse idea.
TARGET_MODE = "log_return"

# Window size for the moving average baseline.
MA_WINDOW = 5

# ---------------------------------------------------------------------------
# Modeling and training
# ---------------------------------------------------------------------------
MODEL_NAME = "lstm"                  # default architecture: "lstm", "gru" or "cnn"
MODEL_NAMES = ("lstm", "gru", "cnn")  # all architectures used in the project

EPOCHS = 100          # maximum passes over the training data (early stopping usually ends sooner)
BATCH_SIZE = 32       # number of windows used for each weight update
LEARNING_RATE = 1e-3  # step size for the Adam optimizer
DROPOUT = 0.2         # fraction of units randomly switched off during training (regularization)
PATIENCE = 10         # epochs without validation improvement before early stopping

# Random seed: makes weight initialization and shuffling repeatable.
# (GPU kernels can still add tiny non-deterministic differences.)
SEED = 42
