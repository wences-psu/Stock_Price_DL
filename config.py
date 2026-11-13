"""
config.py - Central configuration for the whole project.

Every notebook and module reads its settings from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"                             # data directory
DATA_PATH = DATA_DIR / "AAPL_only.csv"                       # raw input data
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"                   # trained models, scalers, metadata
PLOTS_DIR = ARTIFACTS_DIR / "plots"                          # figures saved by the evaluation step
TUNING_DIR = ARTIFACTS_DIR / "tuning"                        # KerasTuner trials (notebook 06)
METRICS_PATH = ARTIFACTS_DIR / "metrics.csv"                 # model comparison table

TICKER = "AAPL"  # only used for titles/labels

# ---------------------------------------------------------------------------
# Running the notebooks
# ---------------------------------------------------------------------------
# RETRAIN = True  -> notebooks 03 and 06 train every model (and run the tuning search) from scratch.
# RETRAIN = False -> they reload the models and tuning results saved in artifacts/ by an earlier complete run.
RETRAIN = True

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
#                 Used by the six models (see docs/02_time_series_concepts.md).
#   "price":      tomorrow's closing price directly (scaled). Kept for comparison;
#                 its runs are saved with a "_price-target" suffix.
TARGET_MODE = "log_return"

# What the network sees as inputs:
#   "levels":  the closing price (and the volume), scaled with the training mean
#              and std. Used by the six models. Prices far above the training
#              range are a problem: the scaled inputs keep growing and the
#              network has never seen such values.
#   "returns": stationary inputs: the daily log return (and the log volume and
#              its daily change). Their scale does not drift with the price
#              level. Used by the second iteration in notebook 06; its runs are
#              saved with a "_returns" suffix.
INPUT_MODE = "levels"

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
UNITS = 64            # LSTM/GRU units, or CNN filters
KERNEL_SIZE = 5       # CNN kernel size (days covered by one filter)
PATIENCE = 10         # epochs without validation improvement before early stopping
TUNER_TRIALS = 20     # hyperparameter combinations tried in notebook 06

# Random seed: makes weight initialization and dropout repeatable.
# (GPU kernels can still add tiny non-deterministic differences.)
SEED = 42
