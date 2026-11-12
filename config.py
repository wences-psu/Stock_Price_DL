"""
config.py - Central configuration for the whole project.

The values match the settings cell of notebooks/Team7_Deep_Learning_Final_Project.ipynb,
so the phase notebooks (01-06) and the single notebook produce the same results.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"                             # data directory
DATA_PATH = DATA_DIR / "AAPL_only.csv"                       # raw input data
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"                   # trained models, scalers, metadata
LOG_DIR = ARTIFACTS_DIR / "logs"                             # TensorBoard logs
TUNING_DIR = ARTIFACTS_DIR / "tuning"                        # KerasTuner trials
PLOTS_DIR = ARTIFACTS_DIR / "plots"                          # figures saved by the notebooks
METRICS_PATH = ARTIFACTS_DIR / "metrics.csv"                 # model comparison table

TICKER = "AAPL"  # only used for titles/labels

# ---------------------------------------------------------------------------
# Running the notebooks
# ---------------------------------------------------------------------------
# RETRAIN = True  -> notebooks 03 and 06 train every model from scratch.
# RETRAIN = False -> they reload the models saved in artifacts/ by an earlier complete run.
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
# Input columns of each feature set. The run name is "<model>_<feature set>",
# for example "gru_close_volume".
FEATURE_SETS = {"close": ["Close"], "close_volume": ["Close", "Volume"]}

# Inputs used by the log-return experiment (notebook 06).
LOG_FEATURE_SETS = {"close": ["log_return"], "close_volume": ["log_return", "log_volume", "volume_change"]}

# What the network learns to predict:
#   "price":      tomorrow's closing price (scaled). Used by the six main models.
#   "log_return": tomorrow's log return, ln(Close_{t+1} / Close_t). The price is
#                 rebuilt afterwards as Close_t * exp(predicted_return). Used in notebook 06.
TARGET_COLUMNS = {"price": "target_price", "log_return": "target_log_return"}
TARGET_MODE = "price"

# Kept for older code: USE_VOLUME=True selects the "close_volume" feature set.
USE_VOLUME = False

# Window size for the moving average baseline.
MA_WINDOW = 5

# ---------------------------------------------------------------------------
# Modeling and training
# ---------------------------------------------------------------------------
MODEL_NAMES = ("cnn", "lstm", "gru")  # all architectures used in the project

EPOCHS = 100          # maximum passes over the training data (early stopping usually ends sooner)
PATIENCE = 10         # epochs without validation improvement before early stopping
TUNER_TRIALS = 20     # hyperparameter combinations tried in notebook 06

# Settings shared by all six configurations.
BASE_HPARAMS = {"units": 64, "kernel_size": 5, "dropout": 0.2, "learning_rate": 1e-3, "batch_size": 32}

# Individual names, kept for readability in the notebooks.
BATCH_SIZE = BASE_HPARAMS["batch_size"]        # number of windows used for each weight update
LEARNING_RATE = BASE_HPARAMS["learning_rate"]  # step size for the Adam optimizer
DROPOUT = BASE_HPARAMS["dropout"]              # fraction of units switched off during training

# Random seed: makes weight initialization and dropout repeatable.
# (GPU kernels can still add small non-deterministic differences.)
SEED = 42
