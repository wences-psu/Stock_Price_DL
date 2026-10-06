"""
train.py - CRISP-DM phase 4: Modeling (training + saving artifacts).

Usage from the command line (run from the project root):
    python -m src.train --model lstm                    # price only
    python -m src.train --model lstm --use-volume       # price + volume
    python -m src.train --model all --compare-volume    # every model, with and without volume

What gets saved (one folder per run, e.g. artifacts/lstm_close_volume/):
    model.h5               trained Keras model (weights of the best validation epoch)
    feature_scaler.joblib  scaler fitted on training inputs
    target_scaler.joblib   scaler fitted on training targets
    metadata.json          settings needed to reproduce preprocessing at prediction time
    history.json           loss/metric per epoch (for learning curves)
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

# Allow `python src/train.py` as well as `python -m src.train`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

import config  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import PreparedData, load_prepared_data  # noqa: E402


@dataclass
class TrainingResult:
    """What train_model() returns, so notebooks can inspect everything."""

    model: keras.Model
    history: dict
    data: PreparedData
    artifact_dir: Path
    metadata: dict


def set_seeds(seed: int = config.SEED) -> None:
    """
    Fix every source of randomness we control (Python, NumPy, TensorFlow),
    so re-running an experiment gives (nearly) the same result.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_name(model_name: str, use_volume: bool, target_mode: str = config.TARGET_MODE) -> str:
    """Folder name for a run, e.g. 'lstm_close', 'gru_close_volume', 'lstm_close_price-target'."""
    name = f"{model_name}_{'close_volume' if use_volume else 'close'}"
    if target_mode != "log_return":
        name += "_price-target"
    return name


def train_model(
    model_name: str = config.MODEL_NAME,
    use_volume: bool = config.USE_VOLUME,
    target_mode: str = config.TARGET_MODE,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    patience: int = config.PATIENCE,
    lookback: int = config.LOOKBACK,
    start_date: Optional[str] = config.START_DATE,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    verbose: int = 2,
) -> TrainingResult:
    """
    Full training run: prepare data -> build model -> fit -> save artifacts.
    """
    set_seeds(config.SEED)
    name = run_name(model_name, use_volume, target_mode)
    out_dir = Path(artifacts_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Training run: {name} ===")

    # ---- 1. Data preparation (see src/preprocessing.py) --------------------
    data = load_prepared_data(
        use_volume=use_volume, target_mode=target_mode, lookback=lookback, start_date=start_date
    )
    train, val = data.splits["train"], data.splits["val"]
    print(f"[train] features={data.feature_columns}  X_train={train.X.shape}  X_val={val.X.shape}")

    # ---- 2. Build the network (see src/models.py) --------------------------
    input_shape = train.X.shape[1:]  # (lookback, n_features)
    model = build_model(model_name, input_shape, learning_rate=learning_rate)
    if verbose:
        model.summary()

    # ---- 3. Callbacks: code Keras runs at the end of each epoch ------------
    callbacks = [
        # Stop when validation loss has not improved for `patience` epochs and
        # roll back to the best weights -> prevents overfitting and wasted time.
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        # If validation loss plateaus, halve the learning rate so the optimizer
        # can take finer steps toward a minimum.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(1, patience // 2), min_lr=1e-5, verbose=1
        ),
        # Save the best model seen so far to disk (useful if training is interrupted).
        keras.callbacks.ModelCheckpoint(str(out_dir / "model.h5"), monitor="val_loss", save_best_only=True),
    ]

    # ---- 4. Fit ------------------------------------------------------------
    # shuffle=True shuffles the ORDER OF WINDOWS inside the training set only.
    # That is safe: each window is a self-contained (past -> next day) example,
    # and validation/test data stay strictly later in time. Shuffling makes
    # mini-batches less correlated, which usually helps gradient descent.
    start = time.time()
    history = model.fit(
        train.X,
        train.y,
        validation_data=(val.X, val.y),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    train_seconds = time.time() - start

    # ---- 5. Save artifacts ---------------------------------------------------
    # EarlyStopping restored the best weights, so this saves the best model.
    model.save(out_dir / "model.h5")
    joblib.dump(data.feature_scaler, out_dir / "feature_scaler.joblib")
    joblib.dump(data.target_scaler, out_dir / "target_scaler.joblib")

    # JSON cannot store NumPy floats, so convert every value to a Python float.
    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    best_epoch = int(np.argmin(history_dict["val_loss"])) + 1
    split_ranges = {
        name_: {
            "samples": s.n_samples,
            "first_target_date": s.target_dates.min().date().isoformat(),
            "last_target_date": s.target_dates.max().date().isoformat(),
        }
        for name_, s in data.splits.items()
    }
    metadata = {
        "run_name": name,
        "model_name": model_name,
        "use_volume": use_volume,
        "target_mode": target_mode,
        "feature_columns": data.feature_columns,
        "lookback": lookback,
        "start_date": start_date,
        "train_ratio": config.TRAIN_RATIO,
        "val_ratio": config.VAL_RATIO,
        "splits": split_ranges,
        "epochs_run": len(history_dict["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": float(min(history_dict["val_loss"])),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_parameters": int(model.count_params()),
        "train_seconds": round(train_seconds, 1),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "tensorflow_version": tf.__version__,
        "data_path": str(config.DATA_PATH),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[train] Done in {train_seconds:.0f}s - {metadata['epochs_run']} epochs, "
        f"best epoch {best_epoch}, best val_loss {metadata['best_val_loss']:.4f}. Saved to {out_dir}"
    )
    return TrainingResult(model=model, history=history_dict, data=data, artifact_dir=out_dir, metadata=metadata)

