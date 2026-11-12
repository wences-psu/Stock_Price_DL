"""
train.py - CRISP-DM phase 4: Modeling (training + saving artifacts).

What gets saved (one folder per run, e.g. artifacts/lstm_close_volume/):
    model.keras / model.h5 trained Keras model (best validation epoch); .keras on Keras >= 2.12, else .h5
    feature_scaler.joblib  scaler fitted on training inputs
    target_scaler.joblib   scaler fitted on training targets
    metadata.json          settings needed to reproduce preprocessing at prediction time
    history.json           loss/metric per epoch (for learning curves)
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow import keras

import config
from src.models import build_model
from src.predict import load_artifacts
from src.preprocessing import PreparedData, load_prepared_data


# The native .keras format was introduced in Keras 2.12; older versions use HDF5.
KERAS_VERSION = tuple(int(p) for p in keras.__version__.split(".")[:2])
MODEL_FILENAME = "model.keras" if KERAS_VERSION >= (2, 12) else "model.h5"


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
    so re-running an experiment gives the same result on the same hardware.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    keras.utils.set_random_seed(seed)


def run_name(model_name: str, use_volume: bool, target_mode: str = config.TARGET_MODE, suffix: str = "") -> str:
    """
    Folder name for a run, e.g. 'lstm_close', 'gru_close_volume', 'lstm_close_price-target',
    or 'lstm_close_volume_tuned' with suffix='_tuned'.
    """
    name = f"{model_name}_{'close_volume' if use_volume else 'close'}"
    if target_mode != "log_return":
        name += "_price-target"
    return name + suffix


def training_callbacks(
    checkpoint_path: Optional[Path] = None, patience: int = config.PATIENCE, verbose: int = 0
) -> list:
    """
    Code Keras runs at the end of each epoch (shared by train_model and the tuner).

    EarlyStopping     : stop when validation loss has not improved for `patience` epochs
                        and roll back to the best weights (prevents overfitting).
    ReduceLROnPlateau : halve the learning rate when validation loss plateaus, so the
                        optimizer can take finer steps toward a minimum.
    ModelCheckpoint   : save the best model seen so far (only when checkpoint_path is given).
    """
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(1, patience // 2), min_lr=1e-5, verbose=1 if verbose else 0
        ),
    ]
    if checkpoint_path is not None:
        callbacks.append(keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True))
    return callbacks


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
    units: int = config.UNITS,
    kernel_size: int = config.KERNEL_SIZE,
    dropout: float = config.DROPOUT,
    suffix: str = "",
) -> TrainingResult:
    """
    Full training run: prepare data -> build model -> fit -> save artifacts.

    units, kernel_size, dropout, learning_rate and batch_size default to the
    shared settings in config.py; notebook 06 passes the tuned values and
    suffix="_tuned" so the tuned run gets its own folder.

    Every run starts from a cleared Keras session and the same seed, so its
    result does not depend on which runs were trained before it.
    """
    keras.backend.clear_session()
    set_seeds(config.SEED)
    name = run_name(model_name, use_volume, target_mode, suffix)
    out_dir = Path(artifacts_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\n=== Training run: {name} ===")

    # Data preparation (see src/preprocessing.py)
    data = load_prepared_data(
        use_volume=use_volume, target_mode=target_mode, lookback=lookback, start_date=start_date,
        verbose=bool(verbose),
    )
    train, val = data.splits["train"], data.splits["val"]
    if verbose:
        print(f"[train] features={data.feature_columns}  X_train={train.X.shape}  X_val={val.X.shape}")

    # Build the network (see src/models.py)
    input_shape = train.X.shape[1:]  # (lookback, n_features)
    model = build_model(model_name, input_shape, learning_rate=learning_rate, dropout=dropout,
                        units=units, kernel_size=kernel_size)
    if verbose:
        model.summary()

    # Callbacks: EarlyStopping, ReduceLROnPlateau and ModelCheckpoint (see training_callbacks)
    callbacks = training_callbacks(out_dir / MODEL_FILENAME, patience=patience, verbose=verbose)

    # Fit
    # Keep training windows in chronological order.
    # Validation and test data remain separate and later in time.
    start = time.time()
    history = model.fit(
        train.X,
        train.y,
        validation_data=(val.X, val.y),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )
    train_seconds = time.time() - start

    # Save artifacts
    # EarlyStopping restored the best weights, so this saves the best model.
    model.save(out_dir / MODEL_FILENAME)
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
        "units": units,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "n_parameters": int(model.count_params()),
        "train_seconds": round(train_seconds, 1),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "tensorflow_version": tf.__version__,
        "data_path": Path(config.DATA_PATH).resolve().relative_to(PROJECT_ROOT).as_posix(),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"{name:<24} {metadata['epochs_run']:>3} epochs, best epoch {best_epoch:>3}, "
        f"val_loss {metadata['best_val_loss']:.4f}, {train_seconds:.0f}s"
    )
    return TrainingResult(model=model, history=history_dict, data=data, artifact_dir=out_dir, metadata=metadata)


def load_run(name: str, artifacts_dir: Path = config.ARTIFACTS_DIR) -> TrainingResult:
    """
    Reload a saved run (model, history, metadata) and rebuild its data with
    the run's own saved scalers, so nothing is refitted.
    """
    run_dir = Path(artifacts_dir) / name
    arts = load_artifacts(run_dir)
    meta = arts["metadata"]
    with open(run_dir / "history.json", "r", encoding="utf-8") as f:
        history = json.load(f)
    data = load_prepared_data(
        use_volume=meta["use_volume"],
        target_mode=meta["target_mode"],
        lookback=meta["lookback"],
        start_date=meta["start_date"],
        feature_scaler=arts["feature_scaler"],
        target_scaler=arts["target_scaler"],
        verbose=False,
    )
    return TrainingResult(model=arts["model"], history=history, data=data, artifact_dir=run_dir, metadata=meta)


def train_or_load(
    model_name: str,
    use_volume: bool,
    retrain: Optional[bool] = None,
    target_mode: str = config.TARGET_MODE,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    hparams: Optional[dict] = None,
    suffix: str = "",
) -> TrainingResult:
    """
    Train a run when retrain is True (config.RETRAIN by default); otherwise
    reload the copy saved in artifacts/ by an earlier complete run.

    hparams: optional dict with any of units, kernel_size, dropout,
    learning_rate and batch_size (e.g. the best tuning trial).
    """
    retrain = config.RETRAIN if retrain is None else retrain
    if retrain:
        return train_model(model_name, use_volume, target_mode, artifacts_dir=artifacts_dir, verbose=0,
                           suffix=suffix, **(hparams or {}))
    result = load_run(run_name(model_name, use_volume, target_mode, suffix), artifacts_dir)
    m = result.metadata
    print(f"{m['run_name']:<24} reloaded ({m['epochs_run']} epochs, best epoch {m['best_epoch']}, "
          f"val_loss {m['best_val_loss']:.4f})")
    return result

