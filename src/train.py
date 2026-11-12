"""
train.py - CRISP-DM phase 4: Modeling (training + saving artifacts).

What gets saved (one folder per run, e.g. artifacts/lstm_close_volume/):
    model.keras            trained Keras model (best validation epoch)
    feature_scaler.joblib  scaler fitted on training inputs
    target_scaler.joblib   scaler fitted on training targets
    metadata.json          settings needed to reproduce preprocessing at prediction time
    history.json           loss/metric per epoch (for learning curves)

TensorBoard logs go to artifacts/logs/<run name>/.

train_run() follows exactly the same steps as the single notebook
(seed, clear session, tf.data pipelines, callbacks), so both produce the same models.
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import joblib
import keras
import numpy as np
import tensorflow as tf

import config
from src.models import build_model
from src.preprocessing import PreparedData, make_dataset

MODEL_FILENAME = "model.keras"


def run_name(model_name: str, feature_set: str, suffix: str = "") -> str:
    """Folder name for a run, e.g. 'lstm_close', 'gru_close_volume', 'gru_close_volume_tuned'."""
    return f"{model_name}_{feature_set}" + (f"_{suffix}" if suffix else "")


def training_callbacks(log_dir: Path, patience: int = config.PATIENCE) -> list:
    """
    Code Keras runs at the end of each epoch.

    EarlyStopping     : stop when validation loss has not improved for `patience` epochs
                        and roll back to the best weights (prevents overfitting).
    ReduceLROnPlateau : halve the learning rate when validation loss plateaus.
    TensorBoard       : write logs for `tensorboard --logdir artifacts/logs`.
    """
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, patience // 2), min_lr=1e-5),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]


def save_run(run_dir: Path, model: keras.Model, data: PreparedData, history: dict, metadata: dict) -> None:
    """Serialization: the model alone isn't enough, so save its scalers and settings beside it."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / MODEL_FILENAME))
    joblib.dump(data.feature_scaler, run_dir / "feature_scaler.joblib")
    joblib.dump(data.target_scaler, run_dir / "target_scaler.joblib")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_run(name: str, artifacts_dir: Path = config.ARTIFACTS_DIR) -> SimpleNamespace:
    """Reload everything needed to use a saved model, without any training."""
    run_dir = Path(artifacts_dir) / name
    return SimpleNamespace(
        model=keras.models.load_model(str(run_dir / MODEL_FILENAME), compile=False),
        feature_scaler=joblib.load(run_dir / "feature_scaler.joblib"),
        target_scaler=joblib.load(run_dir / "target_scaler.joblib"),
        history=json.loads((run_dir / "history.json").read_text(encoding="utf-8")),
        metadata=json.loads((run_dir / "metadata.json").read_text(encoding="utf-8")),
    )


def train_run(
    name: str,
    model_name: str,
    data: PreparedData,
    hparams: dict = config.BASE_HPARAMS,
    retrain: Optional[bool] = None,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    log_root: Path = config.LOG_DIR,
) -> SimpleNamespace:
    """
    Train and save one configuration, or reload it when retrain (default: config.RETRAIN) is False.

    Returns a namespace with run_name, model, history, metadata and data.
    """
    retrain = config.RETRAIN if retrain is None else retrain
    if not retrain:
        saved = load_run(name, artifacts_dir)
        return SimpleNamespace(run_name=name, model=saved.model, history=saved.history,
                               metadata=saved.metadata, data=data)

    # Same seed before every run, so each run is repeatable on its own.
    keras.utils.set_random_seed(config.SEED)
    keras.backend.clear_session()
    log_dir = Path(log_root) / name
    shutil.rmtree(log_dir, ignore_errors=True)

    train, val = data.splits["train"], data.splits["val"]
    model = build_model(model_name, train.X.shape[1:], units=hparams["units"], kernel_size=hparams["kernel_size"],
                        dropout=hparams["dropout"], learning_rate=hparams["learning_rate"])
    start = time.time()
    # shuffle=False and no shuffle step in make_dataset: windows stay in chronological order.
    fit = model.fit(make_dataset(train, hparams["batch_size"]),
                    validation_data=make_dataset(val, hparams["batch_size"]),
                    epochs=config.EPOCHS, callbacks=training_callbacks(log_dir), shuffle=False, verbose=0)
    seconds = time.time() - start

    # JSON cannot store NumPy floats, so convert every value to a Python float.
    history = {key: [float(value) for value in values] for key, values in fit.history.items()}
    metadata = {
        "run_name": name, "model_name": model_name, "target_mode": data.target_mode,
        "feature_columns": data.feature_columns, "lookback": data.lookback, "hparams": hparams,
        "epochs_run": len(history["loss"]), "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "best_val_loss": float(min(history["val_loss"])), "n_parameters": int(model.count_params()),
        "train_seconds": round(seconds, 1), "trained_at": datetime.now().isoformat(timespec="seconds"),
        "tensorflow_version": tf.__version__,
    }
    # EarlyStopping restored the best weights, so this saves the best model.
    save_run(Path(artifacts_dir) / name, model, data, history, metadata)
    print(f"{name:20s} {metadata['epochs_run']:3d} epochs, best epoch {metadata['best_epoch']:3d}, "
          f"val_loss {metadata['best_val_loss']:.4f}, {seconds:.0f}s")
    return SimpleNamespace(run_name=name, model=model, history=history, metadata=metadata, data=data)
