"""
tuning.py - CRISP-DM next iteration: hyperparameter tuning with KerasTuner (notebook 06).

The search uses training and validation data only; the test set plays no part.
Each trial trains with the same callbacks as the main runs (see src/train.py)
and is scored by its best validation loss.

Saved files:
    artifacts/tuning/<run>/                     KerasTuner trials (one folder per trial)
    artifacts/tuning/<run>/tuning_results.json  every trial's hyperparameters and best val loss
"""

import json
import time
from pathlib import Path
from typing import Optional

import keras_tuner
from tensorflow import keras

import config
from src.models import build_model
from src.preprocessing import PreparedData
from src.train import training_callbacks

# Search space. The CNN also tunes its kernel size; LSTM and GRU ignore it.
UNITS_CHOICES = [32, 64, 128]
KERNEL_SIZE_CHOICES = [3, 5, 7]
DROPOUT_RANGE = (0.0, 0.5, 0.1)          # min, max, step
LEARNING_RATE_RANGE = (1e-4, 3e-3)       # sampled on a log scale
BATCH_SIZE_CHOICES = [16, 32, 64]


class TunedModel(keras_tuner.HyperModel):
    """Search space for the selected architecture. Uses training and validation data only."""

    def __init__(self, model_name: str, data: PreparedData):
        super().__init__()
        self.model_name = model_name
        self.data = data

    def build(self, hp):
        return build_model(
            self.model_name,
            self.data.splits["train"].X.shape[1:],
            units=hp.Choice("units", UNITS_CHOICES),
            kernel_size=hp.Choice("kernel_size", KERNEL_SIZE_CHOICES) if self.model_name == "cnn" else config.KERNEL_SIZE,
            dropout=hp.Float("dropout", *DROPOUT_RANGE[:2], step=DROPOUT_RANGE[2]),
            learning_rate=hp.Float("learning_rate", *LEARNING_RATE_RANGE, sampling="log"),
        )

    def fit(self, hp, model, *args, **kwargs):
        train, val = self.data.splits["train"], self.data.splits["val"]
        # shuffle=False keeps the training windows in chronological order, as in train_model.
        return model.fit(train.X, train.y, validation_data=(val.X, val.y),
                         batch_size=hp.Choice("batch_size", BATCH_SIZE_CHOICES), shuffle=False, *args, **kwargs)


def tuning_results_path(selected_run: str, tuning_dir: Path = config.TUNING_DIR) -> Path:
    return Path(tuning_dir) / selected_run / "tuning_results.json"


def run_tuning(
    selected_run: str,
    model_name: str,
    data: PreparedData,
    retrain: Optional[bool] = None,
    max_trials: int = config.TUNER_TRIALS,
    tuning_dir: Path = config.TUNING_DIR,
) -> list:
    """
    Run the random search (or reuse its saved results when retrain is False) and
    return the trials sorted from best to worst validation loss.
    """
    retrain = config.RETRAIN if retrain is None else retrain
    results_path = tuning_results_path(selected_run, tuning_dir)

    if retrain:
        keras.utils.set_random_seed(config.SEED)
        tuner = keras_tuner.RandomSearch(
            TunedModel(model_name, data),
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=1,
            seed=config.SEED,
            directory=str(tuning_dir),
            project_name=selected_run,
            overwrite=True,
        )
        start = time.time()
        tuner.search(epochs=config.EPOCHS, verbose=0, callbacks=training_callbacks())
        print(f"Tuning finished in {time.time() - start:.0f}s")

        trials = [{**trial.hyperparameters.values, "best val loss": float(trial.score)}
                  for trial in tuner.oracle.get_best_trials(num_trials=max_trials)]
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(trials, indent=2), encoding="utf-8")

    return json.loads(results_path.read_text(encoding="utf-8"))


def best_hparams_from(trials: list) -> dict:
    """The shared settings from config.py, overwritten by the best trial's values."""
    base = {"units": config.UNITS, "kernel_size": config.KERNEL_SIZE, "dropout": config.DROPOUT,
            "learning_rate": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE}
    return {**base, **{key: value for key, value in trials[0].items() if key != "best val loss"}}
