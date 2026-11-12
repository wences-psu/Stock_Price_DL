"""
tuning.py - CRISP-DM phase 5 follow-up: hyperparameter tuning with KerasTuner.

The search uses training and validation data only; the test set plays no part.
run_tuning() follows the same steps as the single notebook, so both find the
same best combination.

Saved files:
    artifacts/tuning/<run>/                     KerasTuner trials (one folder per trial)
    artifacts/tuning/<run>/tuning_results.json  every trial's hyperparameters and best val loss
    artifacts/logs/tuning/                      TensorBoard logs of the trials
"""

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import keras
import keras_tuner

import config
from src.models import build_model
from src.preprocessing import PreparedData, make_dataset
from src.train import training_callbacks


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
            units=hp.Choice("units", [32, 64, 128]),
            kernel_size=hp.Choice("kernel_size", [3, 5, 7]) if self.model_name == "cnn" else config.BASE_HPARAMS["kernel_size"],
            dropout=hp.Float("dropout", 0.0, 0.5, step=0.1),
            learning_rate=hp.Float("learning_rate", 1e-4, 3e-3, sampling="log"),
        )

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", [16, 32, 64])
        return model.fit(make_dataset(self.data.splits["train"], batch_size),
                         validation_data=make_dataset(self.data.splits["val"], batch_size),
                         shuffle=False, *args, **kwargs)


def tuning_results_path(selected_run: str, tuning_dir: Path = config.TUNING_DIR) -> Path:
    return Path(tuning_dir) / selected_run / "tuning_results.json"


def run_tuning(
    selected_run: str,
    model_name: str,
    data: PreparedData,
    retrain: Optional[bool] = None,
    max_trials: int = config.TUNER_TRIALS,
    tuning_dir: Path = config.TUNING_DIR,
    log_root: Path = config.LOG_DIR,
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
        shutil.rmtree(Path(log_root) / "tuning", ignore_errors=True)
        start = time.time()
        tuner.search(epochs=config.EPOCHS, verbose=0, callbacks=training_callbacks(Path(log_root) / "tuning"))
        print(f"Tuning finished in {time.time() - start:.0f}s")

        trials = [{**trial.hyperparameters.values, "best val loss": float(trial.score)}
                  for trial in tuner.oracle.get_best_trials(num_trials=max_trials)]
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(trials, indent=2), encoding="utf-8")

    return json.loads(results_path.read_text(encoding="utf-8"))


def best_hparams_from(trials: list) -> dict:
    """The base settings, overwritten by the best trial's values."""
    return {**config.BASE_HPARAMS, **{key: value for key, value in trials[0].items() if key != "best val loss"}}
