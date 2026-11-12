"""
evaluate.py - CRISP-DM phase 5: Evaluation.

Responsibilities
----------------
1. Convert scaled model predictions back into closing prices in dollars.
2. Build simple, non-ML BASELINES. A model is only useful if it beats them.
3. Compute metrics (MAE, RMSE, MAPE, directional accuracy).
4. Compare trained runs in one table (compare_runs saves it to artifacts/metrics.csv).

Baselines Comparison
--------------------
Tomorrow's close is usually very close to today's close. So the trivial
prediction "tomorrow = today" already has a small error.
That is why the model should compare against baselines.

The functions match the Model Evaluation section of the single notebook.
Plots live in src/plotting.py.
"""

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import config
from src.data_loader import load_stock_data
from src.features import build_modeling_frame
from src.preprocessing import PreparedData, prepare_data
from src.train import load_run

# Column in the predictions table -> label shown in comparison tables.
BASELINES = {
    "naive": "Naive baseline",
    "drift": "Random walk + drift",
    "moving_average": f"{config.MA_WINDOW}-day moving average",
}

METRIC_COLUMNS = ["MAE", "RMSE", "MAPE_%", "DirAcc_%"]


def to_price(pred_target, close_t, target_mode: str):
    """
    Convert predictions in target units into next-day prices in dollars.

    log_return mode: Close_{t+1} = Close_t * exp(r_{t+1})
    price mode     : the prediction already is the price.
    """
    return close_t * np.exp(pred_target) if target_mode == "log_return" else pred_target


# Older name, kept for existing code.
target_to_price = to_price


def predict_split(model, data: PreparedData, split: str = "test") -> pd.DataFrame:
    """
    Run the model on one split and return a table indexed by the predicted date:

        close_t        today's close (known when predicting)
        actual         tomorrow's real close
        predicted      model prediction for tomorrow's close
        naive          baseline: tomorrow = today
        drift          baseline: today * exp(average daily log return in TRAIN)
        moving_average baseline: mean of the last MA_WINDOW closes
    """
    s = data.splits[split]
    # Model output is scaled -> undo the scaling -> convert to price.
    pred_scaled = model.predict(s.X, verbose=0).reshape(-1, 1)
    pred_target = data.target_scaler.inverse_transform(pred_scaled).ravel()
    drift = data.frame["log_return"].iloc[: data.boundaries["train"][1]].mean()   # measured on training rows only
    table = pd.DataFrame({
        "close_t": s.close_t,
        "actual": s.next_close,
        "predicted": to_price(pred_target, s.close_t, data.target_mode),
        "naive": s.close_t,
        "drift": s.close_t * np.exp(drift),
        "moving_average": data.frame["Close"].rolling(config.MA_WINDOW).mean().to_numpy()[s.row_idx],
    }, index=s.target_dates)
    table.index.name = "target_date"
    return table


def regression_metrics(actual, predicted, prev_close) -> dict:
    """
    MAE      mean absolute error in dollars: the average miss.
    RMSE     root mean squared error in dollars: punishes big misses more.
    MAPE_%   mean absolute percentage error: error relative to price level.
    DirAcc_% directional accuracy: % of days where the predicted move
             (up/down vs today's close) matches the real move. 50% is a coin flip.
             Undefined (NaN) for predictions that never move, like the naive baseline.
    """
    actual, predicted, prev_close = (np.asarray(x, dtype=float) for x in (actual, predicted, prev_close))
    errors = predicted - actual
    predicted_move, actual_move = np.sign(predicted - prev_close), np.sign(actual - prev_close)
    direction = np.nan if np.all(predicted_move == 0) else float(np.mean(predicted_move == actual_move) * 100)
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAPE_%": float(np.mean(np.abs(errors / actual)) * 100),
        "DirAcc_%": direction,
    }


def baseline_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Metrics table for every baseline column in a predictions table."""
    rows = [{"run": label, "kind": "baseline",
             **regression_metrics(pred_df["actual"], pred_df[column], pred_df["close_t"])}
            for column, label in BASELINES.items()]
    return pd.DataFrame(rows).set_index("run")


def comparison_table(runs: dict, split: str = "test") -> pd.DataFrame:
    """
    One table with every run and every baseline, sorted by RMSE.

    `runs` maps a run name to an object with .model and .data (as returned by src.train.train_run).
    """
    rows, predictions = [], None
    for name, run in runs.items():
        predictions = predict_split(run.model, run.data, split)
        rows.append({"run": name, "kind": "model",
                     **regression_metrics(predictions["actual"], predictions["predicted"], predictions["close_t"])})
    table = pd.concat([pd.DataFrame(rows).set_index("run"), baseline_metrics(predictions)])
    return table.sort_values("RMSE")


def reload_prepared_run(name: str, artifacts_dir: Path = config.ARTIFACTS_DIR, frame: Optional[pd.DataFrame] = None):
    """
    Reload a saved run and rebuild its data with ITS OWN saved scalers and settings,
    just as it would be in deployment. Returns (saved run, PreparedData).
    """
    saved = load_run(name, artifacts_dir)
    meta = saved.metadata
    if frame is None:
        frame = build_modeling_frame(load_stock_data(verbose=False))
    data = prepare_data(frame, meta["feature_columns"], target_mode=meta["target_mode"], lookback=meta["lookback"],
                        feature_scaler=saved.feature_scaler, target_scaler=saved.target_scaler)
    return saved, data


def compare_runs(
    run_names: Optional[Iterable[str]] = None,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    split: str = "test",
    save: bool = True,
) -> pd.DataFrame:
    """
    Evaluate saved runs on the same split, add the baselines, and return one
    table sorted by RMSE. With save=True it also writes each run's predictions to
    artifacts/<run>/predictions_<split>.csv and the table to artifacts/metrics.csv
    (metrics_<split>.csv for other splits).

    Each run is evaluated with its own saved scalers and settings (from metadata.json).
    """
    artifacts_dir = Path(artifacts_dir)
    if run_names is None:
        run_names = [f"{model}_{feature_set}" for model in config.MODEL_NAMES for feature_set in config.FEATURE_SETS]
    run_names = list(run_names)
    if not run_names:
        raise ValueError("No runs to compare.")

    frame = build_modeling_frame(load_stock_data(verbose=False))
    rows = []
    reference = None
    for name in run_names:
        saved, data = reload_prepared_run(name, artifacts_dir, frame)
        meta = saved.metadata
        pred_df = predict_split(saved.model, data, split)

        # Every run must predict the same days with the same actual closes, or the comparison is unfair.
        if reference is None:
            reference = pred_df
        elif not (pred_df.index.equals(reference.index)
                  and np.allclose(pred_df["actual"], reference["actual"], rtol=0, atol=1e-8)):
            raise ValueError(f"Run {name} does not cover the same {split} days as {run_names[0]}.")

        if save:
            pred_df.to_csv(artifacts_dir / name / f"predictions_{split}.csv")
        rows.append({
            "run": name,
            "kind": "model",
            **regression_metrics(pred_df["actual"], pred_df["predicted"], pred_df["close_t"]),
            "model": meta["model_name"],
            "features": "close + volume" if len(meta["feature_columns"]) > 1 else "close",
            "target_mode": meta["target_mode"],
            # Best validation loss: the number to use for MODEL SELECTION.
            # Test metrics only report how the selected model performs.
            "val_loss": meta.get("best_val_loss"),
            "parameters": meta.get("n_parameters"),
            "epochs": meta.get("epochs_run"),
        })

    # Baselines do not depend on the model, only on the days being predicted.
    table = pd.concat([pd.DataFrame(rows).set_index("run"), baseline_metrics(reference)])
    columns = ["kind", "model", "features", "target_mode"] + METRIC_COLUMNS + ["val_loss", "parameters", "epochs"]
    table = table.reindex(columns=columns).sort_values("RMSE")
    # Nullable integers: baselines have no parameter/epoch counts (shown as <NA> instead of 7777.0).
    table[["parameters", "epochs"]] = table[["parameters", "epochs"]].astype("Int64")

    if save:
        path = artifacts_dir / ("metrics.csv" if split == "test" else f"metrics_{split}.csv")
        table.to_csv(path)
    return table
