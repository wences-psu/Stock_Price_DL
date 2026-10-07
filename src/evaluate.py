"""
evaluate.py - CRISP-DM phase 5: Evaluation.

Responsibilities
----------------
1. Turn model outputs (scaled log returns) back into dollar prices.
2. Build simple, non-ML BASELINES. A model is only useful if it beats them.
3. Compute metrics (MAE, RMSE, MAPE, directional accuracy).
4. Compare every trained run in one table (saved to artifacts/metrics.csv).

Baselines Comparison
--------------------
Tomorrow's close is usually very close to today's close. So the trivial
prediction "tomorrow = today" already has a small error (roughly 1-2% of the
price). That is why the model should compare against baselines.

Plots live in src/plotting.py.
"""

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import config
from src.preprocessing import PreparedData, load_prepared_data
from src.predict import list_artifact_dirs, load_artifacts

# Column in the predictions table -> label shown in comparison tables.
BASELINES = {
    "naive": "baseline: naive (tomorrow = today)",
    "drift": "baseline: random walk + drift",
    "moving_average": f"baseline: {config.MA_WINDOW}-day moving average",
}

METRIC_COLUMNS = ["MAE", "RMSE", "MAPE_%", "DirAcc_%"]


def target_to_price(pred_target: np.ndarray, close_t: np.ndarray, target_mode: str) -> np.ndarray:
    """
    Convert predictions in target units into next-day prices.

    log_return mode: Close_{t+1} = Close_t * exp(r_{t+1})
    price mode     : the prediction already is the price.
    """
    if target_mode == "log_return":
        return close_t * np.exp(pred_target)
    return pred_target


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
    predicted = target_to_price(pred_target, s.close_t, data.target_mode)

    # Drift = average daily log return measured on the TRAINING period only.
    train_end = data.boundaries["train"][1]
    drift = data.frame["log_return"].iloc[:train_end].mean()
    moving_average = data.frame["Close"].rolling(config.MA_WINDOW).mean().to_numpy()[s.row_idx]

    table = pd.DataFrame(
        {
            "date_t": s.dates,
            "close_t": s.close_t,
            "actual": s.next_close,
            "predicted": predicted,
            "naive": s.close_t,
            "drift": s.close_t * np.exp(drift),
            "moving_average": moving_average,
        },
        index=s.target_dates,
    )
    table.index.name = "target_date"
    return table


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, prev_close: np.ndarray) -> dict:
    """
    MAE      mean absolute error in dollars: the average miss.
    RMSE     root mean squared error in dollars: punishes big misses more.
    MAPE_%   mean absolute percentage error: error relative to price level.
    DirAcc_% directional accuracy: % of days where the predicted move
             (up/down vs today's close) matches the real move. 50% is a coin flip.
             Undefined (NaN) for predictions that never move, like the naive baseline.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    prev_close = np.asarray(prev_close, dtype=float)
    errors = predicted - actual

    predicted_move = np.sign(predicted - prev_close)
    actual_move = np.sign(actual - prev_close)
    if np.all(predicted_move == 0):
        directional_accuracy = np.nan
    else:
        directional_accuracy = float(np.mean(predicted_move == actual_move) * 100)

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE_%": float(np.mean(np.abs(errors / actual)) * 100),
        "DirAcc_%": directional_accuracy,
    }


def baseline_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Metrics table for every baseline column in a predictions table."""
    rows = []
    for column, label in BASELINES.items():
        m = regression_metrics(pred_df["actual"], pred_df[column], pred_df["close_t"])
        m.update({"run": label, "kind": "baseline"})
        rows.append(m)
    return pd.DataFrame(rows).set_index("run")


def evaluate_model(model, data: PreparedData, run_label: str, split: str = "test", include_baselines: bool = True):
    """
    Evaluate one model (e.g. straight from a notebook after training).

    Returns (predictions_table, metrics_table).
    """
    pred_df = predict_split(model, data, split)
    m = regression_metrics(pred_df["actual"], pred_df["predicted"], pred_df["close_t"])
    m.update({"run": run_label, "kind": "model"})
    table = pd.DataFrame([m]).set_index("run")
    if include_baselines:
        table = pd.concat([table, baseline_metrics(pred_df)])
    return pred_df, table[["kind"] + METRIC_COLUMNS]


def compare_runs(
    run_names: Optional[Iterable[str]] = None,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    split: str = "test",
    save: bool = True,
) -> pd.DataFrame:
    """
    Evaluate saved runs on the same split, add the baselines, and return one
    table sorted by RMSE. Also saves each run's predictions to
    artifacts/<run>/predictions_<split>.csv and the table to artifacts/metrics.csv.

    Each run is evaluated with ITS OWN saved scalers and settings (from
    metadata.json), just as it would be in deployment.
    """
    artifacts_dir = Path(artifacts_dir)
    if run_names is None:
        run_dirs = list_artifact_dirs(artifacts_dir)
    else:
        run_dirs = [artifacts_dir / name for name in run_names]
    if not run_dirs:
        raise FileNotFoundError(f"No trained runs found in {artifacts_dir}.")

    rows = []
    baseline_table = None
    for run_dir in run_dirs:
        arts = load_artifacts(run_dir)
        meta = arts["metadata"]
        data = load_prepared_data(
            use_volume=meta["use_volume"],
            target_mode=meta["target_mode"],
            lookback=meta["lookback"],
            start_date=meta["start_date"],
            feature_scaler=arts["feature_scaler"],
            target_scaler=arts["target_scaler"],
            verbose=False,
        )
        pred_df = predict_split(arts["model"], data, split)
        pred_df.to_csv(run_dir / f"predictions_{split}.csv")

        m = regression_metrics(pred_df["actual"], pred_df["predicted"], pred_df["close_t"])
        m.update(
            {
                "run": run_dir.name,
                "kind": "model",
                "model": meta["model_name"],
                "features": "close + volume" if meta["use_volume"] else "close",
                "target_mode": meta["target_mode"],
                # Best validation loss: the number to use for MODEL SELECTION.
                # Test metrics only report how the selected model performs.
                "val_loss": meta.get("best_val_loss"),
                "parameters": meta.get("n_parameters"),
                "epochs": meta.get("epochs_run"),
            }
        )
        rows.append(m)

        # Baselines do not depend on the model, only on the test days.
        # All runs share the same days when they share START_DATE and LOOKBACK.
        if baseline_table is None:
            baseline_table = baseline_metrics(pred_df)

    table = pd.concat([pd.DataFrame(rows).set_index("run"), baseline_table])
    columns = ["kind", "model", "features", "target_mode"] + METRIC_COLUMNS + ["val_loss", "parameters", "epochs"]
    table = table.reindex(columns=columns).sort_values("RMSE")
    # Nullable integers: baselines have no parameter/epoch counts (shown as <NA> instead of 7777.0).
    table[["parameters", "epochs"]] = table[["parameters", "epochs"]].astype("Int64")

    if save:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(artifacts_dir / "metrics.csv")
    return table
