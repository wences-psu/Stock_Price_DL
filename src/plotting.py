"""
plotting.py - Shared chart style and plotting helpers.

Used by the notebooks (exploration + evaluation), so
every figure in the project looks consistent.

Design:
  * Thin lines and hairline, solid gridlines.
  * A fixed color per role: "actual" is always blue, "predicted" always orange.
  * A legend whenever a chart has many series; single-series charts are named by the title.
  * No dual y-axes: price and volume have different scales, so they get
    separate stacked panels instead of two axes on one plot.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cycler import cycler
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import config

# Chart surface and ink colors.
COLORS = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

# Categorical series colors, always used in this order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua

ROLE_COLORS = {
    "actual": SERIES[0],
    "predicted": SERIES[1],
    "baseline": SERIES[2],
}
SPLIT_COLORS = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}

# Matplotlib line widths are in points: 1.3pt is about 2 screen pixels at 110 dpi.
LINE_WIDTH = 1.3


def set_plot_style() -> None:
    """Apply the project-wide Matplotlib style. Call once at the top of a notebook."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["surface"],
            "axes.facecolor": COLORS["surface"],
            "savefig.facecolor": COLORS["surface"],
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink_secondary"],
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "xtick.labelcolor": COLORS["muted"],
            "ytick.labelcolor": COLORS["muted"],
            "lines.linewidth": LINE_WIDTH,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.labelcolor": COLORS["ink_secondary"],
            "axes.prop_cycle": cycler(color=SERIES),
        }
    )


def _dollar_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))


def save_figure(fig: plt.Figure, filename: str, directory: Path = config.PLOTS_DIR) -> Path:
    """Save a figure as PNG under artifacts/plots/ and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Data understanding (exploration) plots
# ---------------------------------------------------------------------------
def plot_price_and_volume(df: pd.DataFrame, title: str = f"{config.TICKER} close and volume") -> plt.Figure:
    """Two stacked panels sharing the date axis: closing price (top), volume (bottom)."""
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_price.plot(df.index, df["Close"], color=SERIES[0])
    ax_price.set_title(f"{title} - closing price (USD)")
    _dollar_axis(ax_price)

    volume_m = df["Volume"] / 1e6
    ax_vol.fill_between(df.index, volume_m, color=SERIES[0], alpha=0.10, linewidth=0)
    ax_vol.plot(df.index, volume_m, color=SERIES[0], linewidth=0.6)
    ax_vol.set_title("Daily volume (millions of shares)")
    ax_vol.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def plot_return_distribution(returns: pd.Series, title: str = "Distribution of daily log returns") -> plt.Figure:
    """
    Histogram of daily log returns with a normal curve of the same mean/std.
    The normal curve is not a fit, just a reference to show how fat the tails are.
    """
    returns = returns.dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(
        returns, bins=120, density=True, color=SERIES[0], alpha=0.85,
        edgecolor=COLORS["surface"], linewidth=0.4, label="Observed returns",
    )
    x = np.linspace(returns.min(), returns.max(), 400)
    mu, sigma = returns.mean(), returns.std()
    normal_pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(x, normal_pdf, color=SERIES[1], label="Normal distribution (same mean/std)")
    ax.set_title(title)
    ax.set_xlabel("log return")
    ax.set_ylabel("density")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_split_timeline(
    frame: pd.DataFrame, boundaries: Dict[str, Tuple[int, int]], title: str = "Chronological train / validation / test split"
) -> plt.Figure:
    """Closing price colored by the split each day belongs to."""
    labels = {"train": "Train", "val": "Validation", "test": "Test"}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for name, (start, end) in boundaries.items():
        part = frame.iloc[start:end]
        first, last = part.index.min().date(), part.index.max().date()
        ax.plot(part.index, part["Close"], color=SPLIT_COLORS[name], label=f"{labels[name]}: {first} to {last}")
    ax.set_title(title)
    _dollar_axis(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Modeling and evaluation plots
# ---------------------------------------------------------------------------
def plot_history(history: dict, title: str = "Learning curves") -> plt.Figure:
    """
    Training vs validation loss per epoch.
    The dot marks the best validation epoch.
    """
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = int(np.argmin(history["val_loss"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["loss"], color=SERIES[0], label="Training loss")
    ax.plot(epochs, history["val_loss"], color=SERIES[1], label="Validation loss")
    ax.scatter(
        [epochs[best]], [history["val_loss"][best]], s=40, color=SERIES[1],
        edgecolor=COLORS["surface"], linewidth=1.5, zorder=3,
    )
    ax.annotate(
        f"best epoch {epochs[best]}", (epochs[best], history["val_loss"][best]),
        textcoords="offset points", xytext=(6, 8), color=COLORS["ink_secondary"], fontsize=9,
    )
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss (scaled target)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_predictions(
    pred_df: pd.DataFrame,
    title: str = "Actual vs predicted next-day close",
    last_n: Optional[int] = None,
    show_naive: bool = False,
) -> plt.Figure:
    """Actual next-day close vs model prediction."""
    df = pred_df.tail(last_n) if last_n else pred_df
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["actual"], color=ROLE_COLORS["actual"], label="Actual close")
    ax.plot(df.index, df["predicted"], color=ROLE_COLORS["predicted"], label="Model prediction")
    if show_naive:
        ax.plot(df.index, df["naive"], color=ROLE_COLORS["baseline"], label="Naive baseline (today's close)")
    ax.set_title(title)
    _dollar_axis(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_residuals(pred_df: pd.DataFrame, title: str = "Prediction errors (predicted - actual)") -> plt.Figure:
    """Histogram of dollar errors."""
    errors = pred_df["predicted"] - pred_df["actual"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=60, color=SERIES[0], edgecolor=COLORS["surface"], linewidth=0.6)
    ax.axvline(0, color=COLORS["ink_secondary"], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("error (USD)")
    ax.set_ylabel("number of days")
    fig.tight_layout()
    return fig


def plot_metric_comparison(
    table: pd.DataFrame, metric: str = "RMSE", title: Optional[str] = None, higher_is_better: bool = False
) -> plt.Figure:
    """
    Horizontal bars of one metric for every run. Neural networks in blue, baselines in gray.
    `table` is the output of src.evaluate.compare_runs(). The best run is on top.
    """
    t = table.dropna(subset=[metric]).sort_values(metric, ascending=not higher_is_better)
    if title is None:
        title = f"Test-set {metric} ({'higher' if higher_is_better else 'lower'} is better)"
    colors = [COLORS["muted"] if kind == "baseline" else SERIES[0] for kind in t["kind"]]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.4))
    bars = ax.barh(t.index, t[metric], color=colors, height=0.6, edgecolor=COLORS["surface"], linewidth=1.5)
    ax.invert_yaxis()  # best run at the top
    for bar, value in zip(bars, t[metric]):
        ax.annotate(
            f"{value:,.3f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=COLORS["ink_secondary"],
        )
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(
        handles=[Patch(color=SERIES[0], label="Neural network"), Patch(color=COLORS["muted"], label="Baseline")],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    ax.margins(x=0.15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Deployment plot
# ---------------------------------------------------------------------------
def plot_recent_with_prediction(df: pd.DataFrame, prediction: dict, days: int = 120) -> plt.Figure:
    """
    Recent closing prices plus the predicted next close.

    `prediction` is the dict returned by src.predict.predict_next_day().
    """
    recent = df["Close"].tail(days)
    next_date = pd.Timestamp(prediction["next_date"])
    predicted = prediction["predicted_close"]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(recent.index, recent, color=ROLE_COLORS["actual"], label="Actual close")
    ax.plot(
        [recent.index[-1], next_date], [recent.iloc[-1], predicted],
        color=ROLE_COLORS["predicted"], label="Predicted next close",
    )
    ax.scatter(
        [next_date], [predicted], s=40, color=ROLE_COLORS["predicted"],
        edgecolor=COLORS["surface"], linewidth=1.5, zorder=3,
    )
    ax.annotate(
        f"${predicted:,.2f}", (next_date, predicted), xytext=(8, 0), textcoords="offset points",
        va="center", fontsize=9, color=COLORS["ink_secondary"],
    )
    ax.set_title(f"Last {days} trading days and the prediction for {prediction['next_date']}")
    _dollar_axis(ax)
    ax.margins(x=0.06)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
