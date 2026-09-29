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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cycler import cycler
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