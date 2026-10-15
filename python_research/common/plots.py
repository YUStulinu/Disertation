"""Dissertation figures (matplotlib): one consistent, print-friendly style for every chart."""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
BLUES = LinearSegmentedColormap.from_list("blues", ["#f2f6fc", "#2a78d6", "#0d3b75"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titleweight": "bold",
    "axes.titlesize": 12, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def bar_chart(names, series: dict[str, list[float]], title: str, xlabel: str, path: Path,
              fmt: str = "{:.2f}", xlim: tuple[float, float] | None = None) -> Path:
    """Horizontal bars, one group per name; one colour per series, legend when >= 2 series."""
    names = list(names)
    n_series = len(series)
    height = 0.7 / n_series
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(names) * max(n_series, 1.4) + 1.2))
    y = np.arange(len(names))
    for k, (label, values) in enumerate(series.items()):
        pos = y - 0.35 + height * (k + 0.5)
        bars = ax.barh(pos, values, height=height * 0.9, color=SERIES[k], label=label)
        ax.bar_label(bars, [fmt.format(v) for v in values], padding=3, color=INK_2, fontsize=8.5)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(textwrap.fill(title, 60), loc="left", pad=22 if n_series > 1 else 6)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + (hi - lo) * 0.12)  # room for value labels
    if n_series > 1:
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n_series,
                  borderaxespad=0.2, handlelength=1.2)
    return _save(fig, path)


def confusion_heatmap(matrices: dict[str, np.ndarray], labels, path: Path) -> Path:
    """One or more confusion matrices side by side (rows = actual, columns = predicted)."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.6 * len(matrices), 4.4), squeeze=False)
    for ax, (title, cm) in zip(axes[0], matrices.items()):
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)  # colour = row share, text = count
        ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(textwrap.fill(title, 34), loc="left", fontsize=11)
        ax.grid(False)
    return _save(fig, path)
