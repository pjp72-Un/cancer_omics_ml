"""Shared Matplotlib styling for the project figures.

Use `with plot_theme():` to avoid permanently changing Matplotlib defaults.
"""

import os
from contextlib import AbstractContextManager
from numbers import Integral
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

THEME_COLOURS = {
    "mineral": "#629987",
    "orange": "#EB6834",
    "cactus": "#BCD1CA",
    "sky": "#6A9BCC",
    "plum": "#827DBD",
    "fig": "#C46686",
    "olive": "#788C5D",
    "yellow": "#EDA100",
    "slate": "#5E5D59",
}
PAPER, INK, MUTED, GRID = "#FAF9F5", "#141413", "#5E5D59", "#D1CFC5"
THEME_PALETTES = {
    "classic": list(THEME_COLOURS.values()),
    "blue_warm": [
        "#6A9BCC",
        "#EB6834",
        "#C5D3E0",
        "#827DBD",
        "#629987",
        "#C46686",
        "#788C5D",
        "#EDA100",
        "#5E5D59",
    ],
    "blue_cool": [
        "#6A9BCC",
        "#629987",
        "#827DBD",
        "#C5D3E0",
        "#BCD1CA",
        "#CBCADB",
        "#C46686",
        "#788C5D",
        "#5E5D59",
    ],
}


def theme_palette(n: int = 3, palette: str = "classic") -> list[str]:
    """Return the first n colours of a named palette, in fixed order."""
    if isinstance(n, bool) or not isinstance(n, Integral) or not 1 <= n <= 9:
        raise ValueError("n must be an integer from 1 to 9.")
    if palette not in THEME_PALETTES:
        raise ValueError(f"palette must be one of {list(THEME_PALETTES)}")
    return THEME_PALETTES[palette][:n]


def plot_theme(
    base_size: float = 11,
    base_family: str = "sans-serif",
    background: str = PAPER,
    grid: str = "y",
    palette: str = "classic",
    categorical_axis: str = "none",
) -> AbstractContextManager[None]:
    """Return a temporary rc context. Create and save plots inside the context.

    Line markers and bars get black outlines automatically. For scatter plots,
    use theme_scatter: Matplotlib has no rc setting for scatter edge width.
    """
    if grid not in {"x", "y", "both", "none"}:
        raise ValueError("grid must be 'x', 'y', 'both', or 'none'.")
    if categorical_axis not in {"none", "x", "y", "both"}:
        raise ValueError(
            "categorical_axis must be 'none', 'x', 'y', or 'both'."
        )
    return mpl.rc_context(
        {
            "font.family": base_family,
            "font.size": base_size,
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "text.color": INK,
            "figure.facecolor": background,
            "axes.facecolor": background,
            "axes.edgecolor": INK,
            "axes.labelcolor": "black",
            "axes.labelsize": base_size,
            "axes.labelpad": 12,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=theme_palette(9, palette)),
            "axes.axisbelow": True,
            "axes.grid": grid != "none",
            "axes.grid.axis": "y" if grid == "none" else grid,
            "axes.grid.which": "major",
            "grid.color": GRID,
            "grid.linewidth": 0.85,
            "grid.linestyle": "-",
            "xtick.color": "black"
            if categorical_axis in {"x", "both"}
            else MUTED,
            "ytick.color": "black"
            if categorical_axis in {"y", "both"}
            else MUTED,
            "xtick.labelsize": base_size * 0.93,
            "ytick.labelsize": base_size * 0.93,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "lines.linewidth": 1.81,
            "lines.markersize": 7.6,
            "lines.markeredgecolor": "black",
            "lines.markeredgewidth": 0.65,
            "patch.edgecolor": "black",
            "patch.force_edgecolor": True,
            "patch.linewidth": 1,
            "scatter.edgecolors": "black",
            "legend.frameon": False,
            "legend.fontsize": base_size * 0.93,
            "figure.figsize": (8, 5.5),
            "savefig.dpi": 300,
            "savefig.facecolor": "auto",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def theme_figure(
    title: str,
    subtitle: str = "",
    caption: str = "",
    width: float = 8,
    height: float = 5.5,
) -> tuple[Figure, Axes]:
    """Single-panel layout with space for a top legend and a bottom caption.

    Use within plot_theme(). Wrap long titles/captions explicitly with
    newlines and increase height or adjust subplots for unusually long text.
    """
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.105, right=0.96, bottom=0.20, top=0.75)
    size = mpl.rcParams["font.size"]
    fig.text(0.035, 0.95, title, va="top", weight="bold", fontsize=size * 1.3)
    fig.text(0.035, 0.90, subtitle, va="top", color=MUTED, fontsize=size)
    fig.text(
        0.035, 0.05, caption, va="bottom", color=MUTED, fontsize=size * 0.82
    )
    return fig, ax


def theme_scatter(
    ax: Axes, x: ArrayLike, y: ArrayLike, **kwargs: Any
) -> PathCollection:
    """Filled circles with black outlines; pass c= for group colours."""
    options = dict(s=7.6**2, edgecolors="black", linewidths=0.65, zorder=3)
    options.update(kwargs)
    return ax.scatter(x, y, **options)


def save_figure(
    fig: Figure,
    filename: str | os.PathLike[str],
    width: float | None = None,
    height: float | None = None,
    dpi: int = 300,
    **kwargs: Any,
) -> None:
    """Export PNG, PDF or SVG at a fixed physical size (no tight cropping)."""
    if width is not None or height is not None:
        old_width, old_height = fig.get_size_inches()
        fig.set_size_inches(
            old_width if width is None else width,
            old_height if height is None else height,
        )
    options = dict(dpi=dpi, facecolor=fig.get_facecolor())
    options.update(kwargs)
    fig.savefig(filename, **options)
