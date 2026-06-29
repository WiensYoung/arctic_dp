"""Publication-quality figure style for Arctic DP experiments.

Provides:
1. Okabe-Ito colorblind-safe palette
2. Unified rcParams for consistent figure styling
3. PDF/SVG/PNG output helpers
4. Marker and line-style differentiation for controllers

Reference:
- Okabe & Ito (2002) "Color Universal Design"
- Wong, B. (2011) "Points of view: Color blindness." Nature Methods
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Okabe-Ito colorblind-safe palette
# ============================================================

COLORBLIND_SAFE: Dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#999999",
}

# Controller-specific colors (consistent across all figures)
CONTROLLER_COLORS: Dict[str, str] = {
    "full": COLORBLIND_SAFE["blue"],
    "pid": COLORBLIND_SAFE["gray"],
    "smc": COLORBLIND_SAFE["purple"],
    "precision": COLORBLIND_SAFE["sky"],
    "ice_aware": COLORBLIND_SAFE["green"],
    "nmpc": COLORBLIND_SAFE["orange"],
    "lqg": COLORBLIND_SAFE["yellow"],
    "dob_nmpc": COLORBLIND_SAFE["red"],
    "adrc": "#882255",
    "leso_adrc": "#44AA99",
    "robust_mpc": "#332288",
    "tube_mpc": "#117733",
    "no_cbf": "#DDCC77",
    "no_cvar": "#CC6677",
    "no_observer": "#88CCEE",
    "no_fallback": "#AA4499",
    "oracle_full": "#000000",
}

# Controller-specific markers
CONTROLLER_MARKERS: Dict[str, str] = {
    "full": "o",
    "pid": "s",
    "smc": "D",
    "precision": "^",
    "ice_aware": "v",
    "nmpc": "P",
    "lqg": "X",
    "dob_nmpc": "*",
    "adrc": "p",
    "leso_adrc": "h",
    "robust_mpc": "8",
    "tube_mpc": "d",
    "no_cbf": "1",
    "no_cvar": "2",
    "no_observer": "3",
    "no_fallback": "4",
    "oracle_full": "+",
}

# Controller-specific line styles
CONTROLLER_LINESTYLES: Dict[str, str] = {
    "full": "-",
    "pid": "--",
    "smc": "-.",
    "precision": ":",
    "ice_aware": "-",
    "nmpc": "--",
    "lqg": "-.",
    "dob_nmpc": ":",
    "adrc": "-",
    "leso_adrc": "--",
    "robust_mpc": "-.",
    "tube_mpc": ":",
    "no_cbf": "-",
    "no_cvar": "--",
    "no_observer": "-.",
    "no_fallback": ":",
    "oracle_full": "-",
}

# Safety mode colors (colorblind-safe)
SAFETY_MODE_COLORS: Dict[str, str] = {
    "NORMAL": COLORBLIND_SAFE["blue"],
    "CAUTION": COLORBLIND_SAFE["orange"],
    "SAFETY_FILTER_ACTIVE": COLORBLIND_SAFE["red"],
    "EMERGENCY_BACKUP": COLORBLIND_SAFE["black"],
}


def setup_publication_style(
    font_family: str = "serif",
    font_size: float = 10,
    fig_dpi: int = 320,
    fig_width: float = 3.5,  # single-column width in inches
    fig_height: float = 2.625,  # 3/4 of width (golden ratio approx)
    line_width: float = 1.5,
    marker_size: float = 6,
    grid_alpha: float = 0.3,
    use_tex: bool = False,
) -> None:
    """Configure matplotlib for publication-quality figures.

    Call this once at the beginning of figure generation.

    Args:
        font_family: "serif" (for LaTeX) or "sans-serif"
        font_size: Base font size in points
        fig_dpi: Resolution for raster output
        fig_width: Default figure width in inches
        fig_height: Default figure height in inches
        line_width: Default line width
        marker_size: Default marker size
        grid_alpha: Grid transparency
        use_tex: Whether to use LaTeX rendering (requires LaTeX installation)
    """
    plt.rcParams.update({
        # Font
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,

        # Figure
        "figure.figsize": (fig_width, fig_height),
        "figure.dpi": fig_dpi,
        "savefig.dpi": fig_dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Lines and markers
        "lines.linewidth": line_width,
        "lines.markersize": marker_size,

        # Axes
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": grid_alpha,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,

    })
    # 注意: 不在全局rcParams中设置backend="Agg"
    # 否则import figures.py会永久破坏交互式绘图
    # 需要非交互式后端时，在脚本入口处单独设置:
    #   matplotlib.use("Agg")

    if use_tex:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        })


def get_controller_color(controller: str) -> str:
    """Get color for a controller."""
    return CONTROLLER_COLORS.get(controller, COLORBLIND_SAFE["gray"])


def get_controller_marker(controller: str) -> str:
    """Get marker for a controller."""
    return CONTROLLER_MARKERS.get(controller, "o")


def get_controller_linestyle(controller: str) -> str:
    """Get line style for a controller."""
    return CONTROLLER_LINESTYLES.get(controller, "-")


def apply_controller_style(
    ax: plt.Axes,
    controller: str,
    label: Optional[str] = None,
    line: bool = True,
    marker: bool = True,
    **kwargs,
) -> None:
    """Apply consistent style to a plot element for a given controller.

    Args:
        ax: Matplotlib axes
        controller: Controller name
        label: Legend label (defaults to controller name)
        line: Whether to show line
        marker: Whether to show markers
        **kwargs: Additional plot kwargs
    """
    color = get_controller_color(controller)
    style = {
        "color": color,
        "label": label or controller,
    }
    if line:
        style["linestyle"] = get_controller_linestyle(controller)
    if marker:
        style["marker"] = get_controller_marker(controller)
        style["markevery"] = max(1, kwargs.pop("markevery", 10))

    style.update(kwargs)
    return style


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    formats: List[str] = ["pdf", "png"],
    dpi: int = 320,
) -> List[Path]:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        output_path: Base output path (without extension)
        formats: List of formats to save
        dpi: Resolution for raster formats

    Returns:
        List of saved file paths
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = []
    for fmt in formats:
        path = output_path.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved.append(path)

    return saved


def create_figure(
    n_rows: int = 1,
    n_cols: int = 1,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a publication-style figure.

    Args:
        n_rows: Number of subplot rows
        n_cols: Number of subplot columns
        fig_width: Figure width in inches
        fig_height: Figure height in inches
        **kwargs: Additional subplot kwargs

    Returns:
        (fig, axes) tuple
    """
    w = fig_width or plt.rcParams["figure.figsize"][0]
    h = fig_height or plt.rcParams["figure.figsize"][1]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w * n_cols, h * n_rows), **kwargs)

    return fig, axes
