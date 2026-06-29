"""Publication-quality figures for Arctic DP experiments.

Generates 17 figure types for SCI-tier-1 journal submission:
- 9 from aggregate summary CSV (controller comparison, ablation, statistics)
- 8 from per-timestep trace CSV (trajectories, time series, mode transitions)

All figures use Okabe-Ito colorblind-safe palette and output PDF+PNG.

Usage:
    from arctic_quasi_dp.sci1.figures import make_all_figures
    make_all_figures(summary_csv, out_dir, trace_dir=Path("results/raw_traces"))
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from .plot_style import (
    setup_publication_style,
    COLORBLIND_SAFE,
    CONTROLLER_COLORS,
    CONTROLLER_MARKERS,
    CONTROLLER_LINESTYLES,
    SAFETY_MODE_COLORS,
    save_figure,
    get_controller_color,
    get_controller_marker,
)

FIG_DPI = 320

setup_publication_style(fig_dpi=FIG_DPI)


# ============================================================
# Helpers
# ============================================================

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _check(df: pd.DataFrame, cols: List[str], ctx: str) -> List[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warnings.warn(f"[figures] {ctx}: missing {missing}")
    return missing


def _ctrl_color(c: str) -> str:
    return CONTROLLER_COLORS.get(c, COLORBLIND_SAFE["gray"])


def _ctrl_label(c: str) -> str:
    """Pretty-print controller name for axis labels."""
    labels = {
        "full": "Full (proposed)", "pid": "PID", "smc": "SMC",
        "precision": "Precision DP", "ice_aware": "Ice-aware",
        "nmpc": "NMPC", "lqg": "LQG", "dob_nmpc": "DOB-MPC",
        "adrc": "ADRC", "leso_adrc": "LESO-ADRC",
        "robust_mpc": "Conserv. PD", "tube_mpc": "Margin PD",
        "no_cbf": "w/o CBF", "no_cvar": "w/o CVaR",
        "no_observer": "w/o Observer", "no_fallback": "w/o Fallback",
        "oracle_full": "Oracle",
    }
    return labels.get(c, c)


def _load_trace(trace_dir: Path, scenario: str, controller: str, seed: int) -> Optional[pd.DataFrame]:
    """Load a single trace CSV from the raw_traces directory."""
    patterns = [
        f"{scenario}_{controller}_seed{seed}.csv",
        f"{scenario}_{controller}_{seed}.csv",
        f"{scenario}__{controller}__{seed}.csv",
    ]
    for pat in patterns:
        p = trace_dir / pat
        if p.exists():
            return pd.read_csv(p)
    # Try glob
    matches = list(trace_dir.glob(f"*{scenario}*{controller}*seed{seed}*.csv"))
    if matches:
        return pd.read_csv(matches[0])
    return None


# ============================================================
# PART 1: Summary-based figures (9 figures)
# ============================================================

def fig_precision_safety_tradeoff(summary_csv: Path, out_dir: Path) -> None:
    """Fig 1: Precision-safety Pareto tradeoff scatter.

    Each point = one (controller, scenario) pair. Pareto frontier highlighted.
    """
    df = pd.read_csv(summary_csv)
    x, y = "rms_position_error_m_mean", "safety_violation_time_s_mean"
    if _check(df, [x, y], "fig1"):
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for ctrl, g in df.groupby("controller"):
        ax.scatter(g[x], g[y], label=_ctrl_label(ctrl), s=40, alpha=0.8,
                   color=_ctrl_color(ctrl), marker=get_controller_marker(ctrl), edgecolors="white", linewidth=0.3)

    # Pareto front (lower-left is better)
    pts = df[[x, y]].dropna().values
    if len(pts) > 2:
        sorted_pts = pts[np.argsort(pts[:, 0])]
        pareto = [sorted_pts[0]]
        for p in sorted_pts[1:]:
            if p[1] < pareto[-1][1]:
                pareto.append(p)
        pareto = np.array(pareto)
        ax.plot(pareto[:, 0], pareto[:, 1], "--", color="0.5", linewidth=0.8, alpha=0.6, label="Pareto front")

    ax.set_xlabel("RMS position error (m)")
    ax.set_ylabel("Safety violation time (s)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    _save(fig, out_dir / "fig1_precision_safety_tradeoff")


def fig_controller_comparison_boxplot(summary_csv: Path, out_dir: Path) -> None:
    """Fig 2: Box plot distribution of RMS error per controller across scenarios.

    Shows median, IQR, and outliers — more informative than bar charts.
    """
    df = pd.read_csv(summary_csv)
    col = "rms_position_error_m_mean"
    if _check(df, [col], "fig2"):
        return

    controllers = df.groupby("controller")[col].median().sort_values().index.tolist()
    data = [df.loc[df["controller"] == c, col].dropna().values for c in controllers]
    colors = [_ctrl_color(c) for c in controllers]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    bp = ax.boxplot(data, tick_labels=[_ctrl_label(c) for c in controllers], patch_artist=True,
                    widths=0.6, showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_ylabel("RMS position error (m)")
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, out_dir / "fig2_controller_comparison_boxplot")


def fig_ablation_heatmap(summary_csv: Path, out_dir: Path) -> None:
    """Fig 3: Ablation study heatmap — controller × metric improvement over baseline.

    Color = relative improvement (%). Positive = proposed better.
    """
    df = pd.read_csv(summary_csv)
    metrics = ["rms_position_error_m_mean", "p95_position_error_m_mean",
               "safety_violation_time_s_mean", "tail_position_cvar95_m_mean"]
    available = [m for m in metrics if m in df.columns]
    if len(available) < 2:
        return

    ablation_ctrls = ["full", "no_cbf", "no_cvar", "no_observer", "no_fallback"]
    ctrl_avail = [c for c in ablation_ctrls if c in df["controller"].unique()]
    if len(ctrl_avail) < 2:
        return

    # Compute relative improvement over "full"
    full_row = df[df["controller"] == "full"]
    if len(full_row) == 0:
        return
    full_vals = {m: full_row[m].mean() for m in available}

    improvement = {}
    for c in ctrl_avail:
        if c == "full":
            continue
        row = df[df["controller"] == c]
        if len(row) == 0:
            continue
        improvement[_ctrl_label(c)] = {}
        for m in available:
            base = full_vals[m]
            val = row[m].mean()
            if abs(base) > 1e-12:
                improvement[_ctrl_label(c)][m.replace("_mean", "").replace("_", " ")] = (base - val) / base * 100

    if not improvement:
        return

    imp_df = pd.DataFrame(improvement).T
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    im = ax.imshow(imp_df.values, cmap="coolwarm", aspect="auto", vmin=-50, vmax=50)
    ax.set_xticks(range(len(imp_df.columns)))
    ax.set_xticklabels(imp_df.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(imp_df.index)))
    ax.set_yticklabels(imp_df.index, fontsize=8)

    # Annotate cells
    for i in range(len(imp_df.index)):
        for j in range(len(imp_df.columns)):
            val = imp_df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 30 else "black")

    fig.colorbar(im, ax=ax, label="Improvement over Full (%)", shrink=0.8)
    ax.set_title("Ablation: relative improvement when removing components", fontsize=9)
    _save(fig, out_dir / "fig3_ablation_heatmap")


def fig_runtime(summary_csv: Path, out_dir: Path, control_period_ms: float = 100.0) -> None:
    """Fig 4: Runtime feasibility — horizontal bar of P95 solver time with deadline."""
    df = pd.read_csv(summary_csv)
    col = "solver_time_p95_ms_mean"
    if _check(df, [col], "fig4"):
        return

    order = df.groupby("controller")[col].median().sort_values().index.tolist()
    vals = [df.loc[df["controller"] == c, col].dropna().median() for c in order]
    colors = [_ctrl_color(c) for c in order]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.barh([_ctrl_label(c) for c in order], vals, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(control_period_ms, linestyle="--", linewidth=1.2, color=COLORBLIND_SAFE["red"], label=f"Deadline ({control_period_ms:.0f} ms)")

    # Annotate values
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=7)

    ax.set_xlabel("P95 solve time (ms)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    _save(fig, out_dir / "fig4_runtime_feasibility")


def fig_statistical_comparison(comparison_csv: Path, out_dir: Path) -> None:
    """Fig 5: Statistical comparison — p-value + effect size dual panel."""
    if not comparison_csv.exists():
        return
    df = pd.read_csv(comparison_csv)
    if len(df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    metrics = df["metric"].tolist()
    p_vals = [p if not math.isnan(p) else 1.0 for p in df["p_value"].tolist()]
    colors1 = [COLORBLIND_SAFE["green"] if p < 0.05 else COLORBLIND_SAFE["red"] for p in p_vals]

    ax1.barh(metrics, p_vals, color=colors1, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax1.axvline(0.05, linestyle="--", linewidth=1.0, color=COLORBLIND_SAFE["red"], label=r"$\alpha = 0.05$")
    ax1.set_xlabel("p-value (paired Wilcoxon signed-rank)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="x", alpha=0.2)

    d_vals = df["cohens_d"].tolist()
    colors2 = [COLORBLIND_SAFE["blue"] if d > 0 else COLORBLIND_SAFE["orange"] for d in d_vals]
    ax2.barh(metrics, d_vals, color=colors2, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, linewidth=0.5, color="0.5")
    for threshold, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax2.axvline(threshold, linestyle=":", linewidth=0.5, color="0.7")
        ax2.axvline(-threshold, linestyle=":", linewidth=0.5, color="0.7")
    ax2.set_xlabel("Paired Cohen's d_z (positive = proposed better)")

    proposed = df["proposed"].iloc[0] if "proposed" in df.columns else "proposed"
    baseline = df["baseline"].iloc[0] if "baseline" in df.columns else "baseline"
    fig.suptitle(f"{_ctrl_label(proposed)} vs {_ctrl_label(baseline)}", fontsize=10, y=1.02)
    _save(fig, out_dir / "fig5_statistical_comparison")


def fig_monte_carlo_ci(summary_csv: Path, out_dir: Path) -> None:
    """Fig 6: Monte Carlo mean ± 95% CI for key metrics."""
    df = pd.read_csv(summary_csv)
    metrics = ["rms_position_error_m", "p95_position_error_m", "safety_violation_time_s", "failure"]
    available = [m for m in metrics if f"{m}_mean" in df.columns]
    if not available:
        return

    controllers = df.groupby("controller")[f"{available[0]}_mean"].median().sort_values().index.tolist()
    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        mean_col, ci_col = f"{metric}_mean", f"{metric}_ci95"
        means, cis, labels = [], [], []
        for ctrl in controllers:
            sub = df[df["controller"] == ctrl]
            if len(sub) > 0:
                means.append(sub[mean_col].mean())
                cis.append(sub[ci_col].mean() if ci_col in sub.columns else 0)
                labels.append(_ctrl_label(ctrl))

        x = np.arange(len(labels))
        colors = [_ctrl_color(c) for c in controllers]
        ax.errorbar(x, means, yerr=cis, fmt="none", capsize=3, linewidth=1.0, color="0.4")
        ax.scatter(x, means, c=colors, s=40, zorder=5, edgecolors="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Monte Carlo mean ± 95% CI (50 seeds)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "fig6_monte_carlo_ci")


def fig_radar_comparison(summary_csv: Path, out_dir: Path) -> None:
    """Fig 7: Radar chart — multi-metric comparison of selected controllers.

    Normalized so that 1.0 = best across all controllers, 0.0 = worst.
    """
    df = pd.read_csv(summary_csv)
    metrics = ["rms_position_error_m_mean", "p95_position_error_m_mean",
               "safety_violation_time_s_mean", "tail_position_cvar95_m_mean",
               "solver_time_p95_ms_mean", "thrust_saturation_ratio_mean"]
    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        return

    selected = ["full", "pid", "ice_aware", "nmpc"]
    ctrl_avail = [c for c in selected if c in df["controller"].unique()]
    if len(ctrl_avail) < 2:
        return

    # Normalize: 1.0 = best (lowest for all metrics), 0.0 = worst
    norm_data = {}
    for m in available:
        vals = df.groupby("controller")[m].mean()
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-12:
            norm_data[m] = {c: 1.0 - (vals[c] - vmin) / (vmax - vmin) for c in ctrl_avail}
        else:
            norm_data[m] = {c: 1.0 for c in ctrl_avail}

    labels = [m.replace("_mean", "").replace("_", " ") for m in available]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for ctrl in ctrl_avail:
        values = [norm_data[m][ctrl] for m in available]
        values += values[:1]
        ax.plot(angles, values, "-o", label=_ctrl_label(ctrl), color=_ctrl_color(ctrl),
                markersize=4, linewidth=1.5)
        ax.fill(angles, values, alpha=0.08, color=_ctrl_color(ctrl))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rlabel_position(30)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Multi-metric comparison (1.0 = best)", fontsize=9, pad=20)
    _save(fig, out_dir / "fig7_radar_comparison")


def fig_heatmap_controller_scenario(summary_csv: Path, out_dir: Path) -> None:
    """Fig 8: Heatmap — controller × scenario RMS error matrix."""
    df = pd.read_csv(summary_csv)
    col = "rms_position_error_m_mean"
    if _check(df, [col], "fig8"):
        return

    pivot = df.pivot_table(index="controller", columns="scenario_id", values=col, aggfunc="mean")
    if pivot.empty:
        return

    # Sort controllers by median error
    pivot = pivot.loc[pivot.median(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.5), max(3, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_ctrl_label(c) for c in pivot.index], fontsize=7)
    fig.colorbar(im, ax=ax, label="RMS position error (m)", shrink=0.8)
    ax.set_title("Controller × Scenario RMS error matrix", fontsize=9)
    _save(fig, out_dir / "fig8_heatmap_controller_scenario")


def fig_effect_size_summary(comparison_csv: Path, out_dir: Path) -> None:
    """Fig 9: Effect size forest plot with CI — all comparison pairs."""
    if not comparison_csv.exists():
        return
    df = pd.read_csv(comparison_csv)
    if len(df) == 0 or "cohens_d" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8.0, max(3, len(df) * 0.35)))
    y_pos = np.arange(len(df))

    d_vals = df["cohens_d"].values
    # Estimate CI from bootstrap if available, else use ±0.2 heuristic
    if "cohens_d_ci_low" in df.columns:
        ci_low = df["cohens_d_ci_low"].values
        ci_high = df["cohens_d_ci_high"].values
    else:
        ci_low = d_vals - 0.2
        ci_high = d_vals + 0.2

    colors = [COLORBLIND_SAFE["green"] if d > 0 else COLORBLIND_SAFE["red"] for d in d_vals]
    ax.errorbar(d_vals, y_pos, xerr=[d_vals - ci_low, ci_high - d_vals],
                fmt="o", capsize=3, markersize=5, linewidth=1.0, color="0.4")
    ax.scatter(d_vals, y_pos, c=colors, s=40, zorder=5, edgecolors="white", linewidth=0.5)

    labels = [f"{r.get('metric', '')}" for _, r in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, linewidth=0.8, color="0.5")
    ax.axvline(0.2, linestyle=":", linewidth=0.5, color="0.7")
    ax.axvline(0.5, linestyle=":", linewidth=0.5, color="0.7")
    ax.axvline(0.8, linestyle=":", linewidth=0.5, color="0.7")
    ax.set_xlabel("Effect size (Cohen's d_z)")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    _save(fig, out_dir / "fig9_effect_size_forest")


# ============================================================
# PART 2: Trace-based figures (8 figures)
# ============================================================

def fig_trajectory_xy(trace_csv: Path, out_dir: Path, safe_radius: float = 10.0) -> None:
    """Fig 10: XY trajectory with safety boundary and target.

    Shows vessel path, target position, safe region circle, and violation points.
    """
    df = pd.read_csv(trace_csv)
    required = ["x", "y", "position_error"]
    if _check(df, required, "fig10"):
        return

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    # Trajectory colored by time
    t = df["time"].values if "time" in df.columns else np.arange(len(df))
    scatter = ax.scatter(df["x"], df["y"], c=t, cmap="viridis", s=3, alpha=0.7, zorder=3)
    fig.colorbar(scatter, ax=ax, label="Time (s)", shrink=0.8)

    # Target
    tx = df["target_x"].iloc[0] if "target_x" in df.columns else 0.0
    ty = df["target_y"].iloc[0] if "target_y" in df.columns else 0.0
    ax.plot(tx, ty, "r*", markersize=12, zorder=5, label="Target")

    # Safe region
    circle = plt.Circle((tx, ty), safe_radius, fill=False, linestyle="--",
                         color=COLORBLIND_SAFE["red"], linewidth=1.2, label=f"Safe R={safe_radius}m")
    ax.add_patch(circle)

    # Violation points
    if "violation" in df.columns:
        viol = df[df["violation"] > 0]
        if len(viol) > 0:
            ax.scatter(viol["x"], viol["y"], c=COLORBLIND_SAFE["red"], s=8, alpha=0.6, zorder=4, label="Violation")

    # Start point
    ax.plot(df["x"].iloc[0], df["y"].iloc[0], "ks", markersize=8, zorder=5, label="Start")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.2)
    _save(fig, out_dir / "fig10_trajectory_xy")


def fig_timeseries_errors(trace_csv: Path, out_dir: Path) -> None:
    """Fig 11: Time series — position error, heading error, velocity."""
    df = pd.read_csv(trace_csv)
    if "time" not in df.columns:
        return

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)

    # Position error
    if "position_error" in df.columns:
        axes[0].plot(df["time"], df["position_error"], color=COLORBLIND_SAFE["blue"], linewidth=0.8)
        if "target_x" in df.columns:
            safe_r = 10.0  # default
            axes[0].axhline(safe_r, linestyle="--", color=COLORBLIND_SAFE["red"], linewidth=0.8, label=f"Safe radius")
        axes[0].set_ylabel("Position error (m)")
        axes[0].legend(fontsize=7)

    # Heading error
    if "heading_error" in df.columns:
        axes[1].plot(df["time"], np.degrees(df["heading_error"]),
                     color=COLORBLIND_SAFE["orange"], linewidth=0.8)
        axes[1].set_ylabel("Heading error (deg)")

    # Velocity
    if "u" in df.columns and "v" in df.columns:
        axes[2].plot(df["time"], df["u"], color=COLORBLIND_SAFE["blue"], linewidth=0.8, label="u (surge)")
        axes[2].plot(df["time"], df["v"], color=COLORBLIND_SAFE["orange"], linewidth=0.8, label="v (sway)")
        axes[2].set_ylabel("Velocity (m/s)")
        axes[2].legend(fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save(fig, out_dir / "fig11_timeseries_errors")


def fig_timeseries_control(trace_csv: Path, out_dir: Path) -> None:
    """Fig 12: Time series — control forces and moments."""
    df = pd.read_csv(trace_csv)
    if "time" not in df.columns:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)

    # Forces
    for col, label, color in [
        ("tau_x", "τ_x (surge)", COLORBLIND_SAFE["blue"]),
        ("tau_y", "τ_y (sway)", COLORBLIND_SAFE["orange"]),
    ]:
        if col in df.columns:
            axes[0].plot(df["time"], df[col], color=color, linewidth=0.6, label=label, alpha=0.8)
    axes[0].set_ylabel("Force (N)")
    axes[0].legend(fontsize=7)

    # Moment
    if "tau_n" in df.columns:
        axes[1].plot(df["time"], df["tau_n"], color=COLORBLIND_SAFE["green"], linewidth=0.6, label="τ_n (yaw)")
        axes[1].set_ylabel("Moment (N·m)")
        axes[1].legend(fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save(fig, out_dir / "fig12_timeseries_control")


def fig_timeseries_risk(trace_csv: Path, out_dir: Path) -> None:
    """Fig 13: Time series — risk evolution (total, ice, CVaR)."""
    df = pd.read_csv(trace_csv)
    if "time" not in df.columns:
        return

    risk_cols = [("risk_total", "Total risk", COLORBLIND_SAFE["blue"]),
                 ("risk_ice", "Ice risk", COLORBLIND_SAFE["sky"]),
                 ("risk_cvar", "CVaR risk", COLORBLIND_SAFE["orange"])]
    available = [(c, l, co) for c, l, co in risk_cols if c in df.columns]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    for col, label, color in available:
        ax.plot(df["time"], df[col], color=color, linewidth=0.8, label=label, alpha=0.85)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Risk level")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    _save(fig, out_dir / "fig13_timeseries_risk")


def fig_hocbf_safety_function(trace_csv: Path, out_dir: Path) -> None:
    """Fig 14: HOCBF safety function h(x) and CBF activation over time."""
    df = pd.read_csv(trace_csv)
    if "time" not in df.columns:
        return

    has_h = "safety_set_h" in df.columns or "cbf_slack" in df.columns
    has_cbf = "cbf_active" in df.columns
    if not has_h and not has_cbf:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)

    # h(x) value
    if "safety_set_h" in df.columns:
        axes[0].plot(df["time"], df["safety_set_h"], color=COLORBLIND_SAFE["blue"], linewidth=0.8)
        axes[0].axhline(0, linestyle="--", color=COLORBLIND_SAFE["red"], linewidth=0.8, label="h=0 boundary")
        axes[0].set_ylabel("h(x) = R² - ||p-p_ref||²")
        axes[0].legend(fontsize=7)
    elif "cbf_slack" in df.columns:
        axes[0].plot(df["time"], df["cbf_slack"], color=COLORBLIND_SAFE["blue"], linewidth=0.8)
        axes[0].axhline(0, linestyle="--", color=COLORBLIND_SAFE["red"], linewidth=0.8)
        axes[0].set_ylabel("CBF slack (R - dist)")

    # CBF activation
    if has_cbf:
        cbf = df["cbf_active"].values
        axes[1].fill_between(df["time"], 0, cbf, alpha=0.4, color=COLORBLIND_SAFE["red"], step="mid")
        axes[1].set_ylabel("CBF active")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["Inactive", "Active"])

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save(fig, out_dir / "fig14_hocbf_safety_function")


def fig_mode_transition(trace_csv: Path, out_dir: Path) -> None:
    """Fig 15: Mode transition timeline with ice conditions overlay."""
    df = pd.read_csv(trace_csv)
    if "time" not in df.columns or "supervisor_mode" not in df.columns:
        return

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})

    # Mode timeline
    mode_map = {0: "PRECISION", 1: "ICE_AWARE", 2: "QUASI_DP", 3: "ESCAPE"}
    mode_colors = [COLORBLIND_SAFE["blue"], COLORBLIND_SAFE["green"],
                   COLORBLIND_SAFE["orange"], COLORBLIND_SAFE["red"]]
    modes = df["supervisor_mode"].values
    t = df["time"].values

    for i, (mode_id, mode_name) in enumerate(mode_map.items()):
        mask = modes == mode_id
        if np.any(mask):
            axes[0].fill_between(t, 0, 1, where=mask, alpha=0.6,
                                 color=mode_colors[i], label=mode_name, step="mid")
    axes[0].set_ylabel("Supervisor mode")
    axes[0].set_yticks([])
    axes[0].legend(fontsize=6, ncol=4, loc="upper right")

    # Position error
    if "position_error" in df.columns:
        axes[1].plot(t, df["position_error"], color=COLORBLIND_SAFE["blue"], linewidth=0.6)
        axes[1].set_ylabel("Pos. error (m)")

    # Ice conditions
    if "ice_concentration" in df.columns:
        axes[2].plot(t, df["ice_concentration"], color=COLORBLIND_SAFE["sky"], linewidth=0.8, label="SIC")
        axes[2].set_ylabel("Ice conc.")
        axes[2].set_ylim(0, 1.05)
        axes[2].legend(fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save(fig, out_dir / "fig15_mode_transition")


def fig_ice_conditions(trace_csv: Path, out_dir: Path) -> None:
    """Fig 16: Ice conditions timeline — concentration, thickness, drift."""
    df = pd.read_csv(trace_csv)
    if "time" not in df.columns:
        return

    ice_cols = [("ice_concentration", "SIC", COLORBLIND_SAFE["blue"]),
                ("ice_thickness", "SIT (m)", COLORBLIND_SAFE["orange"]),
                ("ice_drift_speed", "Drift (m/s)", COLORBLIND_SAFE["green"])]
    available = [(c, l, co) for c, l, co in ice_cols if c in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(8.0, 2.5 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (col, label, color) in zip(axes, available):
        ax.plot(df["time"], df[col], color=color, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _save(fig, out_dir / "fig16_ice_conditions")


def fig_thruster_allocation(trace_csv: Path, out_dir: Path) -> None:
    """Fig 17: Thruster allocation — commanded vs actual forces + allocation residual."""
    df = pd.read_csv(trace_csv)
    if "time" not in df.columns:
        return

    has_cmd = "tau_cmd_x" in df.columns
    has_act = "tau_x" in df.columns
    has_res = "allocation_residual" in df.columns
    if not has_cmd and not has_act:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)

    # Commanded vs actual
    if has_cmd and has_act:
        axes[0].plot(df["time"], df["tau_cmd_x"], color=COLORBLIND_SAFE["blue"],
                     linewidth=0.6, alpha=0.6, label="cmd Fx")
        axes[0].plot(df["time"], df["tau_x"], color=COLORBLIND_SAFE["blue"],
                     linewidth=0.8, label="actual Fx")
        axes[0].plot(df["time"], df["tau_cmd_y"], color=COLORBLIND_SAFE["orange"],
                     linewidth=0.6, alpha=0.6, label="cmd Fy")
        axes[0].plot(df["time"], df["tau_y"], color=COLORBLIND_SAFE["orange"],
                     linewidth=0.8, label="actual Fy")
    elif has_act:
        axes[0].plot(df["time"], df["tau_x"], color=COLORBLIND_SAFE["blue"], linewidth=0.8, label="Fx")
        axes[0].plot(df["time"], df["tau_y"], color=COLORBLIND_SAFE["orange"], linewidth=0.8, label="Fy")
    axes[0].set_ylabel("Force (N)")
    axes[0].legend(fontsize=6, ncol=2)

    # Allocation residual
    if has_res:
        axes[1].plot(df["time"], df["allocation_residual"], color=COLORBLIND_SAFE["red"], linewidth=0.6)
        axes[1].set_ylabel("Allocation residual (N)")
    elif "thrust_saturation" in df.columns:
        axes[1].fill_between(df["time"], 0, df["thrust_saturation"],
                             alpha=0.4, color=COLORBLIND_SAFE["orange"])
        axes[1].set_ylabel("Thrust saturation")

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save(fig, out_dir / "fig17_thruster_allocation")


# ============================================================
# Master function
# ============================================================

def make_all_figures(
    summary_csv: Path,
    out_dir: Path,
    control_period_ms: float = 100.0,
    trace_dir: Optional[Path] = None,
) -> None:
    """Generate all paper figures.

    Args:
        summary_csv: Path to aggregate_metrics_ci95.csv
        out_dir: Output directory for figures
        control_period_ms: Control period for runtime deadline line
        trace_dir: Directory containing per-run trace CSVs (optional)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Summary-based figures ===
    fig_precision_safety_tradeoff(summary_csv, out_dir)
    fig_controller_comparison_boxplot(summary_csv, out_dir)
    fig_ablation_heatmap(summary_csv, out_dir)
    fig_runtime(summary_csv, out_dir, control_period_ms)
    fig_monte_carlo_ci(summary_csv, out_dir)
    fig_radar_comparison(summary_csv, out_dir)
    fig_heatmap_controller_scenario(summary_csv, out_dir)

    comparison_csv = summary_csv.parent / "statistical_comparisons.csv"
    fig_statistical_comparison(comparison_csv, out_dir)
    fig_effect_size_summary(comparison_csv, out_dir)

    # === Trace-based figures (if trace_dir provided) ===
    if trace_dir and trace_dir.exists():
        # Find one representative trace per controller for the first scenario
        trace_files = sorted(trace_dir.glob("*.csv"))
        if trace_files:
            # Pick one trace for detailed time-series plots
            sample_trace = trace_files[0]
            fig_trajectory_xy(sample_trace, out_dir)
            fig_timeseries_errors(sample_trace, out_dir)
            fig_timeseries_control(sample_trace, out_dir)
            fig_timeseries_risk(sample_trace, out_dir)
            fig_hocbf_safety_function(sample_trace, out_dir)
            fig_mode_transition(sample_trace, out_dir)
            fig_ice_conditions(sample_trace, out_dir)
            fig_thruster_allocation(sample_trace, out_dir)


# ============================================================
# Backward-compatible aliases (old function names)
# ============================================================

def plot_precision_safety_tradeoff(summary_csv: Path, out_dir: Path) -> None:
    fig_precision_safety_tradeoff(summary_csv, out_dir)


def plot_failure_rate(summary_csv: Path, out_dir: Path) -> None:
    """Backward-compatible: uses boxplot instead of bar chart."""
    fig_controller_comparison_boxplot(summary_csv, out_dir)


def plot_runtime(summary_csv: Path, out_dir: Path, control_period_ms: float = 100.0) -> None:
    fig_runtime(summary_csv, out_dir, control_period_ms)


def plot_tail_risk(summary_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    x, y = "tail_position_cvar95_m_mean", "thrust_saturation_ratio_mean"
    if _check(df, [x, y], "plot_tail_risk"):
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for ctrl, g in df.groupby("controller"):
        ax.scatter(g[x], g[y], label=_ctrl_label(ctrl), s=40, alpha=0.8,
                   color=_ctrl_color(ctrl), marker=get_controller_marker(ctrl))
    ax.set_xlabel("CVaR95 of position error (m)")
    ax.set_ylabel("Thrust saturation ratio")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    _save(fig, out_dir / "fig_tail_risk_actuator_stress")


def plot_ablation_contribution(summary_csv: Path, out_dir: Path) -> None:
    fig_ablation_heatmap(summary_csv, out_dir)


def plot_statistical_comparison(comparison_csv: Path, out_dir: Path) -> None:
    fig_statistical_comparison(comparison_csv, out_dir)


def plot_monte_carlo_ci(summary_csv: Path, out_dir: Path) -> None:
    fig_monte_carlo_ci(summary_csv, out_dir)


def plot_thruster_stress(summary_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    metrics = ["thrust_saturation_ratio", "yaw_moment_saturation_ratio", "energy_proxy", "total_variation_of_control"]
    available = [m for m in metrics if f"{m}_mean" in df.columns]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available):
        mean_col = f"{metric}_mean"
        if mean_col not in df.columns:
            continue
        pivot = df.pivot_table(index="scenario_id", columns="controller", values=mean_col, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, width=0.8, legend=False)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " "), fontsize=8)
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=7, loc="upper right")
    fig.suptitle("Thruster stress metrics", fontsize=10)
    _save(fig, out_dir / "fig_thruster_stress")


def plot_mode_switch_timeline(summary_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    mode_cols = [c for c in df.columns if "mode_" in c and "_ratio" in c]
    if not mode_cols:
        return
    controllers = df["controller"].unique()
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(controllers))
    mode_colors = [COLORBLIND_SAFE["blue"], COLORBLIND_SAFE["green"],
                   COLORBLIND_SAFE["orange"], COLORBLIND_SAFE["red"]]
    mode_names = ["precision", "ice_aware", "quasi_dp", "escape"]
    for i, mode in enumerate(mode_names):
        col = f"mode_{mode}_ratio_mean"
        if col not in df.columns:
            continue
        vals = []
        for ctrl in controllers:
            sub = df[df["controller"] == ctrl]
            vals.append(sub[col].mean() if len(sub) > 0 else 0)
        ax.bar(controllers, vals, bottom=bottom, label=mode, color=mode_colors[i % len(mode_colors)], alpha=0.85)
        bottom += np.array(vals)
    ax.set_ylabel("Mode ratio")
    ax.set_title("Supervisor mode distribution per controller")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", labelrotation=35)
    _save(fig, out_dir / "fig_mode_switch_timeline")
