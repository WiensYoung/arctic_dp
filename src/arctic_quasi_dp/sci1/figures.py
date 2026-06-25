"""生成顶刊风格实验图。

所有图均基于 CSV 结果生成，避免手工复制数据。默认输出 PNG 和 PDF。
修复:
- 增加列名校验，缺失列时给出明确警告
- 新增消融贡献图 (fig_ablation_contribution)
- 新增统计比较图 (fig_statistical_comparison)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FIG_DPI = 320


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _check_columns(df: pd.DataFrame, required: List[str], context: str = "") -> List[str]:
    """检查 DataFrame 是否包含所需列，返回缺失列列表。"""
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.warn(f"[figures] {context}: missing columns {missing}. Available: {list(df.columns)}")
    return missing


def plot_precision_safety_tradeoff(summary_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    x = "rms_position_error_m_mean"
    y = "safety_violation_time_s_mean"
    missing = _check_columns(df, [x, y], "plot_precision_safety_tradeoff")
    if missing:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for controller, g in df.groupby("controller"):
        ax.scatter(g[x], g[y], label=controller, s=48, alpha=0.85)
    ax.set_xlabel("RMS position error (m)")
    ax.set_ylabel("Safety violation time (s)")
    ax.set_title("Precision-safety trade-off across ice DP scenarios")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    _save(fig, out_dir / "fig_precision_safety_tradeoff")


def plot_failure_rate(summary_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    col = "failure_mean"
    if col not in df.columns:
        _check_columns(df, [col], "plot_failure_rate")
        return
    pivot = df.pivot_table(index="scenario_id", columns="controller", values=col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylabel("Failure probability")
    ax.set_title("Loss-of-position / safety failure probability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(axis="x", labelrotation=35)
    _save(fig, out_dir / "fig_failure_probability")


def plot_runtime(summary_csv: Path, out_dir: Path, control_period_ms: float = 100.0) -> None:
    df = pd.read_csv(summary_csv)
    col = "solver_time_p95_ms_mean"
    if col not in df.columns:
        _check_columns(df, [col], "plot_runtime")
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    order = df.groupby("controller")[col].median().sort_values().index.tolist()
    vals = [df.loc[df["controller"] == c, col].dropna().median() for c in order]
    ax.barh(order, vals)
    ax.axvline(control_period_ms, linestyle="--", linewidth=1.0, label="Control period")
    ax.set_xlabel("P95 solve time (ms)")
    ax.set_title("Real-time feasibility of controllers")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, out_dir / "fig_runtime_feasibility")


def plot_tail_risk(summary_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    x = "tail_position_cvar95_m_mean"
    y = "thrust_saturation_ratio_mean"
    missing = _check_columns(df, [x, y], "plot_tail_risk")
    if missing:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for controller, g in df.groupby("controller"):
        ax.scatter(g[x], g[y], label=controller, s=48, alpha=0.85)
    ax.set_xlabel("CVaR95 of position error (m)")
    ax.set_ylabel("Thrust saturation ratio")
    ax.set_title("Tail-risk and actuator-stress comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    _save(fig, out_dir / "fig_tail_risk_actuator_stress")


def plot_ablation_contribution(summary_csv: Path, out_dir: Path) -> None:
    """消融贡献图: 对比 full vs no_cbf/no_cvar/no_observer/no_fallback 在关键指标上的差异。"""
    df = pd.read_csv(summary_csv)
    metric = "rms_position_error_m_mean"
    if metric not in df.columns:
        _check_columns(df, [metric], "plot_ablation_contribution")
        return

    ablation_controllers = ["full", "no_cbf", "no_cvar", "no_observer", "no_fallback"]
    available = [c for c in ablation_controllers if c in df["controller"].unique()]
    if len(available) < 2:
        return

    subset = df[df["controller"].isin(available)]
    pivot = subset.pivot_table(index="scenario_id", columns="controller", values=metric, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_ylabel("RMS position error (m)")
    ax.set_title("Ablation study: component contribution to positioning accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(axis="x", labelrotation=35)
    _save(fig, out_dir / "fig_ablation_contribution")


def plot_statistical_comparison(comparison_csv: Path, out_dir: Path) -> None:
    """统计比较图: 显示各指标的 p 值和效应量。"""
    if not comparison_csv.exists():
        return
    df = pd.read_csv(comparison_csv)
    if len(df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图: p 值 (对数尺度)
    metrics = df["metric"].tolist()
    p_vals = df["p_value"].tolist()
    colors = ["#2ecc71" if p < 0.05 else "#e74c3c" for p in p_vals]
    ax1.barh(metrics, p_vals, color=colors, alpha=0.8)
    ax1.axvline(0.05, linestyle="--", linewidth=1.0, color="red", label="α = 0.05")
    ax1.set_xlabel("p-value")
    ax1.set_title("Statistical significance (Wilcoxon rank-sum)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="x", alpha=0.25)

    # 右图: Cohen's d 效应量
    d_vals = df["cohens_d"].tolist()
    colors2 = ["#3498db" if d > 0 else "#e67e22" for d in d_vals]
    ax2.barh(metrics, d_vals, color=colors2, alpha=0.8)
    ax2.axvline(0, linestyle="-", linewidth=0.5, color="gray")
    ax2.axvline(0.2, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
    ax2.axvline(0.5, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
    ax2.axvline(0.8, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
    ax2.set_xlabel("Cohen's d")
    ax2.set_title("Effect size (positive = proposed better)")
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle(f"Proposed: {df['proposed'].iloc[0]} vs Baseline: {df['baseline'].iloc[0]}", fontsize=10, y=1.02)
    _save(fig, out_dir / "fig_statistical_comparison")


def plot_monte_carlo_ci(summary_csv: Path, out_dir: Path) -> None:
    """Monte Carlo 置信区间图: mean ± CI95 for key metrics per controller."""
    df = pd.read_csv(summary_csv)
    metrics = ["rms_position_error_m", "p95_position_error_m", "safety_violation_time_s", "failure"]
    available = [m for m in metrics if f"{m}_mean" in df.columns]
    if not available:
        return

    controllers = df["controller"].unique()
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        mean_col = f"{metric}_mean"
        ci_col = f"{metric}_ci95"
        means = []
        cis = []
        labels = []
        for ctrl in controllers:
            sub = df[df["controller"] == ctrl]
            if len(sub) > 0:
                m = sub[mean_col].mean()
                c = sub[ci_col].mean() if ci_col in sub.columns else 0
                means.append(m)
                cis.append(c)
                labels.append(ctrl)
        x = np.arange(len(labels))
        ax.errorbar(x, means, yerr=cis, fmt="o", capsize=4, markersize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " "), fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Monte Carlo mean ± 95% CI", fontsize=10)
    _save(fig, out_dir / "fig_monte_carlo_ci")


def plot_thruster_stress(summary_csv: Path, out_dir: Path) -> None:
    """推进器压力图: saturation, yaw saturation, energy, total variation."""
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
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=7, loc="upper right")
    fig.suptitle("Thruster stress metrics", fontsize=10)
    _save(fig, out_dir / "fig_thruster_stress")


def plot_mode_switch_timeline(summary_csv: Path, out_dir: Path) -> None:
    """模式切换图: mode ratio per controller across scenarios."""
    df = pd.read_csv(summary_csv)
    mode_cols = [c for c in df.columns if "mode_" in c and "_ratio" in c]
    if not mode_cols:
        return

    controllers = df["controller"].unique()
    fig, ax = plt.subplots(figsize=(10, 5))

    bottom = np.zeros(len(controllers))
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    mode_names = ["precision", "ice_aware", "quasi_dp", "escape"]

    for i, mode in enumerate(mode_names):
        col = f"mode_{mode}_ratio_mean"
        if col not in df.columns:
            continue
        vals = []
        for ctrl in controllers:
            sub = df[df["controller"] == ctrl]
            vals.append(sub[col].mean() if len(sub) > 0 else 0)
        ax.bar(controllers, vals, bottom=bottom, label=mode, color=colors[i % len(colors)], alpha=0.85)
        bottom += np.array(vals)

    ax.set_ylabel("Mode ratio")
    ax.set_title("Supervisor mode distribution per controller")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=35)
    _save(fig, out_dir / "fig_mode_switch_timeline")


def make_all_figures(summary_csv: Path, out_dir: Path, control_period_ms: float = 100.0) -> None:
    """生成所有论文图表。"""
    plot_precision_safety_tradeoff(summary_csv, out_dir)
    plot_failure_rate(summary_csv, out_dir)
    plot_runtime(summary_csv, out_dir, control_period_ms=control_period_ms)
    plot_tail_risk(summary_csv, out_dir)
    plot_ablation_contribution(summary_csv, out_dir)
    plot_monte_carlo_ci(summary_csv, out_dir)
    plot_thruster_stress(summary_csv, out_dir)
    plot_mode_switch_timeline(summary_csv, out_dir)
    # 统计比较图 (如果存在比较结果)
    comparison_csv = summary_csv.parent / "statistical_comparisons.csv"
    plot_statistical_comparison(comparison_csv, out_dir)
