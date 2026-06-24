"""投稿级指标与统计汇总。

修复:
- failure 阈值从 25m 降到 15m (与安全区域半径对齐)
- 推力饱和检测支持多种列名格式
- 列名检测增加容错
- 新增统计显著性检验 (Wilcoxon rank-sum, Cohen's d)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import math
import warnings

import numpy as np
import pandas as pd


# ---------- 基础统计工具 ----------

def _safe_quantile(x: pd.Series, q: float) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _cvar(x: pd.Series, alpha: float = 0.95) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    q = np.quantile(arr, alpha)
    tail = arr[arr >= q]
    return float(np.mean(tail)) if tail.size else float(q)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """计算 Cohen's d 效应量。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = math.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _wilcoxon_rank_sum_p(a: np.ndarray, b: np.ndarray) -> float:
    """Wilcoxon rank-sum (Mann-Whitney U) 检验的 p 值。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    # 优先使用 scipy (更准确，处理并列更好)
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except ImportError:
        pass
    # 回退: 手动正态近似 (样本量 >= 8 时足够精确)
    combined = np.concatenate([a, b])
    n = na + nb
    # 使用 scipy 的 rankdata 逻辑 (处理并列)
    order = combined.argsort()
    ranked = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1, dtype=float)
    # 处理并列: 相同值取平均排名
    sorted_vals = combined[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(sorted_vals[j] - sorted_vals[i]) < 1e-12:
            j += 1
        if j > i + 1:
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranked[order[k]] = avg_rank
        i = j
    R1 = float(np.sum(ranked[:na]))
    U1 = R1 - na * (na + 1) / 2.0
    mu_U = na * nb / 2.0
    sigma_U = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    if sigma_U < 1e-12:
        return 1.0
    z = (U1 - mu_U) / sigma_U
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return float(max(0.0, min(1.0, p)))


def _norm_cdf(x: float) -> float:
    """标准正态 CDF 的近似 (Abramowitz & Stegun 26.2.17)。"""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-x * x / 2.0) * (t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))))
    return 1.0 - p if x > 0 else p


# ---------- 列名检测工具 ----------

def _find_column(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    """在 DataFrame 中查找候选列名，返回第一个存在的。"""
    for col in candidates:
        if col in df.columns:
            return col
    return default


def _detect_saturation_columns(df: pd.DataFrame) -> List[str]:
    """检测推力饱和相关列 — 支持多种命名约定。"""
    patterns = [
        lambda c: c.startswith("thruster_") and c.endswith("_actual"),
        lambda c: c.startswith("thrust_") and c.endswith("_actual"),
        lambda c: c.startswith("tau_") and c.endswith("_actual"),
        lambda c: "thrust" in c.lower() and "actual" in c.lower(),
        lambda c: c in ("tau_x", "tau_y", "tau_n"),  # 直接力/力矩列
    ]
    for pat in patterns:
        cols = [c for c in df.columns if pat(c)]
        if cols:
            return cols
    return []


# ---------- 核心指标汇总 ----------

def summarize_run(
    df: pd.DataFrame,
    scenario_id: str,
    controller: str,
    seed: int,
    dt: float,
    safe_region_radius: float = 10.0,
    max_force: float = 1500.0,
) -> Dict[str, float | int | str]:
    """Summarise one run into paper-ready metrics.

    Args:
        df: 仿真输出 DataFrame
        scenario_id: 场景 ID
        controller: 控制器名称
        seed: Monte Carlo seed
        dt: 时间步长
        safe_region_radius: 安全区域半径 (m), 用于 failure 判据
    """
    # 列名容错检测
    pos_col = _find_column(df, ["position_error", "pos_error", "position_error_m"], "position_error")
    heading_col = _find_column(df, ["heading_error", "head_error", "heading_error_rad"], "heading_error")
    viol_col = _find_column(df, ["violation", "boundary_violation", "safety_violation"], None)
    solve_col = _find_column(df, ["solve_time_ms", "solver_time_ms", "solve_time"], None)
    energy_col = _find_column(df, ["energy", "energy_proxy", "cumulative_energy"], None)
    cvar_col = _find_column(df, ["risk_cvar", "cvar_risk", "cvar_proxy"], None)
    alloc_col = _find_column(df, ["allocation_success", "alloc_success"], None)
    solver_ok_col = _find_column(df, ["solver_success", "solver_ok", "feasible"], None)

    pos = pd.to_numeric(df.get(pos_col, pd.Series(dtype=float)), errors="coerce") if pos_col else pd.Series(dtype=float)
    heading = np.abs(pd.to_numeric(df.get(heading_col, pd.Series(dtype=float)), errors="coerce")) if heading_col else pd.Series(dtype=float)
    violation = pd.to_numeric(df.get(viol_col, pd.Series(dtype=float)), errors="coerce").fillna(0) if viol_col else pd.Series(np.zeros(len(df)))
    solve_ms = pd.to_numeric(df.get(solve_col, pd.Series(dtype=float)), errors="coerce") if solve_col else pd.Series(dtype=float)
    energy = pd.to_numeric(df.get(energy_col, pd.Series(dtype=float)), errors="coerce") if energy_col else pd.Series(dtype=float)
    cvar_proxy = pd.to_numeric(df.get(cvar_col, pd.Series(dtype=float)), errors="coerce") if cvar_col else pd.Series(dtype=float)

    # 推力饱和检测 — 支持多种列名格式
    sat_cols = _detect_saturation_columns(df)
    saturation_ratio = 0.0
    if sat_cols:
        vals = df[sat_cols].abs().to_numpy(dtype=float)
        saturation_ratio = float(np.mean(vals > 0.95 * max_force))

    alloc_success = pd.to_numeric(df.get(alloc_col, pd.Series(dtype=float)), errors="coerce").fillna(1) if alloc_col else pd.Series(np.ones(len(df)))
    solver_success = pd.to_numeric(df.get(solver_ok_col, pd.Series(dtype=float)), errors="coerce").fillna(1) if solver_ok_col else pd.Series(np.ones(len(df)))

    # 修复: failure 阈值与安全区域半径对齐 (1.5x), 而非固定的 25m
    failure_pos_threshold = 1.5 * safe_region_radius
    failure = int(
        (pos.max(skipna=True) > failure_pos_threshold)
        or (violation.mean() > 0.20)
        or (solver_success.mean() < 0.90)
    )

    return {
        "scenario_id": scenario_id,
        "controller": controller,
        "seed": int(seed),
        "n_steps": int(len(df)),
        "rms_position_error_m": float(math.sqrt(np.nanmean(np.square(pos)))) if len(pos) else float("nan"),
        "p50_position_error_m": _safe_quantile(pos, 0.50),
        "p95_position_error_m": _safe_quantile(pos, 0.95),
        "p99_position_error_m": _safe_quantile(pos, 0.99),
        "max_position_error_m": float(pos.max(skipna=True)) if len(pos) else float("nan"),
        "rms_heading_error_rad": float(math.sqrt(np.nanmean(np.square(heading)))) if len(heading) else float("nan"),
        "safety_violation_time_s": float(violation.sum() * dt),
        "safety_violation_ratio": float(violation.mean()) if len(violation) else 0.0,
        "thrust_saturation_ratio": saturation_ratio,
        "allocation_failure_ratio": float(1.0 - alloc_success.mean()) if len(alloc_success) else 0.0,
        "energy_proxy": float(energy.iloc[-1]) if len(energy) else float("nan"),
        "solver_time_p95_ms": _safe_quantile(solve_ms, 0.95),
        "solver_time_max_ms": float(solve_ms.max(skipna=True)) if len(solve_ms) else float("nan"),
        "infeasible_rate": float(1.0 - solver_success.mean()) if len(solver_success) else 0.0,
        "failure": failure,
        "cvar_risk_p95": _safe_quantile(cvar_proxy, 0.95),
        "tail_position_cvar95_m": _cvar(pos, 0.95),
    }


def aggregate_summary(run_summaries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level metrics into mean/std/CI table."""
    metric_cols = [
        c for c in run_summaries.columns
        if c not in {"scenario_id", "controller", "seed"} and pd.api.types.is_numeric_dtype(run_summaries[c])
    ]
    rows: List[Dict[str, float | str | int]] = []
    for (scenario, controller), g in run_summaries.groupby(["scenario_id", "controller"], dropna=False):
        row: Dict[str, float | str | int] = {"scenario_id": scenario, "controller": controller, "n_seeds": int(g["seed"].nunique())}
        for m in metric_cols:
            vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                row[f"{m}_ci95"] = float("nan")
            else:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                row[f"{m}_ci95"] = float(1.96 * np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario_id", "controller"]).reset_index(drop=True)


def statistical_comparison(
    run_df: pd.DataFrame,
    metric: str,
    baseline: str,
    proposed: str,
    alpha: float = 0.05,
) -> Dict[str, float | str | bool]:
    """对指定指标进行 baseline vs proposed 的统计显著性检验。

    使用 Wilcoxon rank-sum 检验 + Cohen's d 效应量。
    对所有场景汇总比较。
    """
    base_vals = run_df.loc[run_df["controller"] == baseline, metric].dropna().to_numpy(dtype=float)
    prop_vals = run_df.loc[run_df["controller"] == proposed, metric].dropna().to_numpy(dtype=float)
    if len(base_vals) < 2 or len(prop_vals) < 2:
        return {
            "metric": metric,
            "baseline": baseline,
            "proposed": proposed,
            "p_value": float("nan"),
            "cohens_d": float("nan"),
            "significant": False,
            "n_baseline": len(base_vals),
            "n_proposed": len(prop_vals),
            "baseline_mean": float(np.mean(base_vals)) if len(base_vals) else float("nan"),
            "proposed_mean": float(np.mean(prop_vals)) if len(prop_vals) else float("nan"),
            "improvement_pct": float("nan"),
        }
    p = _wilcoxon_rank_sum_p(prop_vals, base_vals)
    d = _cohens_d(prop_vals, base_vals)
    base_mean = float(np.mean(base_vals))
    prop_mean = float(np.mean(prop_vals))
    improvement = (base_mean - prop_mean) / max(abs(base_mean), 1e-9) * 100.0
    return {
        "metric": metric,
        "baseline": baseline,
        "proposed": proposed,
        "p_value": p,
        "cohens_d": d,
        "significant": p < alpha,
        "n_baseline": len(base_vals),
        "n_proposed": len(prop_vals),
        "baseline_mean": base_mean,
        "proposed_mean": prop_mean,
        "improvement_pct": improvement,
    }


def run_all_comparisons(
    run_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    baseline: str = "pid",
    proposed: str = "full",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """对多个指标运行统计比较，返回汇总 DataFrame。"""
    if metrics is None:
        metrics = [
            "rms_position_error_m",
            "p95_position_error_m",
            "safety_violation_time_s",
            "thrust_saturation_ratio",
            "failure",
            "tail_position_cvar95_m",
        ]
    results = []
    for m in metrics:
        if m in run_df.columns:
            results.append(statistical_comparison(run_df, m, baseline, proposed, alpha))
    return pd.DataFrame(results)


def save_tables(run_rows: Iterable[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_df = pd.DataFrame(list(run_rows))
    run_df.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    agg = aggregate_summary(run_df)
    agg.to_csv(out_dir / "aggregate_metrics_ci95.csv", index=False)
    # 保存统计比较结果 (如果有足够数据)
    if len(run_df) > 0 and "controller" in run_df.columns:
        try:
            comparisons = run_all_comparisons(run_df)
            if len(comparisons) > 0:
                comparisons.to_csv(out_dir / "statistical_comparisons.csv", index=False)
        except (ValueError, KeyError, TypeError, OSError) as e:
            import warnings
            warnings.warn(f"Statistical comparison failed: {e}")
