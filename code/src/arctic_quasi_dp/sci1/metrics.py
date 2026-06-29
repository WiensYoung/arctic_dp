"""投稿级指标与统计汇总。

修复:
- failure 阈值: 2.5 * safe_region_radius
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


def _cvar(x: pd.Series, alpha: float = 0.90) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    q = np.quantile(arr, alpha)
    tail = arr[arr >= q]
    return float(np.mean(tail)) if tail.size else float(q)


def _cohens_d(baseline: np.ndarray, proposed: np.ndarray) -> float:
    """计算 Cohen's d 改善效应量 (pooled std version)。

    Deprecated: retained for backward-compatible tests only.
    Paper-facing comparisons use statistics.paired_cohens_dz() instead.
    """
    na, nb = len(baseline), len(proposed)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = math.sqrt(
        ((na - 1) * np.var(baseline, ddof=1) + (nb - 1) * np.var(proposed, ddof=1)) / (na + nb - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(baseline) - np.mean(proposed)) / pooled_std)


def _wilcoxon_rank_sum_p(a: np.ndarray, b: np.ndarray) -> float:
    """Wilcoxon rank-sum (Mann-Whitney U) 检验的 p 值 (独立样本)。"""
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


def _paired_wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Wilcoxon signed-rank 检验的 p 值 (配对样本)。

    用于已按 (scenario_id, seed) 对齐的配对数据。
    a 和 b 必须等长，且 a[i] 与 b[i] 是同一 (scenario, seed) 的配对观测。
    """
    diff = a - b
    # 去除零差值
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(nonzero, alternative="two-sided")
        return float(res.pvalue)
    except ImportError:
        pass
    # 回退: 符号检验 (较弱但不依赖 scipy)
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    n = n_pos + n_neg
    if n < 2:
        return float("nan")
    # 正态近似
    mu = n / 2.0
    sigma = math.sqrt(n / 4.0)
    if sigma < 1e-12:
        return 1.0
    z = (min(n_pos, n_neg) - mu) / sigma
    p = 2.0 * _norm_cdf(z)
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
    """检测推力饱和相关列 — 支持多种命名约定。

    注意: 仅检测力列 (x/y), 不检测力矩列 (n/mz),
    因为力矩的饱和阈值 (max_moment) 与力不同 (max_force)。
    """
    _moment_cols = {"tau_n_actual", "tau_mz_actual", "mz_actual"}
    patterns = [
        lambda c: c.startswith("thruster_") and c.endswith("_actual"),
        lambda c: c.startswith("thrust_") and c.endswith("_actual"),
        lambda c: c.startswith("tau_") and c.endswith("_actual") and c not in _moment_cols,
        lambda c: "thrust" in c.lower() and "actual" in c.lower(),
        lambda c: c in ("tau_x", "tau_y"),  # 直接力列 (力矩 tau_n 单独统计)
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
    max_force: float = 3000.0,
    max_moment: float = 100000.0,
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

    # 推力饱和检测 — 优先使用仿真记录的实际饱和率
    thrust_sat_col = _find_column(df, ["thrust_saturation"], None)
    if thrust_sat_col:
        saturation_ratio = float(pd.to_numeric(df[thrust_sat_col], errors="coerce").fillna(0).mean())
    else:
        sat_cols = _detect_saturation_columns(df)
        saturation_ratio = 0.0
        if sat_cols:
            vals = df[sat_cols].abs().to_numpy(dtype=float)
            saturation_ratio = float(np.mean(vals > 0.95 * max_force))

    alloc_success = pd.to_numeric(df.get(alloc_col, pd.Series(dtype=float)), errors="coerce").fillna(1) if alloc_col else pd.Series(np.ones(len(df)))
    solver_success = pd.to_numeric(df.get(solver_ok_col, pd.Series(dtype=float)), errors="coerce").fillna(1) if solver_ok_col else pd.Series(np.ones(len(df)))

    # failure 阈值: 超出安全区域 2.5 倍或违规率 > 30% 或求解成功率 < 80%
    failure_pos_threshold = 2.5 * safe_region_radius
    failure = int(
        (pos.max(skipna=True) > failure_pos_threshold)
        or (violation.mean() > 0.30)
        or (solver_success.mean() < 0.80)
    )
    # 连续违规严重度: 超出安全区域的时间比例 (比二元 failure 更有区分度)
    violation_severity = float(violation.mean()) if len(violation) > 0 else 0.0
    # 最大超调比: 最大位置误差 / 安全区域半径
    max_overshoot_ratio = float(pos.max(skipna=True) / safe_region_radius) if safe_region_radius > 0 else 0.0

    # Yaw moment saturation (separate from Fx/Fy)
    tau_n_col = _find_column(df, ["tau_n", "tau_cmd_n", "tau_mz",
                                   "tau_n_actual", "tau_cmd_n_actual", "tau_mz_actual"], None)
    yaw_saturation = 0.0
    if tau_n_col:
        tau_n = np.abs(pd.to_numeric(df[tau_n_col], errors="coerce").dropna().to_numpy(dtype=float))
        if len(tau_n) > 0:
            yaw_saturation = float(np.mean(tau_n > 0.95 * max_moment))

    # Total variation of control
    tau_x_col = _find_column(df, ["tau_x", "tau_cmd_x"], None)
    tau_y_col = _find_column(df, ["tau_y", "tau_cmd_y"], None)
    total_variation = 0.0
    if tau_x_col and tau_y_col:
        tau_x = pd.to_numeric(df[tau_x_col], errors="coerce").fillna(0).to_numpy()
        tau_y = pd.to_numeric(df[tau_y_col], errors="coerce").fillna(0).to_numpy()
        if len(tau_x) > 1:
            dx = np.diff(tau_x)
            dy = np.diff(tau_y)
            total_variation = float(np.sum(np.sqrt(dx**2 + dy**2)))

    # Actuator chattering metrics
    thrust_total_var = 0.0
    azimuth_total_var = float("nan")
    azimuth_metric_available = 0.0
    chattering_index = float("nan")
    azimuth_rate_p95 = float("nan")
    azimuth_rate_violation = 0.0
    thrust_rate_violation = 0.0
    power_cap_active_rate = 0.0
    alloc_residual_p95 = float("nan")

    # Thrust total variation
    thrust_cols = [c for c in df.columns if "actual_thrust" in c]
    if thrust_cols:
        for tc in thrust_cols:
            vals = pd.to_numeric(df[tc], errors="coerce").fillna(0).to_numpy()
            if len(vals) > 1:
                thrust_total_var += float(np.sum(np.abs(np.diff(vals))))

    # Azimuth total variation (with wrap-around)
    from ..utils.math_utils import shortest_angle_diff_deg
    angle_cols = [c for c in df.columns if "actual_angle_deg" in c]
    if angle_cols:
        azimuth_total_var = 0.0
        azimuth_metric_available = 1.0
        for ac in angle_cols:
            vals = pd.to_numeric(df[ac], errors="coerce").fillna(0).to_numpy()
            if len(vals) > 1:
                diffs = [abs(shortest_angle_diff_deg(vals[i+1], vals[i])) for i in range(len(vals)-1)]
                azimuth_total_var += float(np.sum(diffs))

    # Chattering index (normalized).
    n_steps = max(len(df) - 1, 1)
    if np.isfinite(azimuth_total_var):
        chattering_index = (thrust_total_var / n_steps + azimuth_total_var / n_steps)
    elif thrust_total_var > 0:
        # 仅有推力变化数据, 无方位角数据 → 仅报告推力 chattering
        chattering_index = thrust_total_var / n_steps

    # Power cap active rate
    pc_col = _find_column(df, ["power_cap_active"], None)
    if pc_col:
        pc_vals = pd.to_numeric(df[pc_col], errors="coerce").fillna(0).to_numpy()
        power_cap_active_rate = float(np.mean(pc_vals > 0.5))

    # Allocation residual p95
    ar_col = _find_column(df, ["allocation_residual_norm", "allocation_residual"], None)
    if ar_col:
        ar_vals = pd.to_numeric(df[ar_col], errors="coerce").dropna().to_numpy()
        if len(ar_vals) > 0:
            alloc_residual_p95 = float(np.percentile(ar_vals, 95))

    # Safety-filter solver metrics (if present).
    sf_success_rate = float("nan")
    sf_infeasible_rate = float("nan")
    sf_solve_time_p50 = float("nan")
    sf_solve_time_p95 = float("nan")
    sf_solve_time_p99 = float("nan")
    sf_deadline_miss_rate = float("nan")
    sf_slack_mean = float("nan")
    sf_slack_p95 = float("nan")
    sf_slack_p99 = float("nan")
    sf_slack_active_rate = float("nan")
    sf_correction_mean = float("nan")
    sf_correction_p95 = float("nan")
    sf_hocbf_margin_min = float("nan")
    sf_hocbf_margin_p01 = float("nan")
    sf_hocbf_margin_p05 = float("nan")
    sf_hocbf_violation_rate = float("nan")
    sf_slack_integral = float("nan")
    sf_soft_certificate_rate = float("nan")
    sf_robust_margin_term_p95 = float("nan")
    sf_robust_disturbance_bound = float("nan")
    sf_nominal_margin_min = float("nan")
    safety_set_h_min = float("nan")
    sf_solver_backend = "unknown"
    sf_success_col = _find_column(df, ["safety_filter_qp_success"], None)
    if sf_success_col:
        sf_success = pd.to_numeric(df[sf_success_col], errors="coerce").dropna()
        if len(sf_success) > 0:
            sf_success_rate = float(np.mean(sf_success > 0.5))
            sf_infeasible_rate = float(1.0 - sf_success_rate)
    sf_time_col = _find_column(df, ["safety_filter_solve_time_ms"], None)
    if sf_time_col:
        sf_time = pd.to_numeric(df[sf_time_col], errors="coerce").dropna()
        if len(sf_time) > 0:
            sf_solve_time_p50 = _safe_quantile(sf_time, 0.50)
            sf_solve_time_p95 = _safe_quantile(sf_time, 0.95)
            sf_solve_time_p99 = _safe_quantile(sf_time, 0.99)
            # Deadline miss: solve time > 100ms (DP control at 10Hz)
            sf_deadline_miss_rate = float(np.mean(sf_time > 100.0))
    sf_slack_col = _find_column(df, ["safety_filter_slack"], None)
    if sf_slack_col:
        sf_slack = pd.to_numeric(df[sf_slack_col], errors="coerce").dropna()
        if len(sf_slack) > 0:
            sf_slack_mean = float(sf_slack.mean())
            sf_slack_p95 = _safe_quantile(sf_slack, 0.95)
            sf_slack_p99 = _safe_quantile(sf_slack, 0.99)
            sf_slack_active_rate = float(np.mean(sf_slack > 1e-9))
            sf_slack_integral = float(np.sum(sf_slack.to_numpy(dtype=float)) * dt)
    sf_corr_col = _find_column(df, ["safety_filter_correction_norm"], None)
    if sf_corr_col:
        sf_corr = pd.to_numeric(df[sf_corr_col], errors="coerce").dropna()
        if len(sf_corr) > 0:
            sf_correction_mean = float(sf_corr.mean())
            sf_correction_p95 = _safe_quantile(sf_corr, 0.95)
    sf_margin_col = _find_column(df, ["hocbf_constraint_margin", "safety_filter_hocbf_margin"], None)
    if sf_margin_col:
        sf_margin = pd.to_numeric(df[sf_margin_col], errors="coerce").dropna()
        if len(sf_margin) > 0:
            sf_hocbf_margin_min = float(sf_margin.min())
            sf_hocbf_margin_p01 = _safe_quantile(sf_margin, 0.01)
            sf_hocbf_margin_p05 = _safe_quantile(sf_margin, 0.05)
            sf_hocbf_violation_rate = float(np.mean(sf_margin < -1e-7))
    sf_cert_col = _find_column(df, ["hocbf_soft_certificate"], None)
    if sf_cert_col:
        cert = pd.to_numeric(df[sf_cert_col], errors="coerce").dropna()
        if len(cert) > 0:
            sf_soft_certificate_rate = float(np.mean(cert > 0.5))
    sf_robust_col = _find_column(df, ["hocbf_robust_disturbance_margin"], None)
    if sf_robust_col:
        robust_vals = pd.to_numeric(df[sf_robust_col], errors="coerce").dropna()
        if len(robust_vals) > 0:
            sf_robust_margin_term_p95 = _safe_quantile(robust_vals, 0.95)
    sf_bound_col = _find_column(df, ["hocbf_disturbance_accel_bound_mps2"], None)
    if sf_bound_col:
        bound_vals = pd.to_numeric(df[sf_bound_col], errors="coerce").dropna()
        if len(bound_vals) > 0:
            sf_robust_disturbance_bound = float(bound_vals.max())
    sf_nominal_margin_col = _find_column(df, ["hocbf_nominal_constraint_margin"], None)
    if sf_nominal_margin_col:
        nominal_vals = pd.to_numeric(df[sf_nominal_margin_col], errors="coerce").dropna()
        if len(nominal_vals) > 0:
            sf_nominal_margin_min = float(nominal_vals.min())
    h_col = _find_column(df, ["safety_set_h"], None)
    if h_col:
        h_vals = pd.to_numeric(df[h_col], errors="coerce").dropna()
        if len(h_vals) > 0:
            safety_set_h_min = float(h_vals.min())
    sf_backend_col = _find_column(df, ["safety_filter_solver_backend"], None)
    if sf_backend_col:
        backends = df[sf_backend_col].dropna().unique()
        if len(backends) > 0:
            sf_solver_backend = str(backends[0])

    # Solver time mean
    solver_time_mean = float(solve_ms.mean()) if len(solve_ms) > 0 else float("nan")

    # Mode switching metrics
    mode_col = _find_column(df, ["supervisor_mode", "mode"], None)
    mode_switch_count = 0
    min_mode_dwell_time_s = float("nan")
    dwell_time_violation_count = 0
    mode_ratios = {}
    if mode_col:
        modes = pd.to_numeric(df[mode_col], errors="coerce").fillna(0).to_numpy()
        if len(modes) > 1:
            transitions = np.flatnonzero(np.diff(modes) != 0) + 1
            mode_switch_count = int(len(transitions))
            segment_edges = np.concatenate(([0], transitions, [len(modes)]))
            dwell_steps = np.diff(segment_edges)
            if len(dwell_steps) > 0:
                min_mode_dwell_time_s = float(np.min(dwell_steps) * dt)
                # SupervisorParams.dwell_time 默认 5.0s，使用 0.5 倍作为违规阈值
                # (即实际驻留时间 < 2.5s 才算违规，避免因仿真离散化产生误报)
                _dwell_threshold = 2.5  # 0.5 * SupervisorParams.dwell_time default
                dwell_time_violation_count = int(np.sum(dwell_steps * dt < _dwell_threshold - 1e-9))
        total_steps = len(modes)
        if total_steps > 0:
            mode_names = {0: "precision", 1: "ice_aware", 2: "quasi_dp", 3: "escape"}
            for mode_id, mode_name in mode_names.items():
                mode_ratios[f"mode_{mode_name}_ratio"] = float(np.mean(modes == mode_id))

    # CBF margin
    cbf_col = _find_column(df, ["cbf_slack", "cbf_margin"], None)
    min_cbf_margin = float("nan")
    if cbf_col:
        cbf_vals = pd.to_numeric(df[cbf_col], errors="coerce").dropna()
        if len(cbf_vals) > 0:
            min_cbf_margin = float(cbf_vals.min())

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
        "p95_heading_error_rad": _safe_quantile(heading, 0.95),
        "safety_violation_time_s": float(violation.sum() * dt),
        "safety_violation_ratio": float(violation.mean()) if len(violation) else 0.0,
        "thrust_saturation_ratio": saturation_ratio,
        "yaw_moment_saturation_ratio": yaw_saturation,
        "allocation_failure_ratio": float(1.0 - alloc_success.mean()) if len(alloc_success) else 0.0,
        "energy_proxy": float(energy.iloc[-1]) if len(energy) else float("nan"),
        "total_variation_of_control": total_variation,
        "solver_time_mean_ms": solver_time_mean,
        "solver_time_p95_ms": _safe_quantile(solve_ms, 0.95),
        "solver_time_max_ms": float(solve_ms.max(skipna=True)) if len(solve_ms) else float("nan"),
        "infeasible_rate": float(1.0 - solver_success.mean()) if len(solver_success) else 0.0,
        "failure": failure,
        "violation_severity": violation_severity,
        "max_overshoot_ratio": max_overshoot_ratio,
        "cvar_risk_p95": _safe_quantile(cvar_proxy, 0.95),
        "tail_position_cvar95_m": _cvar(pos, 0.95),
        "min_cbf_margin": min_cbf_margin,
        "mode_switch_count": mode_switch_count,
        "min_mode_dwell_time_s": min_mode_dwell_time_s,
        "dwell_time_violation_count": dwell_time_violation_count,
        "thrust_total_variation": thrust_total_var,
        "azimuth_total_variation": azimuth_total_var,
        "azimuth_metric_available": azimuth_metric_available,
        "thruster_chattering_index": chattering_index,
        "power_cap_active_rate": power_cap_active_rate,
        "allocation_residual_p95": alloc_residual_p95,
        "safety_filter_qp_success_rate": sf_success_rate,
        "safety_filter_infeasible_rate": sf_infeasible_rate,
        "safety_filter_solver_backend": sf_solver_backend,
        "safety_filter_solve_time_p50_ms": sf_solve_time_p50,
        "safety_filter_solve_time_p95_ms": sf_solve_time_p95,
        "safety_filter_solve_time_p99_ms": sf_solve_time_p99,
        "safety_filter_deadline_miss_rate": sf_deadline_miss_rate,
        "safety_filter_slack_mean": sf_slack_mean,
        "safety_filter_slack_p95": sf_slack_p95,
        "safety_filter_slack_p99": sf_slack_p99,
        "safety_filter_slack_active_rate": sf_slack_active_rate,
        "safety_filter_slack_integral": sf_slack_integral,
        "safety_filter_correction_mean": sf_correction_mean,
        "safety_filter_correction_p95": sf_correction_p95,
        "safety_filter_hocbf_margin_min": sf_hocbf_margin_min,
        "safety_filter_hocbf_margin_p01": sf_hocbf_margin_p01,
        "safety_filter_hocbf_margin_p05": sf_hocbf_margin_p05,
        "safety_filter_hocbf_violation_rate": sf_hocbf_violation_rate,
        "hocbf_soft_certificate_rate": sf_soft_certificate_rate,
        "hocbf_robust_disturbance_margin_p95": sf_robust_margin_term_p95,
        "hocbf_disturbance_accel_bound_mps2": sf_robust_disturbance_bound,
        "hocbf_nominal_constraint_margin_min": sf_nominal_margin_min,
        "safety_set_h_min": safety_set_h_min,
        **mode_ratios,
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
                if vals.size > 1:
                    row[f"{m}_std"] = float(np.std(vals, ddof=1))
                    row[f"{m}_ci95"] = float(1.96 * np.std(vals, ddof=1) / np.sqrt(vals.size))
                else:
                    # n=1: std 和 CI 无定义, 输出 NaN 而非 0 (避免误导)
                    row[f"{m}_std"] = float("nan")
                    row[f"{m}_ci95"] = float("nan")
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

    委托给 statistics.paired_comparison() 作为唯一统计方法来源。
    保持返回字段向后兼容。
    """
    from .statistics import paired_comparison
    result = paired_comparison(run_df, metric, baseline, proposed,
                              lower_is_better=True, alpha=alpha)
    # 字段映射: 保持向后兼容的列名 (baseline/proposed) + 新增方法标注
    return {
        "metric": result.get("metric", metric),
        "baseline": result.get("baseline", baseline),
        "proposed": result.get("proposed", proposed),
        "baseline_controller": result.get("baseline", baseline),
        "candidate_controller": result.get("proposed", proposed),
        "n_pairs": result.get("paired_samples", 0),
        "mean_baseline": result.get("baseline_mean", float("nan")),
        "mean_candidate": result.get("proposed_mean", float("nan")),
        "mean_diff": result.get("diff_mean", float("nan")),
        "diff_ci_lo": result.get("diff_ci_lo", float("nan")),
        "diff_ci_hi": result.get("diff_ci_hi", float("nan")),
        "relative_improvement_pct": result.get("relative_improvement_pct", result.get("improvement_pct", float("nan"))),
        "relative_improvement_ci_lo": result.get("relative_improvement_ci_lo", float("nan")),
        "relative_improvement_ci_hi": result.get("relative_improvement_ci_hi", float("nan")),
        "p_value": result.get("p_value", float("nan")),
        "p_value_holm": result.get("p_value_holm", float("nan")),
        "cohens_dz": result.get("cohens_dz", float("nan")),
        "significant": result.get("significant", False),
        "method": "paired_wilcoxon_signed_rank",
        "paired_by": "scenario_id + seed",
        "effect_size_direction": "positive_means_proposed_better",
        # 向后兼容别名
        "cohens_d": result.get("cohens_d", result.get("cohens_dz", float("nan"))),
        "n_baseline": result.get("n_baseline", 0),
        "n_proposed": result.get("n_proposed", 0),
        "baseline_mean": result.get("baseline_mean", float("nan")),
        "proposed_mean": result.get("proposed_mean", float("nan")),
        "improvement_pct": result.get("improvement_pct", float("nan")),
        "paired_samples": result.get("paired_samples", 0),
    }


def run_all_comparisons(
    run_df: pd.DataFrame,
    metrics_list: Optional[List[str]] = None,
    baseline: str = "pid",
    proposed: str = "full",
    alpha: float = 0.05,
    comparisons: Optional[List[Dict[str, str]]] = None,
) -> pd.DataFrame:
    """对多个指标运行统计比较, 应用 Holm-Bonferroni 校正。

    返回 DataFrame 包含 p_value (原始) 和 p_value_holm (校正后)。
    """
    from .statistics import holm_bonferroni
    if metrics_list is None:
        metrics_list = [
            "rms_position_error_m",
            "p95_position_error_m",
            "safety_violation_time_s",
            "thrust_saturation_ratio",
            "failure",
            "tail_position_cvar95_m",
        ]
    controller_set = set(run_df["controller"].astype(str).unique()) if "controller" in run_df.columns else set()
    pairs = comparisons or [{"baseline": baseline, "candidate": proposed}]
    results = []
    for pair in pairs:
        b = str(pair.get("baseline", baseline))
        c = str(pair.get("candidate", pair.get("proposed", proposed)))
        if b not in controller_set or c not in controller_set:
            # Explicit configs should not silently produce all-NaN tables.
            if comparisons is not None:
                raise ValueError(f"Statistical comparison references missing controller: {b} vs {c}")
            continue
        for m in metrics_list:
            if m in run_df.columns:
                results.append(statistical_comparison(run_df, m, b, c, alpha))

    if not results:
        return pd.DataFrame(results)

    # 应用 Holm-Bonferroni 校正
    p_values = [r.get("p_value", float("nan")) for r in results]
    valid_p = [p for p in p_values if not math.isnan(p)]
    if valid_p:
        holm_results = holm_bonferroni(valid_p)
        # 将校正后的 p 值映射回去
        holm_idx = 0
        for r in results:
            p = r.get("p_value", float("nan"))
            if math.isnan(p):
                r["p_value_holm"] = float("nan")
            else:
                r["p_value_holm"] = holm_results[holm_idx]
                r["significant_holm"] = r["p_value_holm"] < alpha
                holm_idx += 1
    else:
        for r in results:
            r["p_value_holm"] = float("nan")
            r["significant_holm"] = False

    return pd.DataFrame(results)


def save_tables(run_rows: Iterable[Dict], out_dir: Path, statistics_cfg: Optional[Dict] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_df = pd.DataFrame(list(run_rows))
    run_df.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    agg = aggregate_summary(run_df)
    agg.to_csv(out_dir / "aggregate_metrics_ci95.csv", index=False)
    # HOCBF/QP diagnostics table for method claims. This reports the actual
    # QP inequality margin (a@tau_safe + slack - b) when available, not only
    # the geometric safety-set value h(x).
    hocbf_cols = [
        "scenario_id", "controller", "seed",
        "safety_filter_hocbf_margin_min", "safety_filter_hocbf_margin_p01",
        "safety_filter_hocbf_margin_p05", "safety_filter_hocbf_violation_rate",
        "safety_filter_slack_active_rate", "safety_filter_slack_integral",
        "safety_filter_slack_p95", "safety_filter_slack_p99",
        "safety_filter_qp_success_rate", "safety_filter_infeasible_rate",
        "safety_filter_deadline_miss_rate", "safety_set_h_min",
        "hocbf_soft_certificate_rate", "hocbf_robust_disturbance_margin_p95",
        "hocbf_disturbance_accel_bound_mps2", "hocbf_nominal_constraint_margin_min",
        "mode_switch_count", "min_mode_dwell_time_s", "dwell_time_violation_count",
        "safety_filter_solver_backend",
    ]
    existing_hocbf_cols = [c for c in hocbf_cols if c in run_df.columns]
    if existing_hocbf_cols:
        run_df[existing_hocbf_cols].to_csv(out_dir / "hocbf_diagnostics.csv", index=False)
    # 保存统计比较结果 (如果有足够数据). Large parallel runs may
    # intentionally defer this step to a later --statistics-only pass.
    statistics_cfg = statistics_cfg or {}
    if isinstance(statistics_cfg, dict) and bool(statistics_cfg.get("skip", False)):
        return
    if len(run_df) > 0 and "controller" in run_df.columns:
        try:
            comparisons = run_all_comparisons(
                run_df,
                comparisons=statistics_cfg.get("comparisons") if isinstance(statistics_cfg, dict) else None,
            )
            if len(comparisons) > 0:
                comparisons.to_csv(out_dir / "statistical_comparisons.csv", index=False)
        except (ValueError, KeyError, TypeError, OSError) as e:
            import logging
            logging.getLogger(__name__).warning(
                "Statistical comparison failed (no statistical_comparisons.csv generated): %s", e
            )
