"""Statistical analysis module for Monte Carlo experiments.

Provides:
- Paired comparison by (scenario_id, seed)
- Effect size computation (Cohen's dz, Cohen's d with direction convention)
- Bootstrap confidence intervals
- Holm-Bonferroni multiple comparison correction
- Meta-summary across scenarios
- Per-group summary
- Failure case analysis
- Runtime summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import math
import numpy as np
import pandas as pd


def cohens_d_improvement(
    baseline: np.ndarray,
    proposed: np.ndarray,
    lower_is_better: bool = True,
) -> float:
    """Compute Cohen's d effect size.

    Convention: positive = proposed is better.
    For lower-is-better: d = (mean_baseline - mean_proposed) / pooled_std
    For higher-is-better: d = (mean_proposed - mean_baseline) / pooled_std
    """
    na, nb = len(baseline), len(proposed)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = math.sqrt(
        ((na - 1) * np.var(baseline, ddof=1) + (nb - 1) * np.var(proposed, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    if lower_is_better:
        return float((np.mean(baseline) - np.mean(proposed)) / pooled_std)
    else:
        return float((np.mean(proposed) - np.mean(baseline)) / pooled_std)


def bootstrap_ci(
    values: np.ndarray,
    statistic=None,
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng_seed: Optional[int] = None,
) -> tuple:
    """Bootstrap confidence interval for a statistic.

    Args:
        values: 1-D array of values
        statistic: function to compute on each bootstrap sample (default: np.mean)
        n_boot: number of bootstrap replicates
        alpha: significance level (e.g. 0.05 for 95% CI)
        rng_seed: random seed for reproducibility (None = auto-generate from data hash)

    Returns:
        (lo, hi) confidence interval bounds
    """
    if statistic is None:
        statistic = np.mean
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")

    if rng_seed is None:
        import hashlib
        rng_seed = int(hashlib.md5(values.tobytes()[:1024]).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(rng_seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats[i] = statistic(sample)

    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def paired_bootstrap_ci(
    baseline: np.ndarray,
    proposed: np.ndarray,
    statistic=None,
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng_seed: Optional[int] = None,
) -> tuple:
    """Paired bootstrap confidence interval for the difference statistic.

    Resamples paired differences (baseline - proposed) to compute CI
    for the mean improvement. This properly accounts for within-subject
    correlation across (scenario, seed) pairs.

    Args:
        baseline: baseline metric values (aligned by scenario+seed)
        proposed: proposed metric values (same order)
        statistic: function on differences (default: np.mean)
        n_boot: bootstrap replicates
        alpha: significance level (0.05 = 95% CI)
        rng_seed: reproducibility seed (None = auto-generate from data hash)

    Returns:
        (lo, hi) CI for the mean difference (baseline - proposed)
    """
    if statistic is None:
        statistic = np.mean
    baseline = np.asarray(baseline, dtype=float)
    proposed = np.asarray(proposed, dtype=float)
    n = min(len(baseline), len(proposed))
    if n < 2:
        return float("nan"), float("nan")

    diff = baseline[:n] - proposed[:n]
    if rng_seed is None:
        import hashlib
        rng_seed = int(hashlib.md5(diff.tobytes()[:1024]).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(rng_seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        stats[i] = statistic(sample)

    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def paired_cohens_dz(
    baseline: np.ndarray,
    proposed: np.ndarray,
    lower_is_better: bool = True,
) -> float:
    """Paired Cohen's dz effect size.

    Convention: positive = proposed is better.
    For lower-is-better: dz = mean(baseline - proposed) / std(baseline - proposed)
    For higher-is-better: dz = mean(proposed - baseline) / std(proposed - baseline)
    """
    diff = baseline - proposed if lower_is_better else proposed - baseline
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def holm_bonferroni(p_values: list) -> list:
    """Holm-Bonferroni multiple comparison correction.

    Args:
        p_values: list of raw p-values

    Returns:
        list of adjusted p-values (same order as input)
    """
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def paired_comparison(
    run_df: pd.DataFrame,
    metric: str,
    baseline: str,
    proposed: str,
    lower_is_better: bool = True,
    alpha: float = 0.05,
) -> Dict[str, float | str | bool]:
    """Paired statistical comparison by (scenario_id, seed)."""
    has_pair_cols = {"scenario_id", "seed"}.issubset(run_df.columns)

    if has_pair_cols:
        base_df = run_df[run_df["controller"] == baseline][["scenario_id", "seed", metric]].rename(
            columns={metric: "base_val"}
        )
        prop_df = run_df[run_df["controller"] == proposed][["scenario_id", "seed", metric]].rename(
            columns={metric: "prop_val"}
        )
        paired = base_df.merge(prop_df, on=["scenario_id", "seed"], how="inner").dropna()
    else:
        base_vals = run_df.loc[run_df["controller"] == baseline, metric].dropna().to_numpy()
        prop_vals = run_df.loc[run_df["controller"] == proposed, metric].dropna().to_numpy()
        n = min(len(base_vals), len(prop_vals))
        paired = pd.DataFrame({"base_val": base_vals[:n], "prop_val": prop_vals[:n]}) if n >= 2 else pd.DataFrame()

    if len(paired) < 2:
        return {
            "metric": metric, "baseline": baseline, "proposed": proposed,
            "p_value": float("nan"), "cohens_d": float("nan"),
            "significant": False, "paired_samples": 0,
            "effect_size_direction": "positive_means_proposed_better",
        }

    base_vals = paired["base_val"].to_numpy()
    prop_vals = paired["prop_val"].to_numpy()

    # Paired Wilcoxon signed-rank test
    diff = base_vals - prop_vals
    nonzero = diff[diff != 0]
    if len(nonzero) >= 2:
        try:
            from scipy.stats import wilcoxon
            res = wilcoxon(nonzero, alternative="two-sided")
            p = float(res.pvalue)
        except ImportError:
            p = float("nan")
    else:
        p = float("nan")

    # Paired Cohen's dz
    d = paired_cohens_dz(base_vals, prop_vals, lower_is_better)

    # Bootstrap CI for mean difference
    ci_lo, ci_hi = bootstrap_ci(diff, np.mean, n_boot=5000)

    # Bootstrap CI for relative improvement
    base_mean = float(np.mean(base_vals))
    denom = max(abs(base_mean), 1e-9)
    if lower_is_better:
        rel_improvements = (base_vals - prop_vals) / denom * 100.0
    else:
        rel_improvements = (prop_vals - base_vals) / denom * 100.0
    rel_ci_lo, rel_ci_hi = bootstrap_ci(rel_improvements, np.mean, n_boot=5000)

    return {
        "metric": metric,
        "baseline": baseline,
        "proposed": proposed,
        "p_value": p,
        "p_value_holm": float("nan"),  # filled in by run_all_comparisons
        "cohens_dz": d,
        "cohens_d": d,  # backward compat
        "significant": p < alpha if not math.isnan(p) else False,
        "n_baseline": len(base_vals),
        "n_proposed": len(prop_vals),
        "baseline_mean": float(np.mean(base_vals)),
        "proposed_mean": float(np.mean(prop_vals)),
        "improvement_pct": float(
            (np.mean(base_vals) - np.mean(prop_vals)) / max(abs(np.mean(base_vals)), 1e-9) * 100
        ) if lower_is_better else float(
            (np.mean(prop_vals) - np.mean(base_vals)) / max(abs(np.mean(base_vals)), 1e-9) * 100
        ),
        "diff_mean": float(np.mean(diff)),
        "diff_ci_lo": ci_lo,
        "diff_ci_hi": ci_hi,
        "relative_improvement_pct": float(
            (np.mean(base_vals) - np.mean(prop_vals)) / max(abs(np.mean(base_vals)), 1e-9) * 100
        ) if lower_is_better else float(
            (np.mean(prop_vals) - np.mean(base_vals)) / max(abs(np.mean(base_vals)), 1e-9) * 100
        ),
        "relative_improvement_ci_lo": rel_ci_lo,
        "relative_improvement_ci_hi": rel_ci_hi,
        "paired_samples": len(paired),
        "method": "paired_wilcoxon_signed_rank",
        "paired_by": "scenario_id + seed",
        "effect_size_direction": "positive_means_proposed_better",
    }


def meta_summary_across_scenarios(
    per_scenario_results: List[Dict],
) -> Dict[str, float]:
    """Compute meta-summary across scenarios."""
    ds = [r["cohens_d"] for r in per_scenario_results if not math.isnan(r.get("cohens_d", float("nan")))]
    ps = [r["p_value"] for r in per_scenario_results if not math.isnan(r.get("p_value", float("nan")))]

    return {
        "mean_effect_size": float(np.mean(ds)) if ds else float("nan"),
        "median_effect_size": float(np.median(ds)) if ds else float("nan"),
        "min_effect_size": float(np.min(ds)) if ds else float("nan"),
        "max_effect_size": float(np.max(ds)) if ds else float("nan"),
        "n_significant": sum(1 for p in ps if p < 0.05),
        "n_scenarios": len(per_scenario_results),
        "mean_p_value": float(np.mean(ps)) if ps else float("nan"),
    }


def compute_failure_cases(
    run_df: pd.DataFrame,
    failure_threshold: float = 1.5,
    safe_radius_col: str = "safe_region_radius",
) -> pd.DataFrame:
    """Extract failure cases from run data."""
    if "failure" not in run_df.columns:
        return pd.DataFrame()
    failures = run_df[run_df["failure"] == 1].copy()
    return failures


def compute_runtime_summary(
    run_df: pd.DataFrame,
    control_period_ms: float = 100.0,
) -> pd.DataFrame:
    """Compute runtime feasibility summary per controller."""
    if "solver_time_p95_ms" not in run_df.columns:
        return pd.DataFrame()

    rows = []
    for ctrl, g in run_df.groupby("controller"):
        solver_p95 = pd.to_numeric(g["solver_time_p95_ms"], errors="coerce").dropna()
        infeasible = pd.to_numeric(g.get("infeasible_rate", pd.Series(dtype=float)), errors="coerce").dropna()

        rows.append({
            "controller": ctrl,
            "solver_time_mean_ms": float(solver_p95.mean()) if len(solver_p95) else float("nan"),
            "solver_time_p95_ms": float(solver_p95.quantile(0.95)) if len(solver_p95) else float("nan"),
            "solver_time_max_ms": float(solver_p95.max()) if len(solver_p95) else float("nan"),
            "control_period_ms": control_period_ms,
            "infeasible_rate": float(infeasible.mean()) if len(infeasible) else 0.0,
            "real_time_feasible": bool(solver_p95.quantile(0.95) < control_period_ms) if len(solver_p95) else False,
        })

    return pd.DataFrame(rows)


def compute_effect_size_summary(
    run_df: pd.DataFrame,
    baseline: str = "pid",
    proposed: str = "full",
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute effect size summary across all metrics."""
    if metrics is None:
        metrics = [
            "rms_position_error_m", "p95_position_error_m", "safety_violation_time_s",
            "thrust_saturation_ratio", "failure", "tail_position_cvar95_m",
        ]

    lower_is_better_map = {
        "rms_position_error_m": True,
        "p95_position_error_m": True,
        "safety_violation_time_s": True,
        "thrust_saturation_ratio": True,
        "failure": True,
        "tail_position_cvar95_m": True,
        "energy_proxy": True,
        "solver_time_p95_ms": True,
    }

    results = []
    for m in metrics:
        if m not in run_df.columns:
            continue
        result = paired_comparison(
            run_df, m, baseline, proposed,
            lower_is_better=lower_is_better_map.get(m, True),
        )
        results.append(result)

    if not results:
        return pd.DataFrame()

    # Apply Holm-Bonferroni correction
    raw_p = [r["p_value"] for r in results if not math.isnan(r["p_value"])]
    if raw_p:
        adjusted = holm_bonferroni(raw_p)
        idx = 0
        for r in results:
            if not math.isnan(r["p_value"]):
                r["p_value_holm"] = adjusted[idx]
                r["significant_after_correction"] = adjusted[idx] < 0.05
                idx += 1
            else:
                r["p_value_holm"] = float("nan")
                r["significant_after_correction"] = False

    return pd.DataFrame(results)


def compute_group_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-group summary statistics.

    Groups are extracted from scenario_id prefix (e.g. A, B, C, D, E, F, G).
    """
    if "scenario_id" not in run_df.columns or "controller" not in run_df.columns:
        return pd.DataFrame()

    run_df = run_df.copy()
    from .scenarios import scenario_group_from_id
    run_df["group"] = run_df["scenario_id"].apply(scenario_group_from_id)

    metric_cols = [
        "rms_position_error_m", "p95_position_error_m", "p99_position_error_m",
        "safety_violation_time_s", "thrust_saturation_ratio", "failure",
        "tail_position_cvar95_m", "energy_proxy", "solver_time_p95_ms",
    ]
    available = [m for m in metric_cols if m in run_df.columns]

    rows = []
    for (group, ctrl), g in run_df.groupby(["group", "controller"]):
        row = {"group": group, "controller": ctrl, "n_scenarios": g["scenario_id"].nunique()}
        if "seed" in g.columns:
            row["n_seeds"] = g["seed"].nunique()
        for m in available:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            if len(vals) > 0:
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
                ci_lo, ci_hi = bootstrap_ci(vals.to_numpy(), np.mean, n_boot=1000)
                row[f"{m}_ci_lo"] = ci_lo
                row[f"{m}_ci_hi"] = ci_hi
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["group", "controller"]).reset_index(drop=True)
