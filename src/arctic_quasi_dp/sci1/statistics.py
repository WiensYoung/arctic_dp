"""Statistical analysis module for Monte Carlo experiments.

Provides:
- Paired comparison by (scenario_id, seed)
- Effect size computation (Cohen's d with direction convention)
- Meta-summary across scenarios
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

    # Wilcoxon rank-sum
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(prop_vals, base_vals, alternative="two-sided")
        p = float(p)
    except ImportError:
        p = float("nan")

    d = cohens_d_improvement(base_vals, prop_vals, lower_is_better)

    return {
        "metric": metric,
        "baseline": baseline,
        "proposed": proposed,
        "p_value": p,
        "cohens_d": d,
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
        "paired_samples": len(paired),
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

    return pd.DataFrame(results)
