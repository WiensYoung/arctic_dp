"""Table generation for paper-quality output.

Generates 6 tables:
1. table1_scenario_matrix.csv
2. table2_controller_ablation_matrix.csv
3. table3_main_metrics.csv
4. table4_ablation_summary.csv
5. table5_data_provenance.csv
6. table6_runtime_feasibility.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .scenarios import build_sci1_scenarios, SCI1Scenario


# ---------- Table 1: Scenario Matrix ----------

def generate_table1_scenario_matrix(profile: str = "paper") -> pd.DataFrame:
    """Generate scenario matrix table."""
    scenarios = build_sci1_scenarios(profile)
    rows = []
    for s in scenarios:
        rows.append({
            "scenario_id": s.scenario_id,
            "group": s.group,
            "ice_condition": f"c={s.ice_concentration:.2f}, h={s.ice_thickness:.1f}m, v={s.ice_drift_speed:.2f}m/s" if s.ice_concentration > 0 else "open_water",
            "environment_disturbance": "ice" if s.ice_concentration > 0 else "none",
            "thruster_fault": s.degradation_name if s.degradation_name != "no_fault" else "none",
            "sensor_fault": "none",
            "expected_challenge": s.description,
            "data_source_type": s.evidence_level,
        })
    return pd.DataFrame(rows)


# ---------- Table 2: Controller / Ablation Matrix ----------

_CONTROLLER_CAPABILITIES = {
    "pid": {"observer": False, "cvar": False, "cbf": False, "thruster_deg": False,
            "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "smc": {"observer": False, "cvar": False, "cbf": False, "thruster_deg": False,
            "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "adrc": {"observer": True, "cvar": False, "cbf": False, "thruster_deg": False,
             "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "leso_adrc": {"observer": True, "cvar": False, "cbf": False, "thruster_deg": False,
                  "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "precision": {"observer": False, "cvar": False, "cbf": False, "thruster_deg": False,
                  "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "ice_aware": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": False,
                  "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "quasi_dp": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": False,
                 "mode_supervisor": False, "quasi_dp": True, "escape": False, "oracle": False, "casadi": False},
    "escape": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": False,
               "mode_supervisor": False, "quasi_dp": False, "escape": True, "oracle": False, "casadi": False},
    "full": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": True,
             "mode_supervisor": True, "quasi_dp": True, "escape": True, "oracle": False, "casadi": False},
    "nmpc": {"observer": False, "cvar": False, "cbf": True, "thruster_deg": False,
             "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": True},
    "no_cbf": {"observer": True, "cvar": True, "cbf": False, "thruster_deg": True,
               "mode_supervisor": True, "quasi_dp": True, "escape": True, "oracle": False, "casadi": False},
    "no_cvar": {"observer": True, "cvar": False, "cbf": True, "thruster_deg": True,
                "mode_supervisor": True, "quasi_dp": True, "escape": True, "oracle": False, "casadi": False},
    "no_observer": {"observer": False, "cvar": True, "cbf": True, "thruster_deg": True,
                    "mode_supervisor": True, "quasi_dp": True, "escape": True, "oracle": False, "casadi": False},
    "no_fallback": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": True,
                    "mode_supervisor": True, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "oracle_full": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": True,
                    "mode_supervisor": True, "quasi_dp": True, "escape": True, "oracle": True, "casadi": False},
    "lqg": {"observer": True, "cvar": False, "cbf": False, "thruster_deg": False,
            "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "dob_nmpc": {"observer": True, "cvar": False, "cbf": True, "thruster_deg": False,
                 "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": True},
    "robust_mpc": {"observer": True, "cvar": False, "cbf": False, "thruster_deg": False,
                   "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "tube_mpc": {"observer": True, "cvar": False, "cbf": False, "thruster_deg": False,
                 "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "fixed_soft_hocbf": {"observer": False, "cvar": False, "cbf": True, "thruster_deg": False,
                         "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "cvar_soft_hocbf": {"observer": False, "cvar": True, "cbf": True, "thruster_deg": False,
                        "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "no_safety_filter": {"observer": False, "cvar": False, "cbf": False, "thruster_deg": False,
                         "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "no_tail_risk": {"observer": False, "cvar": False, "cbf": True, "thruster_deg": False,
                     "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "no_cvar_proxy": {"observer": False, "cvar": False, "cbf": True, "thruster_deg": False,
                      "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
}


def generate_table2_controller_matrix() -> pd.DataFrame:
    """Generate controller capability matrix."""
    rows = []
    for name, caps in _CONTROLLER_CAPABILITIES.items():
        row = {"controller": name}
        row.update(caps)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- Table 3: Main Metrics ----------

def generate_table3_main_metrics(
    run_df: pd.DataFrame,
    aggregate_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate main metrics table aggregated by controller (across all scenarios)."""
    if aggregate_df is None:
        from .metrics import aggregate_summary
        aggregate_df = aggregate_summary(run_df)

    # 按 controller 分组, 计算跨场景的均值和 CI
    if "controller" in aggregate_df.columns:
        # 识别数值列 (均值、标准差、CI)
        metric_cols = [c for c in aggregate_df.columns
                       if any(c.endswith(s) for s in ("_mean", "_std", "_ci_lo", "_ci_hi"))
                       and c != "controller"]
        if metric_cols:
            grouped = aggregate_df.groupby("controller")[metric_cols].mean().reset_index()
            return grouped

    return aggregate_df


# ---------- Table 4: Ablation Summary ----------

def generate_table4_ablation_summary(
    run_df: pd.DataFrame,
    baseline_controller: str = "full",
) -> pd.DataFrame:
    """Generate ablation summary: delta metrics for each ablation vs full."""
    from .statistics import paired_comparison

    ablations = ["no_cbf", "no_cvar", "no_observer", "no_fallback"]
    metrics = ["rms_position_error_m", "p95_position_error_m", "safety_violation_time_s", "failure"]
    results = []

    for ablation in ablations:
        if ablation not in run_df["controller"].values:
            continue
        for m in metrics:
            if m not in run_df.columns:
                continue
            result = paired_comparison(run_df, m, ablation, baseline_controller, lower_is_better=True)
            result["ablation"] = ablation
            results.append(result)

    return pd.DataFrame(results)


# ---------- Table 5: Data Provenance ----------

def generate_table5_data_provenance() -> pd.DataFrame:
    """Generate data provenance table."""
    from .data_sources import DATA_SOURCES

    rows = []
    for src in DATA_SOURCES:
        rows.append({
            "variable": src.variable,
            "source_name": src.name,
            "source_type": src.authority,
            "observed_or_synthetic": src.status,
            "time_range": src.temporal_resolution,
            "spatial_region": src.spatial_resolution,
            "access_note": src.access_note,
            "used_in_scenarios": src.intended_use,
        })
    return pd.DataFrame(rows)


# ---------- Table 6: Runtime Feasibility ----------

def generate_table6_runtime(run_df: pd.DataFrame) -> pd.DataFrame:
    """Generate runtime feasibility table."""
    from .statistics import compute_runtime_summary
    return compute_runtime_summary(run_df)


# ---------- Save All Tables ----------

def save_all_tables(
    run_df: pd.DataFrame,
    out_dir: Path,
    profile: str = "paper",
) -> None:
    """Save all paper tables to out_dir (should be the summary/ directory)."""
    summary_dir = out_dir
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Table 1
    t1 = generate_table1_scenario_matrix(profile)
    t1.to_csv(summary_dir / "table1_scenario_matrix.csv", index=False)

    # Table 2
    t2 = generate_table2_controller_matrix()
    t2.to_csv(summary_dir / "table2_controller_ablation_matrix.csv", index=False)

    # Table 3
    if len(run_df) > 0:
        t3 = generate_table3_main_metrics(run_df)
        t3.to_csv(summary_dir / "table3_main_metrics.csv", index=False)

        # Table 4
        t4 = generate_table4_ablation_summary(run_df)
        t4.to_csv(summary_dir / "table4_ablation_summary.csv", index=False)

        # Table 6
        t6 = generate_table6_runtime(run_df)
        t6.to_csv(summary_dir / "table6_runtime_feasibility.csv", index=False)

    # Table 5
    try:
        t5 = generate_table5_data_provenance()
        t5.to_csv(summary_dir / "table5_data_provenance.csv", index=False)
    except Exception as e:
        import warnings
        warnings.warn(f"Table 5 (data provenance) generation failed: {e}")

    # Per-group summary
    if len(run_df) > 0 and "scenario_id" in run_df.columns:
        try:
            from .statistics import compute_group_summary, compute_effect_size_summary
            group_df = compute_group_summary(run_df)
            if len(group_df) > 0:
                group_df.to_csv(summary_dir / "aggregate_by_group.csv", index=False)

            # Effect sizes with Holm correction
            effect_df = compute_effect_size_summary(run_df)
            if len(effect_df) > 0:
                effect_df.to_csv(summary_dir / "effect_sizes_holm.csv", index=False)
        except Exception as e:
            import warnings
            warnings.warn(f"Group summary or effect size generation failed: {e}")
