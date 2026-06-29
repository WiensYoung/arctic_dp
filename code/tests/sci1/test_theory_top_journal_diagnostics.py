"""Theory-oriented diagnostics for high-end control-paper claims."""
from pathlib import Path

import numpy as np
import pandas as pd


def test_robust_hocbf_margin_tightens_rhs():
    from arctic_quasi_dp.sci1.control.hocbf import HOCBFParams, compute_hocbf_constraint

    state = np.array([6.0, 8.0, 0.0, 0.1, 0.0, 0.0])
    tau = np.array([20.0, 0.0, 0.0])
    p_nom = HOCBFParams(safe_radius_m=15.0, disturbance_accel_bound_mps2=0.0)
    p_rob = HOCBFParams(safe_radius_m=15.0, disturbance_accel_bound_mps2=0.02)
    c_nom = compute_hocbf_constraint(state, tau, (0.0, 0.0), 0.0, p_nom, 1.0, 1.5, 0.1)
    c_rob = compute_hocbf_constraint(state, tau, (0.0, 0.0), 0.0, p_rob, 1.0, 1.5, 0.1)
    assert c_rob["robust_disturbance_margin"] > 0.0
    assert c_rob["b_hocbf"] > c_nom["b_hocbf"]
    expected = 2.0 * np.hypot(6.0, 8.0) * 0.02
    assert abs(c_rob["robust_disturbance_margin"] - expected) < 1e-12


def test_safety_filter_reports_certificate_fields():
    from arctic_quasi_dp.sci1.control.hocbf import HOCBFParams
    from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter

    params = HOCBFParams(safe_radius_m=15.0, disturbance_accel_bound_mps2=0.01)
    filt = SoftHOCBFSafetyFilter(hocbf_params=params, constraint_mode="box")
    result = filt.filter(
        np.array([5.0, 1.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
        (0.0, 0.0),
        0.0,
        dt=0.1,
    )
    d = result.to_dict()
    for key in [
        "hocbf_robust_disturbance_margin",
        "hocbf_disturbance_accel_bound_mps2",
        "hocbf_nominal_constraint_margin",
        "hocbf_soft_certificate",
        "hocbf_a_norm",
        "hocbf_b",
        "hocbf_b_nominal",
    ]:
        assert key in d
    assert d["hocbf_disturbance_accel_bound_mps2"] == 0.01
    assert d["hocbf_robust_disturbance_margin"] >= 0.0


def test_dwell_time_metrics_and_certificate_rate_are_reported():
    from arctic_quasi_dp.sci1.metrics import summarize_run

    df = pd.DataFrame({
        "position_error": [1.0, 1.1, 1.2, 1.0, 0.9, 0.8],
        "heading_error": [0.0] * 6,
        "violation": [0] * 6,
        "solver_success": [1] * 6,
        "safety_filter_qp_success": [1] * 6,
        "safety_filter_solve_time_ms": [1, 2, 3, 4, 5, 6],
        "safety_filter_slack": [0.0] * 6,
        "hocbf_constraint_margin": [0.1, 0.2, 0.1, 0.05, 0.01, 0.02],
        "hocbf_nominal_constraint_margin": [0.2] * 6,
        "hocbf_robust_disturbance_margin": [0.05] * 6,
        "hocbf_disturbance_accel_bound_mps2": [0.01] * 6,
        "hocbf_soft_certificate": [1] * 6,
        "safety_set_h": [100.0] * 6,
        "supervisor_mode": [0, 0, 1, 1, 1, 0],
    })
    out = summarize_run(df, "I2", "cvar_soft_hocbf", seed=0, dt=0.5, safe_region_radius=10.0)
    assert out["hocbf_soft_certificate_rate"] == 1.0
    assert out["hocbf_robust_disturbance_margin_p95"] == 0.05
    assert out["hocbf_disturbance_accel_bound_mps2"] == 0.01
    assert out["mode_switch_count"] == 2
    assert out["min_mode_dwell_time_s"] == 0.5
    assert out["dwell_time_violation_count"] >= 1


def test_theory_docs_are_conservative():
    text = Path("docs/METHOD_THEORETICAL_ASSUMPTIONS.md").read_text(encoding="utf-8")
    assert "full-scale DP3 validation" in text
    assert "not a real-vessel formal proof" in text
    forbidden = ["proves full-scale", "guarantees real Arctic", "guarantees full-scale"]
    lowered = text.lower()
    for phrase in forbidden:
        assert phrase.lower() not in lowered
