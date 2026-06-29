"""P0 submission-readiness regression tests."""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def test_osqp_solver_is_cached_after_first_step():
    from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter

    f = SoftHOCBFSafetyFilter(constraint_mode="box")
    state = np.array([5.0, 2.0, 0.1, 0.2, 0.1, 0.0])
    tau = np.array([50.0, 25.0, 100.0])
    r1 = f.filter(state, tau, (0.0, 0.0), 0.0, dt=0.1)
    r2 = f.filter(state, tau * 1.1, (0.0, 0.0), 0.0, dt=0.1)
    if r1.solver_backend == "osqp" and r2.solver_backend == "osqp":
        assert r2.solver_setup_count == 1
        assert r2.solver_update_count >= 1
        assert r2.osqp_reused_factorization is True


def test_hocbf_margin_is_constraint_margin_not_safety_set_only():
    from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter

    f = SoftHOCBFSafetyFilter(constraint_mode="box")
    state = np.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    tau = np.array([0.0, 0.0, 0.0])
    r = f.filter(state, tau, (0.0, 0.0), 0.0, dt=0.1)
    assert hasattr(r, "safety_set_h")
    assert hasattr(r, "hocbf_margin")
    assert np.isfinite(r.hocbf_margin)


def test_scale_comparison_config_is_lightweight_and_valid_ids():
    from arctic_quasi_dp.sci1.scenarios import build_sci1_scenarios

    cfg = yaml.safe_load(Path("configs/sci1/sci1_scale_comparison.yaml").read_text())
    assert cfg["profile"] == "scale_comparison"
    assert int(cfg["seeds"]) == 1
    assert cfg["output"]["save_traces"] is False
    valid_ids = {s.scenario_id for s in build_sci1_scenarios("scale_comparison")}
    missing = set(cfg["scenarios"]["include_ids"]) - valid_ids
    assert not missing


def test_real_replay_config_uses_real_scenario_not_mock_alias():
    cfg = yaml.safe_load(Path("configs/sci1/sci1_real_replay_h1.yaml").read_text())
    ids = cfg["scenarios"]["include_ids"]
    assert "H1_real_copernicus_era5_replay" in ids
    assert "H1_data_driven_80N" not in ids
    runtime = cfg.get("runtime", {})
    assert runtime.get("fail_fast") is True
    assert runtime.get("allow_mock_fixture") is False


def test_leso_initial_velocity_is_ned_frame():
    from arctic_quasi_dp.sci1.baseline_controllers import LESOADRCController, LESOADRCParams

    c = LESOADRCController(LESOADRCParams())
    c.set_target(0.0, 0.0, 0.0)
    # Heading 90 deg: body surge u=1 m/s corresponds to NED y velocity +1.
    state = np.array([0.0, 0.0, np.pi / 2, 1.0, 0.0, 0.0])
    c.compute_control(state, dt=0.1)
    assert abs(c._z_x[1]) < 0.5
    assert c._z_y[1] > 0.5
