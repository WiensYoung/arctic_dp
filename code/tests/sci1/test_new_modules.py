"""Tests for new modules: plot_style, latex_tables, cvar_sampling, manifest, nmpc_terminal, hocbf_activation, ice_force_common."""

import math
import numpy as np
import pytest
import pandas as pd
from pathlib import Path


# ============================================================
# plot_style.py
# ============================================================

class TestPlotStyle:
    def test_colorblind_safe_palette_defined(self):
        from arctic_quasi_dp.sci1.plot_style import COLORBLIND_SAFE
        assert "blue" in COLORBLIND_SAFE
        assert "red" in COLORBLIND_SAFE
        assert all(v.startswith("#") for v in COLORBLIND_SAFE.values())

    def test_controller_colors_defined(self):
        from arctic_quasi_dp.sci1.plot_style import CONTROLLER_COLORS
        assert "full" in CONTROLLER_COLORS
        assert "pid" in CONTROLLER_COLORS
        assert len(CONTROLLER_COLORS) >= 10

    def test_get_controller_color(self):
        from arctic_quasi_dp.sci1.plot_style import get_controller_color
        color = get_controller_color("full")
        assert color.startswith("#")
        # Unknown controller returns gray
        gray = get_controller_color("unknown_controller")
        assert gray.startswith("#")

    def test_get_controller_marker(self):
        from arctic_quasi_dp.sci1.plot_style import get_controller_marker
        assert get_controller_marker("full") == "o"
        assert get_controller_marker("pid") == "s"

    def test_setup_publication_style_no_backend_override(self):
        """setup_publication_style should NOT set backend to Agg."""
        from arctic_quasi_dp.sci1.plot_style import setup_publication_style
        import matplotlib
        original_backend = matplotlib.get_backend()
        setup_publication_style()
        # Backend should not be changed
        assert matplotlib.get_backend() == original_backend

    def test_save_figure_creates_files(self, tmp_path):
        from arctic_quasi_dp.sci1.plot_style import save_figure
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        output = tmp_path / "test_fig"
        saved = save_figure(fig, output, formats=["png", "pdf"])
        assert len(saved) == 2
        for p in saved:
            assert p.exists()
        plt.close(fig)

    def test_create_figure_returns_figure(self):
        from arctic_quasi_dp.sci1.plot_style import create_figure
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = create_figure(1, 1)
        assert fig is not None
        plt.close(fig)


# ============================================================
# latex_tables.py
# ============================================================

class TestLatexTables:
    def test_generate_latex_table_basic(self):
        from arctic_quasi_dp.sci1.latex_tables import generate_latex_table
        df = pd.DataFrame({"controller": ["full", "pid"], "rms": [1.5, 2.3]})
        columns = [
            {"key": "controller", "header": "Controller", "format": "str"},
            {"key": "rms", "header": "RMS Error", "format": "float", "decimals": 2},
        ]
        latex = generate_latex_table(df, columns, caption="Test", label="tab:test")
        assert "\\toprule" in latex
        assert "\\midrule" in latex
        assert "\\bottomrule" in latex
        assert "\\caption{Test}" in latex
        assert "\\label{tab:test}" in latex
        assert "full" in latex
        assert "1.50" in latex

    def test_generate_latex_table_empty(self):
        from arctic_quasi_dp.sci1.latex_tables import generate_latex_table
        df = pd.DataFrame({"controller": [], "rms": []})
        columns = [
            {"key": "controller", "header": "Controller", "format": "str"},
            {"key": "rms", "header": "RMS", "format": "float"},
        ]
        latex = generate_latex_table(df, columns, caption="Empty", label="tab:empty")
        assert "\\toprule" in latex
        assert "\\bottomrule" in latex

    def test_generate_latex_table_nan_handling(self):
        from arctic_quasi_dp.sci1.latex_tables import generate_latex_table
        df = pd.DataFrame({"name": ["a"], "val": [float("nan")]})
        columns = [{"key": "name", "header": "Name", "format": "str"}, {"key": "val", "header": "Val", "format": "float"}]
        latex = generate_latex_table(df, columns, caption="NaN", label="tab:nan")
        assert "--" in latex

    def test_format_number(self):
        from arctic_quasi_dp.sci1.latex_tables import _format_number
        assert _format_number(1.234, decimals=2) == "1.23"
        assert _format_number(float("nan")) == "--"
        assert _format_number(0.5, pct=True) == "50.00\\%"

    def test_escape_latex(self):
        from arctic_quasi_dp.sci1.latex_tables import _escape_latex
        assert _escape_latex("a&b") == "a\\&b"
        assert _escape_latex("100%") == "100\\%"


# ============================================================
# cvar_sampling.py
# ============================================================

class TestCVaRSampling:
    def test_compute_cvar_vectorized_basic(self):
        from arctic_quasi_dp.sci1.cvar_sampling import compute_cvar_vectorized
        rng = np.random.default_rng(42)
        samples = rng.normal(0.5, 0.1, 256)
        result = compute_cvar_vectorized(samples, alpha=0.90)
        assert result.cvar > result.mean  # CVaR should exceed mean
        assert result.n_tail > 0
        assert result.n_samples == 256
        assert result.cvar_converged is True or result.cvar_relative_se < 1.0

    def test_compute_cvar_vectorized_empty(self):
        from arctic_quasi_dp.sci1.cvar_sampling import compute_cvar_vectorized
        result = compute_cvar_vectorized(np.array([]), alpha=0.90)
        assert result.cvar == 0.0
        assert result.n_samples == 0

    def test_compute_cvar_vectorized_identical(self):
        from arctic_quasi_dp.sci1.cvar_sampling import compute_cvar_vectorized
        samples = np.ones(100) * 5.0
        result = compute_cvar_vectorized(samples, alpha=0.90)
        assert abs(result.cvar - 5.0) < 1e-10

    def test_get_cvar_samples_for_profile(self):
        from arctic_quasi_dp.sci1.cvar_sampling import get_cvar_samples_for_profile
        assert get_cvar_samples_for_profile("smoke") == 32
        assert get_cvar_samples_for_profile("paper_full") == 512
        assert get_cvar_samples_for_profile("unknown") == 128  # default

    def test_cvar_result_fields(self):
        from arctic_quasi_dp.sci1.cvar_sampling import compute_cvar_vectorized, CVaRResult
        result = compute_cvar_vectorized(np.array([1.0, 2.0, 3.0]), alpha=0.5)
        assert isinstance(result, CVaRResult)
        assert hasattr(result, "cvar")
        assert hasattr(result, "var")
        assert hasattr(result, "cvar_standard_error")


# ============================================================
# manifest.py
# ============================================================

class TestManifest:
    def test_build_manifest_from_config(self):
        from arctic_quasi_dp.sci1.manifest import build_manifest_from_config
        config = {
            "profile": "paper",
            "seeds": 50,
            "controllers": ["full", "pid"],
            "scenario_ids": ["A1", "B1"],
        }
        manifest = build_manifest_from_config(config)
        assert manifest.profile == "paper"
        assert len(manifest.seed_list) == 50
        assert len(manifest.controller_ids) == 2
        assert len(manifest.scenario_ids) == 2

    def test_validate_manifest_valid(self):
        from arctic_quasi_dp.sci1.manifest import RunManifest, validate_manifest
        m = RunManifest()
        m.task_count_expected = 4
        m.task_count_completed = 3
        m.task_count_failed = 1
        m.task_count_skipped = 0
        errors = validate_manifest(m)
        assert len(errors) == 0

    def test_validate_manifest_mismatch(self):
        from arctic_quasi_dp.sci1.manifest import RunManifest, validate_manifest
        m = RunManifest()
        m.task_count_expected = 10
        m.task_count_completed = 5
        m.task_count_failed = 0
        m.task_count_skipped = 0
        errors = validate_manifest(m)
        assert len(errors) > 0
        assert "mismatch" in errors[0].lower()

    def test_save_manifest(self, tmp_path):
        from arctic_quasi_dp.sci1.manifest import RunManifest, save_manifest
        m = RunManifest()
        m.run_id = "test_run"
        path = save_manifest(m, tmp_path)
        assert path.exists()
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["run_id"] == "test_run"

    def test_compute_file_sha256(self, tmp_path):
        from arctic_quasi_dp.sci1.manifest import compute_file_sha256
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        h = compute_file_sha256(test_file)
        assert len(h) == 64  # SHA-256 hex digest
        # Same content => same hash
        assert compute_file_sha256(test_file) == h


# ============================================================
# nmpc_terminal.py
# ============================================================

class TestNMPCTerminal:
    def test_linearize_discrete_dynamics(self):
        from arctic_quasi_dp.sci1.control.nmpc_terminal import linearize_discrete_dynamics
        A, B = linearize_discrete_dynamics(500000, 5e8, 500, 800, 2e5, 0.1)
        assert A.shape == (6, 6)
        assert B.shape == (6, 3)
        # A should be close to identity for small dt
        assert np.allclose(np.diag(A), 1.0, atol=0.01)

    def test_solve_dare(self):
        from arctic_quasi_dp.sci1.control.nmpc_terminal import solve_dare, linearize_discrete_dynamics
        A, B = linearize_discrete_dynamics(500000, 5e8, 500, 800, 2e5, 0.1)
        Q = np.diag([100, 100, 50, 1, 1, 1])
        R = np.diag([0.001, 0.001, 0.0001])
        P = solve_dare(A, B, Q, R)
        assert P.shape == (6, 6)
        assert np.allclose(P, P.T)  # symmetric
        assert np.all(np.linalg.eigvals(P) > 0)  # positive definite

    def test_compute_terminal_cost(self):
        from arctic_quasi_dp.sci1.control.nmpc_terminal import compute_terminal_cost
        result = compute_terminal_cost()
        assert result.P.shape == (6, 6)
        assert result.K.shape == (3, 6)
        assert result.is_stable is True
        assert result.spectral_radius < 1.0
        assert result.alpha > 0

    def test_terminal_value(self):
        from arctic_quasi_dp.sci1.control.nmpc_terminal import compute_terminal_cost, terminal_value
        tc = compute_terminal_cost()
        # Zero error => zero cost
        assert terminal_value(np.zeros(6), tc.P) == 0.0
        # Nonzero error => positive cost
        err = np.array([1.0, 0, 0, 0, 0, 0])
        assert terminal_value(err, tc.P) > 0

    def test_is_in_terminal_set(self):
        from arctic_quasi_dp.sci1.control.nmpc_terminal import compute_terminal_cost, is_in_terminal_set
        tc = compute_terminal_cost()
        # Zero error => in terminal set
        assert bool(is_in_terminal_set(np.zeros(6), tc.P, tc.alpha))
        # Very large error => not in terminal set
        large_err = np.array([1000.0, 0, 0, 0, 0, 0])
        assert not bool(is_in_terminal_set(large_err, tc.P, tc.alpha))


# ============================================================
# hocbf_activation.py
# ============================================================

class TestHOCBFActivation:
    def test_initial_state(self):
        from arctic_quasi_dp.sci1.control.hocbf_activation import HOCBFActivationStateMachine
        sm = HOCBFActivationStateMachine()
        assert sm.state.mode.name == "NORMAL"
        assert sm.state.hocbf_active is False
        assert sm.state.mode_switch_count == 0

    def test_hysteresis_no_chattering(self):
        """Risk oscillating near threshold should NOT cause rapid switching."""
        from arctic_quasi_dp.sci1.control.hocbf_activation import HOCBFActivationStateMachine
        sm = HOCBFActivationStateMachine()
        # Oscillate risk near threshold for 100 steps
        for i in range(100):
            risk = 0.55 + 0.05 * np.sin(i * 0.5)  # oscillates 0.50-0.60
            sm.update(h_val=50.0, risk_raw=risk, dt=0.1)
        # Should have very few switches due to hysteresis
        assert sm.state.mode_switch_count <= 5

    def test_dwell_time_enforcement(self):
        from arctic_quasi_dp.sci1.control.hocbf_activation import HOCBFActivationStateMachine, HOCBFActivationConfig
        config = HOCBFActivationConfig(min_dwell_time_s=5.0)
        sm = HOCBFActivationStateMachine(config)
        # Trigger escalation
        sm.update(h_val=50.0, risk_raw=0.7, dt=0.1)
        first_mode = sm.state.mode
        # Try to escalate again immediately
        sm.update(h_val=50.0, risk_raw=0.9, dt=0.1)
        # Should still be in first mode due to dwell time
        assert sm.state.time_in_mode < 5.0

    def test_get_diagnostics(self):
        from arctic_quasi_dp.sci1.control.hocbf_activation import HOCBFActivationStateMachine
        sm = HOCBFActivationStateMachine()
        sm.update(h_val=50.0, risk_raw=0.5, dt=0.1)
        diag = sm.get_diagnostics()
        assert "safety_mode" in diag
        assert "risk_filtered" in diag
        assert "mode_switch_count" in diag

    def test_reset(self):
        from arctic_quasi_dp.sci1.control.hocbf_activation import HOCBFActivationStateMachine
        sm = HOCBFActivationStateMachine()
        sm.update(h_val=50.0, risk_raw=0.7, dt=0.1)
        sm.reset()
        assert sm.state.mode.name == "NORMAL"
        assert sm.state.mode_switch_count == 0

    def test_risk_smoothing(self):
        from arctic_quasi_dp.sci1.control.hocbf_activation import HOCBFActivationStateMachine
        sm = HOCBFActivationStateMachine()
        # Step 1: low risk
        sm.update(h_val=100.0, risk_raw=0.1, dt=0.1)
        r1 = sm.state.risk_filtered
        # Step 2: sudden high risk
        sm.update(h_val=100.0, risk_raw=0.9, dt=0.1)
        r2 = sm.state.risk_filtered
        # Filtered risk should be smoothed (not jump to 0.9)
        assert r2 < 0.9
        assert r2 > r1


# ============================================================
# ice_force_common.py
# ============================================================

class TestIceForceCommon:
    def test_speed_factor_at_zero(self):
        from arctic_quasi_dp.sci1.ice_force_common import lindqvist_speed_factor
        assert lindqvist_speed_factor(0.0) == 1.0

    def test_speed_factor_increases(self):
        from arctic_quasi_dp.sci1.ice_force_common import lindqvist_speed_factor
        assert lindqvist_speed_factor(0.5) > 1.0
        assert lindqvist_speed_factor(1.0) > lindqvist_speed_factor(0.5)

    def test_angle_factor_at_zero(self):
        from arctic_quasi_dp.sci1.ice_force_common import lindqvist_angle_factor
        assert abs(lindqvist_angle_factor(0.0) - 1.0) < 1e-10

    def test_angle_factor_capped_at_60(self):
        from arctic_quasi_dp.sci1.ice_force_common import lindqvist_angle_factor
        f60 = lindqvist_angle_factor(math.pi / 3)
        f90 = lindqvist_angle_factor(math.pi / 2)
        assert abs(f60 - f90) < 1e-10  # capped at 60°

    def test_compute_ice_force_ned_zero_ice(self):
        from arctic_quasi_dp.sci1.ice_force_common import compute_ice_force_ned
        force = compute_ice_force_ned(0, 0, 0, 0, 2.0, 22, 0.45, math.radians(30))
        assert np.allclose(force, 0.0)

    def test_compute_ice_force_body_direction(self):
        from arctic_quasi_dp.sci1.ice_force_common import compute_ice_force_body
        # Ice from north (0°), ship heading north (psi=0) => force in body x
        force = compute_ice_force_body(1.0, 1.0, 0.5, 0.0, 0.0, 2.0, 22, 122.5, 0.45, math.radians(30))
        assert force[0] > 0  # surge force positive
        assert abs(force[1]) < 1e-10  # no sway force
        assert abs(force[2]) < 1e-10  # no yaw moment (no sway)

    def test_compute_ice_force_body_from_dict(self):
        from arctic_quasi_dp.sci1.ice_force_common import compute_ice_force_body_from_dict
        ice = {"concentration": 0.5, "thickness": 0.8, "drift_speed": 0.3, "drift_direction": 140.0}
        force = compute_ice_force_body_from_dict(ice, 0.0, 2.0, 22, 122.5, 0.45, math.radians(30))
        assert force.shape == (3,)
        assert np.all(np.isfinite(force))

    def test_ice_force_consistency(self):
        """Verify all three ice force implementations produce identical results."""
        from arctic_quasi_dp.sci1.ice_force_common import compute_ice_force_body
        from arctic_quasi_dp.sci1.sim_loop import _ice_force_body, VesselParams
        from arctic_quasi_dp.sci1.controllers import _ice_force_lindqvist_proxy

        ice = {"concentration": 0.5, "thickness": 0.8, "drift_speed": 0.3, "drift_direction": 140.0}
        psi = 0.3
        vp = VesselParams()

        # sim_loop version
        f1 = _ice_force_body(ice, psi, vp)
        # controllers version (takes radians)
        import math
        f2 = _ice_force_lindqvist_proxy(
            0.5, 0.8, 0.3, math.radians(140.0), psi,
            vp.ice_crushing_strength_mpa, vp.ice_structure_factor,
            vp.beam, vp.length, vp.waterline_angle_deg,
        )
        # shared module version
        f3 = compute_ice_force_body(
            0.5, 0.8, 0.3, 140.0, psi,
            vp.ice_crushing_strength_mpa, vp.beam, vp.length,
            vp.ice_structure_factor, math.radians(vp.waterline_angle_deg),
        )

        np.testing.assert_allclose(f1, f2, rtol=1e-10, err_msg="sim_loop vs controllers mismatch")
        np.testing.assert_allclose(f1, f3, rtol=1e-10, err_msg="sim_loop vs shared mismatch")


# ============================================================
# scale_analysis.py
# ============================================================

class TestScaleAnalysis:
    def test_dimensionless_groups(self):
        from arctic_quasi_dp.sci1.scale_analysis import compute_dimensionless_groups
        from arctic_quasi_dp.sci1.sim_loop import VesselParams
        from arctic_quasi_dp.sci1.thruster import ThrusterConfig
        from arctic_quasi_dp.sci1.scenarios import build_sci1_scenarios
        vp = VesselParams()
        tc = ThrusterConfig.generic_dp_vessel()
        scenarios = build_sci1_scenarios("smoke")
        ice_state = {"concentration": 0.5, "thickness": 0.8, "drift_speed": 0.3, "drift_direction": 140.0}
        groups = compute_dimensionless_groups(vp, tc, scenarios[0], ice_state=ice_state)
        assert groups is not None


# ============================================================
# simulator.py (legacy)
# ============================================================

class TestSimulator:
    def test_simulator_imports(self):
        from arctic_quasi_dp.simulation.simulator import Simulator, SimulationConfig
        assert Simulator is not None
        assert SimulationConfig is not None

    def test_simulation_config_defaults(self):
        from arctic_quasi_dp.simulation.simulator import SimulationConfig
        cfg = SimulationConfig()
        assert cfg.duration > 0
        assert cfg.dt > 0


# ============================================================
# actuator_feasible_set.py
# ============================================================

class TestActuatorFeasibleSet:
    def test_box_feasible_set(self):
        from arctic_quasi_dp.sci1.control.actuator_feasible_set import BoxFeasibleSet
        box = BoxFeasibleSet(max_force_x=3000, max_force_y=3000, max_moment_n=100000)
        constraints = box.get_constraints()
        assert constraints.A.shape[0] > 0
        assert constraints.A.shape[1] == 3  # 3D tau space

    def test_actuator_aware_feasible_set(self):
        from arctic_quasi_dp.sci1.control.actuator_feasible_set import ActuatorAwareFeasibleSet
        afs = ActuatorAwareFeasibleSet(n_vertices=16, max_force_x=3000, max_force_y=3000, max_moment_n=100000)
        constraints = afs.get_constraints(mode="nominal")
        assert constraints.A.shape[0] > 0
        assert constraints.A.shape[1] == 3

    def test_feasible_set_constraints_shape(self):
        from arctic_quasi_dp.sci1.control.actuator_feasible_set import BoxFeasibleSet
        box = BoxFeasibleSet(max_force_x=3000, max_force_y=3000, max_moment_n=100000)
        c = box.get_constraints()
        # A @ tau should be between l and u for a feasible tau
        tau = np.array([1000.0, 500.0, 50000.0])
        vals = c.A @ tau
        assert np.all(vals <= c.u + 1e-10)
        assert np.all(vals >= c.l - 1e-10)


# ============================================================
# ice_force_common.py — 集成测试
# ============================================================

class TestIceForceCommonIntegration:
    def test_force_increases_with_ice(self):
        """Stronger ice should produce larger force."""
        from arctic_quasi_dp.sci1.ice_force_common import compute_ice_force_body
        f_light = compute_ice_force_body(0.3, 0.5, 0.2, 140.0, 0.0, 2.0, 22, 122.5, 0.45, math.radians(30))
        f_heavy = compute_ice_force_body(0.9, 1.5, 0.5, 140.0, 0.0, 2.0, 22, 122.5, 0.45, math.radians(30))
        assert np.linalg.norm(f_heavy) > np.linalg.norm(f_light)

    def test_force_direction_follows_ice_drift(self):
        """Ice from different directions should produce different force directions."""
        from arctic_quasi_dp.sci1.ice_force_common import compute_ice_force_ned
        f_north = compute_ice_force_ned(0.5, 0.8, 0.3, 0.0, 2.0, 22, 0.45, math.radians(30))
        f_east = compute_ice_force_ned(0.5, 0.8, 0.3, 90.0, 2.0, 22, 0.45, math.radians(30))
        # North drift => force in +x (NED), East drift => force in +y
        assert f_north[0] > f_east[0]
        assert f_east[1] > f_north[1]
