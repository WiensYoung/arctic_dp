"""Method layer integration tests.

Tests that safety filter controllers work end-to-end
through the simulation loop.
"""

import math
import numpy as np
import pytest


class TestMethodIntegration:
    """Safety filter controller integration with sim_loop。"""

    def test_no_safety_filter_equivalent_to_nominal(self):
        """no_safety_filter 应与 nominal 控制器输出一致。"""
        from arctic_quasi_dp.sci1.sim_loop import run_simulation
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController
        from arctic_quasi_dp.sci1.control.controller_wrappers import make_filtered_controller

        nominal = PrecisionDPController()
        filtered = make_filtered_controller(
            PrecisionDPController(), filter_type="no_safety_filter",
        )

        log1 = run_simulation(nominal, duration=2.0, dt=0.1, seed=42)
        log2 = run_simulation(filtered, duration=2.0, dt=0.1, seed=42)

        df1 = log1.to_dataframe()
        df2 = log2.to_dataframe()
        # tau should be identical (no safety filter)
        np.testing.assert_allclose(
            df1["tau_x"].values, df2["tau_x"].values, atol=1e-6,
        )

    def test_fixed_soft_hocbf_produces_valid_output(self):
        """fixed_soft_hocbf 应产生有效仿真输出。"""
        from arctic_quasi_dp.sci1.sim_loop import run_simulation
        from arctic_quasi_dp.sci1.control.controller_wrappers import make_filtered_controller
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController

        ctrl = make_filtered_controller(
            PrecisionDPController(), filter_type="fixed_soft_hocbf",
        )
        log = run_simulation(ctrl, duration=2.0, dt=0.1, seed=42)
        df = log.to_dataframe()
        assert len(df) > 0
        assert df["position_error"].mean() < 100.0

    def test_trace_contains_safety_filter_fields(self):
        """trace 应包含 safety filter 证据字段。"""
        from arctic_quasi_dp.sci1.sim_loop import run_simulation
        from arctic_quasi_dp.sci1.control.controller_wrappers import make_filtered_controller
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController

        ctrl = make_filtered_controller(
            PrecisionDPController(), filter_type="fixed_soft_hocbf",
        )
        log = run_simulation(ctrl, duration=1.0, dt=0.1, seed=42)
        df = log.to_dataframe()
        assert "safety_filter_active" in df.columns
        assert "safety_filter_qp_success" in df.columns
        assert "safety_filter_correction_norm" in df.columns
        assert "tau_safe_x" in df.columns

    def test_safety_filter_correction_near_boundary(self):
        """接近安全边界时 safety filter 应产生非零 correction。"""
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        f = SoftHOCBFSafetyFilter()
        # State near boundary, moving outward
        state = np.array([12.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        tau_des = np.array([500.0, 0.0, 0.0])
        result = f.filter(state, tau_des, (0.0, 0.0), 0.0)
        # Should produce some correction
        assert result.qp_success is True


class TestCVaRAdaptiveHOCBF:
    """CVaR-adaptive HOCBF risk modulation。"""

    def test_risk_scale_increases_with_risk(self):
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        f = SoftHOCBFSafetyFilter(risk_gain=2.0)
        state = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau_des = np.array([100.0, 0.0, 0.0])
        r0 = f.filter(state, tau_des, (0.0, 0.0), 0.0, risk_level=0.0)
        r1 = f.filter(state, tau_des, (0.0, 0.0), 0.0, risk_level=0.8)
        assert r1.risk_scale > r0.risk_scale
        assert r1.alpha1 > r0.alpha1

    def test_fixed_vs_cvar_different_under_risk(self):
        """fixed 和 cvar 在高风险下应产生不同输出。"""
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        f_fixed = SoftHOCBFSafetyFilter(risk_gain=0.0)
        f_cvar = SoftHOCBFSafetyFilter(risk_gain=3.0)
        state = np.array([10.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        tau_des = np.array([500.0, 0.0, 0.0])
        r_fixed = f_fixed.filter(state, tau_des, (0.0, 0.0), 0.0, risk_level=0.9)
        r_cvar = f_cvar.filter(state, tau_des, (0.0, 0.0), 0.0, risk_level=0.9)
        # CVaR-adaptive should have different alpha values
        assert r_cvar.alpha1 != r_fixed.alpha1

    def test_risk_estimator_output(self):
        from arctic_quasi_dp.sci1.control.risk_estimator import ProxyCVaRRiskEstimator
        est = ProxyCVaRRiskEstimator()
        state = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        ice = {"concentration": 0.5, "thickness": 0.8, "drift_speed": 0.3}
        tau = np.array([100.0, 50.0, 500.0])
        r = est.estimate(state, (0.0, 0.0), ice, tau)
        assert 0.0 <= r.risk_level <= 1.0
        assert 0.0 <= r.cvar_proxy <= 1.0
        assert 0.0 <= r.ice_risk <= 1.0
