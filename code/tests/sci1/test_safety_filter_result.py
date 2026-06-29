"""SafetyFilterResult and SoftHOCBFSafetyFilter tests."""

import math
import numpy as np
import pytest


class TestSafetyFilterResult:
    """SafetyFilterResult 字段完整性。"""

    def test_result_fields_complete(self):
        from arctic_quasi_dp.sci1.control.safety_filter import SafetyFilterResult
        tau = np.array([100.0, 50.0, 500.0])
        r = SafetyFilterResult(tau_des=tau, tau_safe=tau)
        assert r.tau_des.shape == (3,)
        assert r.tau_safe.shape == (3,)
        assert r.active is False
        assert r.qp_success is True
        assert r.correction_norm == 0.0

    def test_disabled_filter_no_change(self):
        from arctic_quasi_dp.sci1.control.safety_filter import DisabledSafetyFilter
        f = DisabledSafetyFilter()
        tau = np.array([100.0, 50.0, 500.0])
        result = f.filter(
            state=np.zeros(6), tau_des=tau,
            target_pos=(0.0, 0.0), target_psi=0.0,
        )
        np.testing.assert_allclose(result.tau_safe, tau)
        assert result.active is False
        assert result.qp_success is True
        assert result.correction_norm == 0.0

    def test_result_to_dict(self):
        from arctic_quasi_dp.sci1.control.safety_filter import SafetyFilterResult
        tau = np.array([100.0, 50.0, 500.0])
        r = SafetyFilterResult(tau_des=tau, tau_safe=tau)
        d = r.to_dict()
        assert "tau_des_x" in d
        assert "tau_safe_x" in d
        assert "safety_filter_active" in d


class TestSoftHOCBFSafetyFilter:
    """Box-constrained Soft-HOCBF-QP safety filter。"""

    def test_qp_solves_normally(self):
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        f = SoftHOCBFSafetyFilter()
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau_des = np.array([100.0, 50.0, 500.0])
        result = f.filter(state, tau_des, (0.0, 0.0), 0.0)
        assert result.qp_success is True
        assert result.tau_safe.shape == (3,)

    def test_tau_safe_respects_box_constraints(self):
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        f = SoftHOCBFSafetyFilter(max_force_x=500.0, max_force_y=500.0, max_moment_n=5000.0)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau_des = np.array([9999.0, 9999.0, 99999.0])
        result = f.filter(state, tau_des, (0.0, 0.0), 0.0)
        assert result.qp_success is True
        assert abs(result.tau_safe[0]) <= 500.0 + 1.0
        assert abs(result.tau_safe[1]) <= 500.0 + 1.0
        assert abs(result.tau_safe[2]) <= 5000.0 + 1.0

    def test_near_target_no_correction(self):
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        f = SoftHOCBFSafetyFilter()
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau_des = np.array([10.0, 5.0, 50.0])
        result = f.filter(state, tau_des, (0.0, 0.0), 0.0)
        # Near target, HOCBF should not be very active
        assert result.correction_norm < np.linalg.norm(tau_des) * 0.5

    def test_slack_avoids_infeasibility(self):
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        f = SoftHOCBFSafetyFilter(max_force_x=100.0, max_force_y=100.0, max_moment_n=1000.0)
        # State far from target with large tau_des
        state = np.array([50.0, 50.0, 0.0, 0.0, 0.0, 0.0])
        tau_des = np.array([99.0, 99.0, 999.0])
        result = f.filter(state, tau_des, (0.0, 0.0), 0.0)
        # Should not be infeasible thanks to slack
        assert result.infeasible is False

    def test_risk_modulation_changes_output(self):
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        f0 = SoftHOCBFSafetyFilter(risk_gain=0.0)
        f1 = SoftHOCBFSafetyFilter(risk_gain=2.0)
        state = np.array([10.0, 10.0, 0.0, 0.5, 0.0, 0.0])
        tau_des = np.array([500.0, 200.0, 2000.0])
        r0 = f0.filter(state, tau_des, (0.0, 0.0), 0.0, risk_level=0.0)
        r1 = f1.filter(state, tau_des, (0.0, 0.0), 0.0, risk_level=0.8)
        # With risk modulation, alpha values should differ
        assert r1.alpha1 > r0.alpha1


class TestSafetyFilteredController:
    """SafetyFilteredController wrapper。"""

    def test_wrapper_preserves_nominal_when_disabled(self):
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController
        from arctic_quasi_dp.sci1.control.controller_wrappers import make_filtered_controller
        nominal = PrecisionDPController()
        wrapper = make_filtered_controller(nominal, filter_type="disabled")
        wrapper.set_target(0.0, 0.0, 0.0)
        state = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        result = wrapper.compute_control(state, dt=0.1)
        # Should produce valid tau
        assert result.tau.shape == (3,)

    def test_wrapper_fixed_soft_hocbf(self):
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController
        from arctic_quasi_dp.sci1.control.controller_wrappers import make_filtered_controller
        nominal = PrecisionDPController()
        wrapper = make_filtered_controller(nominal, filter_type="fixed_soft_hocbf")
        wrapper.set_target(0.0, 0.0, 0.0)
        state = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        result = wrapper.compute_control(state, dt=0.1)
        assert result.tau.shape == (3,)
        diag = wrapper.get_diagnostics()
        assert "safety_filter_active" in diag

    def test_wrapper_cvar_soft_hocbf(self):
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController
        from arctic_quasi_dp.sci1.control.controller_wrappers import make_filtered_controller
        nominal = PrecisionDPController()
        wrapper = make_filtered_controller(nominal, filter_type="cvar_soft_hocbf", risk_gain=1.0)
        wrapper.set_target(0.0, 0.0, 0.0)
        state = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        result = wrapper.compute_control(state, dt=0.1)
        assert result.tau.shape == (3,)
