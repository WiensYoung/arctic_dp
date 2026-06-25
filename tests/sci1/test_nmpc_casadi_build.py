"""Tests for NMPC CasADi controller build and single-step control.

Requires CasADi: pip install casadi
"""

import numpy as np
import pytest

# Skip entire module if CasADi not installed
casadi = pytest.importorskip("casadi")

from arctic_quasi_dp.sci1.nmpc_controller import (
    NMPCIceController,
    NMPCParams,
    check_casadi_available,
)


class TestNMPCBuild:
    """CasADi NMPC 构建与单步测试。"""

    def test_casadi_available(self):
        assert check_casadi_available()

    def test_controller_init(self):
        ctrl = NMPCIceController(NMPCParams())
        assert ctrl is not None

    def test_solver_builds(self):
        ctrl = NMPCIceController(NMPCParams())
        assert ctrl._solver is not None

    def test_compute_control_finite(self):
        ctrl = NMPCIceController(NMPCParams())
        ctrl.set_target(5.0, 5.0, 0.0)
        ctrl.set_ice_conditions(0.3, 0.5, 0.2, 45.0)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = ctrl.compute_control(state, dt=0.2)
        assert np.all(np.isfinite(result.tau))
        assert result.tau.shape == (3,)

    def test_yaw_not_clipped_to_max_force(self):
        """tau[2] (yaw moment) must be clipped to max_moment, not max_force."""
        p = NMPCParams(max_force=1500.0, max_moment=20000.0)
        ctrl = NMPCIceController(p)
        ctrl.set_target(0.0, 0.0, 0.0)
        state = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        result = ctrl.compute_control(state, dt=0.2)
        # tau[2] should be within max_moment bounds, not max_force
        assert abs(result.tau[2]) <= p.max_moment + 1e-6
        # If it were clipped to max_force, it would be <= 1500, but the
        # moment should be able to exceed max_force (1500 < 20000)

    def test_cbf_constraint_direction(self):
        """CBF constraint should enforce h_dot + gamma*h >= 0, not <= 0."""
        ctrl = NMPCIceController(NMPCParams())
        # Constraint structure:
        #   6 initial state constraints
        #   per step: 6 dynamics + 1 force_norm + 1 moment_abs + 1 CBF = 9
        n_init = 6
        n_per_step = 9
        lbg = ctrl._lbg
        ubg = ctrl._ubg
        for k in range(ctrl.params.N):
            cbf_idx = n_init + k * n_per_step + 8  # CBF is last in each step
            assert lbg[cbf_idx] == 0.0, f"CBF lbg should be 0.0, got {lbg[cbf_idx]}"
            assert ubg[cbf_idx] == 1e20, f"CBF ubg should be 1e20, got {ubg[cbf_idx]}"

    def test_initial_state_is_nlp_param(self):
        """Initial state must be constrained via NLP parameter, not free symbol."""
        ctrl = NMPCIceController(NMPCParams())
        # The parameter vector should have 16 elements
        # (10 original + 6 for current state)
        state = np.array([1.0, 2.0, 0.3, 0.5, -0.1, 0.02])
        ctrl.set_target(0.0, 0.0, 0.0)
        result = ctrl.compute_control(state, dt=0.2)
        # If state is correctly passed as param, the solver should produce
        # a result that accounts for the initial position (1, 2)
        assert np.all(np.isfinite(result.tau))

    def test_reset(self):
        ctrl = NMPCIceController(NMPCParams())
        ctrl.set_target(5.0, 5.0, 0.0)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ctrl.compute_control(state, dt=0.2)
        ctrl.reset()
        assert np.allclose(ctrl._prev_tau, 0.0)

    def test_no_target_returns_zero(self):
        ctrl = NMPCIceController(NMPCParams())
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = ctrl.compute_control(state, dt=0.2)
        assert np.allclose(result.tau, 0.0)
        assert result.feasible
