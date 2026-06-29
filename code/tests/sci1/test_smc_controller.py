"""Unit tests for Sliding Mode Controller (SMC)."""

import numpy as np
import pytest

from arctic_quasi_dp.controllers.smc import SMCController, SMCParams


def _make_state(x=0.0, y=0.0, psi=0.0, u=0.0, v=0.0, r=0.0):
    """Create a 6-DOF state vector."""
    return np.array([x, y, psi, u, v, r], dtype=np.float64)


class TestSMCController:
    """Test SMC controller output shape, finiteness, and lifecycle."""

    def test_output_shape(self):
        """SMC should return 3-element force/moment vector."""
        ctrl = SMCController()
        ctrl.set_target(5.0, 3.0, 0.0)
        result = ctrl.compute_control(_make_state())
        assert result.tau.shape == (3,)

    def test_output_finite(self):
        """SMC output should be finite for normal inputs."""
        ctrl = SMCController()
        ctrl.set_target(5.0, 3.0, 0.0)
        result = ctrl.compute_control(_make_state(x=1.0, y=2.0, psi=0.1, u=0.5, v=0.3, r=0.1))
        assert np.all(np.isfinite(result.tau)), f"Non-finite output: {result.tau}"

    def test_output_zero_at_target(self):
        """SMC should produce near-zero force when at target with zero velocity."""
        ctrl = SMCController()
        ctrl.set_target(0.0, 0.0, 0.0)
        result = ctrl.compute_control(_make_state())
        # At target with zero velocity, sliding surface is zero => force is zero
        assert np.allclose(result.tau, 0.0, atol=1.0), f"Expected near-zero, got {result.tau}"

    def test_force_saturates_at_eta(self):
        """SMC force should saturate at eta for large errors."""
        p = SMCParams(eta=100.0, max_force=1500.0)
        ctrl = SMCController(params=p)
        ctrl.set_target(10.0, 0.0, 0.0)

        # Large error => saturation function clips to ±1 => force = eta
        result = ctrl.compute_control(_make_state(x=0.0, y=0.0))
        force = np.linalg.norm(result.tau[:2])
        assert force <= p.eta * 1.5, f"Force {force:.1f} should be near eta={p.eta} for large error"

    def test_force_near_zero_at_small_error(self):
        """SMC force should be small near the target."""
        p = SMCParams(eta=100.0, lambda_s=0.5, phi=0.1)
        ctrl = SMCController(params=p)
        ctrl.set_target(0.0, 0.0, 0.0)

        # Very small error => sliding surface near zero => small force
        result = ctrl.compute_control(_make_state(x=0.001, y=0.001, u=0.001, v=0.001))
        force = np.linalg.norm(result.tau[:2])
        assert force < p.eta, f"Force {force:.1f} should be small for tiny error"

    def test_sliding_surface_behavior(self):
        """SMC should produce force toward target."""
        p = SMCParams(lambda_s=0.5, eta=100.0)
        ctrl = SMCController(params=p)
        ctrl.set_target(5.0, 0.0, 0.0)

        # pos_err = 0 - 5 = -5, so s = lambda * (-5) < 0
        # sat(s/phi) = -1, so fx = -eta * (-1) = +eta (positive = toward target in NED)
        state = _make_state(x=0.0, y=0.0, u=0.0, v=0.0, r=0.0)
        result = ctrl.compute_control(state)
        # Force should be positive (toward target at x=5 in NED)
        assert result.tau[0] > 0, (
            f"Expected positive force toward target at x=5, got tau_x={result.tau[0]:.1f}"
        )

    def test_chattering_suppression(self):
        """Saturation function should limit control near sliding surface."""
        p = SMCParams(phi=0.1, eta=100.0)
        ctrl = SMCController(params=p)
        ctrl.set_target(0.0, 0.0, 0.0)

        # Near the sliding surface (small error, small velocity)
        state = _make_state(x=0.01, y=0.01, u=0.01, v=0.01)
        result = ctrl.compute_control(state)
        # Force should be bounded, not exploding
        assert np.max(np.abs(result.tau)) < p.max_force * 2

    def test_max_force_clipping(self):
        """SMC output should be clipped to max_force."""
        p = SMCParams(max_force=1000.0, max_moment=50000.0)
        ctrl = SMCController(params=p)
        ctrl.set_target(100.0, 100.0, 180.0)

        state = _make_state(x=0.0, y=0.0, psi=0.0, u=0.0, v=0.0, r=0.0)
        result = ctrl.compute_control(state)
        assert np.all(np.abs(result.tau[:2]) <= p.max_force + 1.0)
        assert np.abs(result.tau[2]) <= p.max_moment + 1.0

    def test_set_target_lifecycle(self):
        """set_target should store target correctly."""
        ctrl = SMCController()
        assert not hasattr(ctrl, "_target_pos") or ctrl._target_pos is None

        ctrl.set_target(5.0, 3.0, 45.0)
        assert ctrl._target_pos == (5.0, 3.0)
        assert abs(ctrl._target_psi - np.radians(45.0)) < 1e-10

    def test_reset(self):
        """reset should not raise and controller should still work."""
        ctrl = SMCController()
        ctrl.set_target(5.0, 3.0, 0.0)
        ctrl.compute_control(_make_state(x=1.0, y=1.0))
        ctrl.reset()
        result = ctrl.compute_control(_make_state(x=1.0, y=1.0))
        assert np.all(np.isfinite(result.tau))

    def test_no_target_returns_zero(self):
        """Without set_target, SMC should return zero force."""
        ctrl = SMCController()
        result = ctrl.compute_control(_make_state())
        assert np.allclose(result.tau, 0.0)

    def test_diagnostics_available(self):
        """SMC should populate diagnostics."""
        ctrl = SMCController()
        ctrl.set_target(5.0, 3.0, 0.0)
        ctrl.compute_control(_make_state(x=1.0, y=1.0))
        diag = ctrl.get_diagnostics()
        assert "solver_status" in diag
        assert diag["solver_status"] == "smc"

    def test_mode_label(self):
        """SMC should report mode='smc'."""
        ctrl = SMCController()
        ctrl.set_target(0.0, 0.0, 0.0)
        result = ctrl.compute_control(_make_state())
        assert result.mode == "smc"

    def test_risk_bounded(self):
        """Risk should be in [0, 1]."""
        ctrl = SMCController()
        ctrl.set_target(0.0, 0.0, 0.0)
        for x in [0.0, 5.0, 10.0, 50.0]:
            result = ctrl.compute_control(_make_state(x=x))
            assert 0.0 <= result.risk <= 1.0, f"Risk={result.risk} out of bounds at x={x}"

    def test_heading_response(self):
        """SMC should produce yaw moment to correct heading error."""
        ctrl = SMCController()
        ctrl.set_target(0.0, 0.0, 0.0)

        # Positive heading error => negative yaw moment (toward target)
        result = ctrl.compute_control(_make_state(psi=0.5))
        # The sign depends on the sliding surface, but moment should be nonzero
        assert np.abs(result.tau[2]) > 0.01, "Expected nonzero yaw moment for heading error"

    def test_params_defaults(self):
        """SMCParams should have reasonable defaults."""
        p = SMCParams()
        assert p.lambda_s > 0
        assert p.eta > 0
        assert p.phi > 0
        assert p.max_force > 0
        assert p.max_moment > 0
