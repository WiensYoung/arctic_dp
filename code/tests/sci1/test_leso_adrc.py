"""LESO-ADRC controller tests."""

import math
import numpy as np
import pytest


class TestLESOADRC:
    """Test Linear Extended State Observer ADRC."""

    def test_leso_adrc_construction(self):
        from arctic_quasi_dp.sci1.baseline_controllers import LESOADRCController
        ctrl = LESOADRCController()
        assert ctrl._solver_label == "leso_adrc"

    def test_leso_adrc_dimensions(self):
        """LESO should have 3 states per channel."""
        from arctic_quasi_dp.sci1.baseline_controllers import LESOADRCController
        ctrl = LESOADRCController()
        assert len(ctrl._z_x) == 3
        assert len(ctrl._z_y) == 3
        assert len(ctrl._z_psi) == 3

    def test_leso_adrc_set_target(self):
        from arctic_quasi_dp.sci1.baseline_controllers import LESOADRCController
        ctrl = LESOADRCController()
        ctrl.set_target(5.0, 3.0, 45.0)
        assert ctrl._target_pos == (5.0, 3.0)
        assert abs(ctrl._target_psi - math.radians(45.0)) < 1e-10

    def test_leso_adrc_compute_control_finite(self):
        from arctic_quasi_dp.sci1.baseline_controllers import LESOADRCController
        ctrl = LESOADRCController()
        ctrl.set_target(0.0, 0.0, 0.0)
        state = np.array([5.0, 3.0, 0.1, 0.5, 0.2, 0.01])
        result = ctrl.compute_control(state, dt=0.1)
        assert np.all(np.isfinite(result.tau))
        assert result.mode == "leso_adrc"

    def test_leso_adrc_zero_error_zero_force(self):
        """At target with zero velocity, control should be near zero."""
        from arctic_quasi_dp.sci1.baseline_controllers import LESOADRCController
        ctrl = LESOADRCController()
        ctrl.set_target(0.0, 0.0, 0.0)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = ctrl.compute_control(state, dt=0.1)
        # First call initializes observer, second call should produce small forces
        result2 = ctrl.compute_control(state, dt=0.1)
        assert np.linalg.norm(result2.tau) < 100.0  # should be small

    def test_leso_adrc_reset(self):
        from arctic_quasi_dp.sci1.baseline_controllers import LESOADRCController
        ctrl = LESOADRCController()
        ctrl.set_target(0.0, 0.0, 0.0)
        state = np.array([5.0, 3.0, 0.1, 0.5, 0.2, 0.01])
        ctrl.compute_control(state, dt=0.1)
        ctrl.reset()
        assert not ctrl._initialized
        assert np.allclose(ctrl._z_x, 0.0)

    def test_leso_adrc_diagnostics_complete(self):
        from arctic_quasi_dp.sci1.baseline_controllers import LESOADRCController
        ctrl = LESOADRCController()
        ctrl.set_target(0.0, 0.0, 0.0)
        state = np.array([5.0, 3.0, 0.1, 0.5, 0.2, 0.01])
        ctrl.compute_control(state, dt=0.1)
        diag = ctrl.get_diagnostics()
        assert "leso_disturbance_norm" in diag
        assert "leso_omega_o" in diag
        assert "leso_omega_c" in diag

    def test_runner_builds_leso_adrc(self):
        from arctic_quasi_dp.sci1.runner import build_controller
        ctrl = build_controller("leso_adrc")
        assert ctrl._solver_label == "leso_adrc"
