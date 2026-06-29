"""Test that damping is dissipative (energy decreases with zero control/ice)."""

import math
import numpy as np
import pytest

from arctic_quasi_dp.sci1.sim_loop import VesselState, VesselParams, _dynamics, _rk4_step


class TestDampingPassivity:
    """With zero control and zero ice, kinetic energy must decrease."""

    def test_energy_decreases_zero_control_zero_ice(self):
        """Initial non-zero velocity; tau=0, ice=0; energy must decrease."""
        params = VesselParams()
        dt = 0.1
        n_steps = 200  # 20 seconds

        state = VesselState(x=0.0, y=0.0, psi=0.0, u=2.0, v=1.0, r=0.1)
        tau_zero = np.zeros(3)
        ice_zero = np.zeros(3)

        initial_ke = 0.5 * params.mass * (state.u**2 + state.v**2) + 0.5 * params.Izz * state.r**2

        for _ in range(n_steps):
            state = _rk4_step(state, tau_zero, ice_zero, params, dt)

        final_ke = 0.5 * params.mass * (state.u**2 + state.v**2) + 0.5 * params.Izz * state.r**2

        assert final_ke < initial_ke, (
            f"Kinetic energy should decrease: initial={initial_ke:.1f}, final={final_ke:.1f}"
        )

    def test_velocity_decays_to_zero(self):
        """With zero input, velocities should decay toward zero.

        Note: For a 500t vessel with these damping coefficients, the time
        constant is ~mass/Xu = 1000s.  We run 500s and check that speed
        has decreased significantly (by >50%).
        """
        params = VesselParams()
        dt = 0.1
        n_steps = 5000  # 500 seconds

        state = VesselState(x=0.0, y=0.0, psi=0.0, u=3.0, v=2.0, r=0.5)
        tau_zero = np.zeros(3)
        ice_zero = np.zeros(3)

        initial_speed = math.sqrt(state.u**2 + state.v**2)
        for _ in range(n_steps):
            state = _rk4_step(state, tau_zero, ice_zero, params, dt)

        final_speed = math.sqrt(state.u**2 + state.v**2)
        # Speed should have decreased by at least 50%
        assert final_speed < initial_speed * 0.5, (
            f"Speed should decay significantly: initial={initial_speed:.4f}, "
            f"final={final_speed:.4f}"
        )

    def test_damping_coefficients_positive(self):
        """Damping coefficients must be positive (方案 A convention)."""
        params = VesselParams()
        assert params.Xu > 0, "Xu must be positive"
        assert params.Yv > 0, "Yv must be positive"
        assert params.Nr > 0, "Nr must be positive"
        assert params.Xu_abs > 0, "Xu_abs must be positive"
        assert params.Yv_abs > 0, "Yv_abs must be positive"
        assert params.Nr_abs > 0, "Nr_abs must be positive"

    def test_pure_surge_damping(self):
        """Pure surge motion should dampen."""
        params = VesselParams()
        dt = 0.1
        state = VesselState(u=5.0)
        tau_zero = np.zeros(3)
        ice_zero = np.zeros(3)

        for _ in range(100):
            state = _rk4_step(state, tau_zero, ice_zero, params, dt)

        assert state.u < 5.0, "Surge velocity should decrease"
        assert state.u > 0.0, "Surge velocity should remain positive (no reversal from damping)"

    def test_pure_sway_damping(self):
        """Pure sway motion should dampen."""
        params = VesselParams()
        dt = 0.1
        state = VesselState(v=3.0)
        tau_zero = np.zeros(3)
        ice_zero = np.zeros(3)

        for _ in range(100):
            state = _rk4_step(state, tau_zero, ice_zero, params, dt)

        assert state.v < 3.0, "Sway velocity should decrease"
        assert state.v > 0.0, "Sway velocity should remain positive"
