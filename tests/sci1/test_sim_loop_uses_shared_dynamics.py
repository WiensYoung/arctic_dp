"""Test that both simulators use the same dynamics core."""

import numpy as np
import pytest

from arctic_quasi_dp.sci1.sim_loop import (
    compute_dynamics_derivatives, VesselState, VesselParams, _dynamics,
)
from arctic_quasi_dp.simulation.simulator import Simulator


class TestSharedDynamics:
    """Both simulators must produce the same derivatives."""

    def test_sim_loop_dynamics_uses_shared(self):
        """sim_loop._dynamics should use compute_dynamics_derivatives."""
        params = VesselParams()
        state = VesselState(x=0, y=0, psi=0.3, u=2.0, v=1.0, r=0.1)
        tau_ctrl = np.array([500.0, 200.0, 1000.0])
        tau_ice = np.array([100.0, 50.0, 500.0])

        result = _dynamics(state, tau_ctrl, tau_ice, params)
        expected = compute_dynamics_derivatives(
            state.psi, state.u, state.v, state.r,
            tau_ctrl, tau_ice,
            params.mass, params.Izz,
            params.Xu, params.Yv, params.Nr,
            params.Xu_abs, params.Yv_abs, params.Nr_abs,
        )
        np.testing.assert_allclose(result, expected)

    def test_simulator_dynamics_matches_shared(self):
        """Simulator._dynamics should produce same result as shared function."""
        sim = Simulator()
        state = np.array([0.0, 0.0, 0.3, 2.0, 1.0, 0.1])
        tau_ctrl = np.array([500.0, 200.0, 1000.0])
        tau_ice = np.array([100.0, 50.0, 500.0])

        result = sim._dynamics(state, tau_ctrl, tau_ice)
        expected = compute_dynamics_derivatives(
            state[2], state[3], state[4], state[5],
            tau_ctrl, tau_ice,
            sim.mass, sim.Izz,
            sim.Xu, sim.Yv, sim.Nr,
            sim.Xu_abs, sim.Yv_abs, sim.Nr_abs,
        )
        np.testing.assert_allclose(result, expected)

    def test_no_duplicate_dynamics_formula(self):
        """Source files should not contain duplicate dynamics formulas."""
        from pathlib import Path
        sim_src = Path(Simulator.__module__.replace(".", "/") + ".py")
        # The simulator should import from sim_loop, not define its own dynamics
        import arctic_quasi_dp.simulation.simulator as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        # Should NOT contain the old duplicate formula
        assert "self.Xu * u - self.Xu_abs * abs(u) * u" not in src, (
            "Simulator still contains duplicate dynamics formula"
        )
