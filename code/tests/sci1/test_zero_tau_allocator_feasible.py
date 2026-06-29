"""Test that zero desired tau is always feasible."""

import numpy as np
import pytest

from arctic_quasi_dp.sci1.thruster import ThrusterConfig, ThrusterAllocator


class TestZeroTauFeasible:
    """Zero tau_cmd must be feasible (not infeasible)."""

    def test_zero_tau_feasible_generic(self):
        config = ThrusterConfig.generic_dp_vessel()
        alloc = ThrusterAllocator(config)
        tau_zero = np.zeros(3)
        thrusts, feasible = alloc.allocate(tau_zero)
        assert feasible, "Zero tau should be feasible"

    def test_zero_tau_feasible_xuelong2(self):
        config = ThrusterConfig.vessel_xuelong2()
        alloc = ThrusterAllocator(config)
        tau_zero = np.zeros(3)
        thrusts, feasible = alloc.allocate(tau_zero)
        assert feasible, "Zero tau should be feasible"

    def test_zero_tau_produces_zero_thrust(self):
        config = ThrusterConfig.generic_dp_vessel()
        alloc = ThrusterAllocator(config)
        tau_zero = np.zeros(3)
        thrusts, _ = alloc.allocate(tau_zero)
        np.testing.assert_allclose(thrusts, 0.0, atol=1e-10)

    def test_zero_tau_produces_zero_actual_tau(self):
        config = ThrusterConfig.generic_dp_vessel()
        alloc = ThrusterAllocator(config)
        tau_zero = np.zeros(3)
        thrusts, _ = alloc.allocate(tau_zero)
        actual = alloc.resulting_tau(thrusts)
        np.testing.assert_allclose(actual, 0.0, atol=1e-10)
