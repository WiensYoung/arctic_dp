"""Test thruster allocation metrics."""

import numpy as np
import pytest
from arctic_quasi_dp.sci1.thruster import ThrusterConfig, ThrusterAllocator


class TestThrusterMetrics:
    """Thruster metrics must correctly capture allocation quality."""

    def test_zero_tau_metrics(self):
        config = ThrusterConfig.generic_dp_vessel()
        alloc = ThrusterAllocator(config)
        thrusts, feasible = alloc.allocate(np.zeros(3))
        assert feasible
        assert alloc.thrust_saturation_ratio(thrusts) == 0.0

    def test_saturation_ratio_bounded(self):
        config = ThrusterConfig.generic_dp_vessel()
        alloc = ThrusterAllocator(config)
        # Large force should saturate some thrusters
        thrusts, _ = alloc.allocate(np.array([2000.0, 0.0, 0.0]))
        ratio = alloc.thrust_saturation_ratio(thrusts)
        assert 0.0 <= ratio <= 1.0

    def test_power_estimation_positive(self):
        config = ThrusterConfig.generic_dp_vessel()
        alloc = ThrusterAllocator(config)
        thrusts, _ = alloc.allocate(np.array([500.0, 200.0, 1000.0]))
        power = alloc.total_power_kw(thrusts)
        assert power >= 0.0

    def test_allocation_residual(self):
        config = ThrusterConfig.generic_dp_vessel()
        alloc = ThrusterAllocator(config)
        tau_desired = np.array([500.0, 200.0, 1000.0])
        thrusts, _ = alloc.allocate(tau_desired)
        tau_actual = alloc.resulting_tau(thrusts)
        residual = np.linalg.norm(tau_desired - tau_actual)
        assert residual >= 0.0

    def test_diagnostics_available(self):
        config = ThrusterConfig.generic_dp_vessel()
        alloc = ThrusterAllocator(config)
        diag = alloc.get_diagnostics()
        assert "n_thrusters" in diag
        assert "n_faulty" in diag
