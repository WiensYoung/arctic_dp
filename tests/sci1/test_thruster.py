"""推进器分配模型测试。"""

import numpy as np
import pytest

from arctic_quasi_dp.sci1.thruster import (
    ThrusterUnit,
    ThrusterConfig,
    ThrusterAllocator,
    ThrusterDegradationProfile,
)


class TestThrusterConfig:
    def test_xuelong2_config(self):
        cfg = ThrusterConfig.vessel_xuelong2()
        assert len(cfg.thrusters) == 4
        assert cfg.name == "xuelong2"

    def test_generic_dp_config(self):
        cfg = ThrusterConfig.generic_dp_vessel()
        assert len(cfg.thrusters) == 5


class TestThrusterAllocator:
    def setup_method(self):
        self.cfg = ThrusterConfig.generic_dp_vessel()
        self.alloc = ThrusterAllocator(self.cfg)

    def test_allocate_produces_thrusts(self):
        tau = np.array([500.0, 0.0, 0.0])
        thrusts, feasible = self.alloc.allocate(tau)
        assert thrusts.shape == (5,)
        assert np.all(np.isfinite(thrusts))

    def test_resulting_tau_matches_direction(self):
        tau = np.array([500.0, 0.0, 0.0])
        thrusts, _ = self.alloc.allocate(tau)
        actual = self.alloc.resulting_tau(thrusts)
        # 实际力应与期望力方向一致
        assert np.sign(actual[0]) == np.sign(tau[0])

    def test_zero_tau_zero_thrusts(self):
        tau = np.array([0.0, 0.0, 0.0])
        thrusts, feasible = self.alloc.allocate(tau)
        np.testing.assert_allclose(thrusts, 0.0, atol=1.0)

    def test_saturation_limits(self):
        tau = np.array([1e6, 0.0, 0.0])  # 极端期望力
        thrusts, feasible = self.alloc.allocate(tau)
        for i, t in enumerate(self.cfg.thrusters):
            assert abs(thrusts[i]) <= t.max_thrust * 1.01  # 允许 1% 误差

    def test_faulty_thruster_zero_thrust(self):
        self.alloc.fault_thruster(0)
        tau = np.array([500.0, 0.0, 0.0])
        thrusts, _ = self.alloc.allocate(tau)
        assert thrusts[0] == 0.0

    def test_degradation_reduces_max(self):
        self.alloc.degrade_thruster(0, 0.5)
        tau = np.array([2000.0, 0.0, 0.0])
        thrusts, _ = self.alloc.allocate(tau)
        assert abs(thrusts[0]) <= self.cfg.thrusters[0].max_thrust * 0.5 * 1.01

    def test_reset_restores(self):
        self.alloc.fault_thruster(0)
        self.alloc.degrade_thruster(1, 0.3)
        self.alloc.reset()
        assert not self.cfg.thrusters[0].faulty
        assert self.cfg.thrusters[1].degraded == 1.0

    def test_power_estimate(self):
        thrusts = np.array([100.0, 200.0, 150.0, 100.0, 50.0])
        power = self.alloc.total_power_kw(thrusts)
        assert power > 0

    def test_saturation_ratio(self):
        thrusts = np.array([t.max_thrust * 0.99 for t in self.cfg.thrusters])
        ratio = self.alloc.thrust_saturation_ratio(thrusts)
        assert ratio == 1.0

    def test_diagnostics(self):
        diag = self.alloc.get_diagnostics()
        assert "n_thrusters" in diag
        assert diag["n_thrusters"] == 5


class TestDegradationProfile:
    def test_no_fault(self):
        profile = ThrusterDegradationProfile.no_fault()
        assert profile.name == "no_fault"

    def test_single_loss(self):
        cfg = ThrusterConfig.generic_dp_vessel()
        alloc = ThrusterAllocator(cfg)
        profile = ThrusterDegradationProfile.single_thruster_loss("fwd_port")
        profile.apply(alloc)
        assert cfg.thrusters[0].faulty

    def test_bow_degradation(self):
        cfg = ThrusterConfig.generic_dp_vessel()
        alloc = ThrusterAllocator(cfg)
        profile = ThrusterDegradationProfile.bow_degradation(0.5)
        profile.apply(alloc)
        assert cfg.thrusters[0].degraded == 0.5
        assert cfg.thrusters[1].degraded == 0.5
