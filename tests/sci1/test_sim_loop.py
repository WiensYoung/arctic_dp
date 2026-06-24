"""仿真循环测试。"""

import math
import numpy as np
import pytest

from arctic_quasi_dp.sci1.controllers import (
    PrecisionDPController,
    IceAwarePrecisionDPController,
    ModeSupervisedIceDPController,
)
from arctic_quasi_dp.sci1.ice_schedule import ConstantIce, LinearRampIce
from arctic_quasi_dp.sci1.sim_loop import (
    run_simulation,
    VesselState,
    VesselParams,
    _ice_force_body,
    _dynamics,
    _rk4_step,
)


class TestVesselState:
    def test_to_array(self):
        s = VesselState(1.0, 2.0, 0.5, 0.1, 0.2, 0.01)
        arr = s.to_array()
        assert arr.shape == (6,)
        assert arr[0] == 1.0

    def test_from_array(self):
        arr = np.array([1.0, 2.0, 0.5, 0.1, 0.2, 0.01])
        s = VesselState.from_array(arr)
        assert s.x == 1.0
        assert s.psi == 0.5


class TestIceForce:
    def test_zero_ice_zero_force(self):
        ice = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
        f = _ice_force_body(ice, 0.0, VesselParams())
        np.testing.assert_allclose(f, 0.0, atol=1e-6)

    def test_force_increases_with_ice(self):
        ice_low = {"concentration": 0.1, "thickness": 0.2, "drift_speed": 0.05, "drift_direction": 90.0}
        ice_high = {"concentration": 0.9, "thickness": 1.5, "drift_speed": 0.6, "drift_direction": 90.0}
        f_low = np.linalg.norm(_ice_force_body(ice_low, 0.0, VesselParams()))
        f_high = np.linalg.norm(_ice_force_body(ice_high, 0.0, VesselParams()))
        assert f_high > f_low


class TestDynamics:
    def test_zero_input_stationary(self):
        state = VesselState()
        tau = np.zeros(3)
        tau_ice = np.zeros(3)
        d = _dynamics(state, tau, tau_ice, VesselParams())
        # 无输入无运动
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_force_produces_acceleration(self):
        state = VesselState()
        tau = np.array([1000.0, 0.0, 0.0])
        tau_ice = np.zeros(3)
        d = _dynamics(state, tau, tau_ice, VesselParams())
        assert d[3] > 0  # surge acceleration


class TestRK4:
    def test_stationary_stays_stationary(self):
        state = VesselState()
        new_state = _rk4_step(state, np.zeros(3), np.zeros(3), VesselParams(), 0.1)
        assert abs(new_state.x) < 1e-10
        assert abs(new_state.u) < 1e-10

    def test_force_moves_vessel(self):
        state = VesselState()
        tau = np.array([500.0, 0.0, 0.0])
        new_state = _rk4_step(state, tau, np.zeros(3), VesselParams(), 1.0)
        assert new_state.x > 0  # 应该向前移动
        assert new_state.u > 0


class TestRunSimulation:
    def test_basic_simulation(self):
        ctrl = PrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        log = run_simulation(ctrl, duration=5.0, dt=0.1)
        df = log.to_dataframe()
        assert len(df) > 0
        assert "position_error" in df.columns
        assert "time" in df.columns

    def test_simulation_with_ice(self):
        ctrl = IceAwarePrecisionDPController()
        ice = ConstantIce(0.5, 0.8, 0.3, 120.0)
        log = run_simulation(ctrl, duration=10.0, dt=0.1, ice_schedule=ice)
        df = log.to_dataframe()
        assert len(df) > 0
        assert df["ice_concentration"].iloc[0] > 0

    def test_timevarying_ice(self):
        ctrl = IceAwarePrecisionDPController()
        ice = LinearRampIce(0.2, 0.5, 0.1, 90.0, 0.8, 1.2, 0.5, 180.0, 10.0)
        log = run_simulation(ctrl, duration=10.0, dt=0.1, ice_schedule=ice)
        df = log.to_dataframe()
        # 冰况应随时间变化
        c_start = df["ice_concentration"].iloc[0]
        c_end = df["ice_concentration"].iloc[-1]
        assert c_start < c_end

    def test_energy_accumulates(self):
        ctrl = IceAwarePrecisionDPController()
        ctrl.set_target(5.0, 3.0, 0.0)
        ice = ConstantIce(0.5, 0.8, 0.3, 120.0)
        log = run_simulation(ctrl, duration=10.0, dt=0.1, ice_schedule=ice)
        df = log.to_dataframe()
        assert df["energy"].iloc[-1] > 0

    def test_supervisor_modes_logged(self):
        ctrl = ModeSupervisedIceDPController()
        ice = ConstantIce(0.7, 1.0, 0.5, 150.0)
        log = run_simulation(ctrl, duration=20.0, dt=0.1, ice_schedule=ice)
        df = log.to_dataframe()
        assert "supervisor_mode" in df.columns
