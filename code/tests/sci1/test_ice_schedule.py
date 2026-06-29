"""冰况调度器测试。"""

import math
import numpy as np
import pytest

from arctic_quasi_dp.sci1.ice_schedule import (
    IceState,
    ConstantIce,
    LinearRampIce,
    StepIce,
    SinusoidalIce,
    RandomWalkIce,
    PiecewiseIce,
    make_b4_ice_schedule,
    make_step_jump_ice,
    make_oscillating_ice,
)


class TestConstantIce:
    def test_returns_same_state(self):
        ice = ConstantIce(0.5, 0.8, 0.3, 120.0)
        s1 = ice.at(0.0)
        s2 = ice.at(100.0)
        assert s1.concentration == s2.concentration == 0.5
        assert s1.thickness == s2.thickness == 0.8

    def test_callable(self):
        ice = ConstantIce(0.5, 0.8, 0.3, 120.0)
        d = ice(50.0)
        assert d["concentration"] == 0.5


class TestLinearRampIce:
    def test_ramp_start(self):
        ice = LinearRampIce(0.2, 0.5, 0.1, 90.0, 0.8, 1.2, 0.5, 180.0, 100.0)
        s = ice.at(0.0)
        assert abs(s.concentration - 0.2) < 0.01
        assert abs(s.drift_speed - 0.1) < 0.01

    def test_ramp_end(self):
        ice = LinearRampIce(0.2, 0.5, 0.1, 90.0, 0.8, 1.2, 0.5, 180.0, 100.0)
        s = ice.at(100.0)
        assert abs(s.concentration - 0.8) < 0.01
        assert abs(s.drift_speed - 0.5) < 0.01

    def test_ramp_mid(self):
        ice = LinearRampIce(0.2, 0.5, 0.1, 90.0, 0.8, 1.2, 0.5, 180.0, 100.0)
        s = ice.at(50.0)
        assert abs(s.concentration - 0.5) < 0.01

    def test_ramp_clamped(self):
        ice = LinearRampIce(0.2, 0.5, 0.1, 90.0, 0.8, 1.2, 0.5, 180.0, 100.0)
        s = ice.at(200.0)  # 超过 duration
        assert s.concentration <= 1.0
        assert s.concentration >= 0.0


class TestStepIce:
    def test_before_step(self):
        ice = StepIce(0.3, 0.5, 0.2, 90.0, 0.9, 1.5, 0.7, 180.0, t_change=50.0)
        s = ice.at(30.0)
        assert s.concentration == 0.3

    def test_after_step(self):
        ice = StepIce(0.3, 0.5, 0.2, 90.0, 0.9, 1.5, 0.7, 180.0, t_change=50.0)
        s = ice.at(60.0)
        assert s.concentration == 0.9

    def test_at_step(self):
        ice = StepIce(0.3, 0.5, 0.2, 90.0, 0.9, 1.5, 0.7, 180.0, t_change=50.0)
        s = ice.at(50.0)
        assert s.concentration == 0.9  # >= t_change


class TestSinusoidalIce:
    def test_oscillates(self):
        ice = SinusoidalIce(0.5, 0.1, 0.8, 0.1, 0.3, 0.05, 135.0, 10.0, period=60.0)
        s0 = ice.at(0.0)
        s_quarter = ice.at(15.0)  # quarter period: sin(pi/2)=1 → max
        s_three_quarter = ice.at(45.0)  # three-quarter: sin(3pi/2)=-1 → min
        # 浓度应在 0.4-0.6 之间振荡
        assert 0.3 < s0.concentration < 0.7
        # quarter 和 three-quarter 应该不同 (一个在均值之上，一个在均值之下)
        assert abs(s_quarter.concentration - s_three_quarter.concentration) > 0.05

    def test_bounded(self):
        ice = SinusoidalIce(0.5, 0.3, 0.8, 0.5, 0.3, 0.2, 135.0, 50.0, period=60.0)
        for t in range(0, 300, 10):
            s = ice.at(float(t))
            assert 0.0 <= s.concentration <= 1.0
            assert s.thickness >= 0.0
            assert s.drift_speed >= 0.0


class TestRandomWalkIce:
    def test_deterministic(self):
        ice1 = RandomWalkIce(0.5, 0.8, 0.3, 120.0, seed=42)
        ice2 = RandomWalkIce(0.5, 0.8, 0.3, 120.0, seed=42)
        for t in [0.0, 10.0, 50.0, 100.0]:
            s1 = ice1.at(t)
            s2 = ice2.at(t)
            assert abs(s1.concentration - s2.concentration) < 1e-10

    def test_different_seeds(self):
        ice1 = RandomWalkIce(0.5, 0.8, 0.3, 120.0, seed=42)
        ice2 = RandomWalkIce(0.5, 0.8, 0.3, 120.0, seed=99)
        # 在足够长的时间后应该不同
        s1 = ice1.at(200.0)
        s2 = ice2.at(200.0)
        # 不一定不同 (随机), 但大概率不同
        # 这里只检查不崩溃
        assert s1.concentration >= 0.0
        assert s2.concentration >= 0.0


class TestPiecewiseIce:
    def test_switches_schedule(self):
        seg1 = ConstantIce(0.3, 0.5, 0.2, 90.0)
        seg2 = ConstantIce(0.8, 1.0, 0.5, 180.0)
        ice = PiecewiseIce([(0.0, seg1), (50.0, seg2)])
        assert ice.at(30.0).concentration == 0.3
        assert ice.at(60.0).concentration == 0.8


class TestPredefined:
    def test_b4_schedule(self):
        ice = make_b4_ice_schedule(300.0)
        s0 = ice.at(0.0)
        s1 = ice.at(300.0)
        assert s0.concentration < s1.concentration

    def test_step_jump(self):
        ice = make_step_jump_ice(300.0)
        s0 = ice.at(0.0)
        s1 = ice.at(200.0)
        assert s0.concentration < s1.concentration

    def test_oscillating(self):
        ice = make_oscillating_ice(300.0)
        s = ice.at(50.0)
        assert 0.0 <= s.concentration <= 1.0
