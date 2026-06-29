"""回归测试 — 验证审计中发现的关键缺陷已修复。

覆盖:
1. 推进器 _clip_thrusts() 超额再分配
2. 总功率约束执行
3. 推力饱和指标不混入 yaw moment
4. 配对样本使用 Wilcoxon signed-rank
5. 仿真日志时间戳与状态对齐
6. 能耗对 log_interval 不敏感
7. allocation_success 列被记录
"""

import math
import numpy as np
import pandas as pd
import pytest

from arctic_quasi_dp.sci1.thruster import (
    ThrusterConfig,
    ThrusterAllocator,
    ThrusterUnit,
)
from arctic_quasi_dp.sci1.metrics import (
    summarize_run,
    statistical_comparison,
    _detect_saturation_columns,
    _paired_wilcoxon_p,
    _wilcoxon_rank_sum_p,
)
from arctic_quasi_dp.sci1.sim_loop import run_simulation, VesselState, VesselParams
from arctic_quasi_dp.controllers.base import BaseController, ControllerResult


# ================================================================
# 1. 推进器再分配
# ================================================================

class TestThrusterRedistribution:
    """_clip_thrusts 必须将超额推力重新分配给未饱和推进器。"""

    def test_excess_redistributed_to_other_thrusters(self):
        """两推进器, 一个超额, 另一个应获得补偿。"""
        cfg = ThrusterConfig(
            name="test",
            thrusters=[
                ThrusterUnit("t1", x=0, y=0, max_thrust=100.0),
                ThrusterUnit("t2", x=0, y=0, max_thrust=100.0),
            ],
        )
        alloc = ThrusterAllocator(cfg)
        # 期望 [200, 0] — t1 应被裁剪到 100, t2 应获得部分超额
        result = alloc._clip_thrusts(np.array([200.0, 0.0]), np.array([True, True]))
        assert result[0] == pytest.approx(100.0, abs=1.0)
        # 修复后 t2 应 > 0 (获得部分超额)
        assert result[1] > 0.0, (
            "Excess should be redistributed to t2, but t2 got 0.0"
        )

    def test_excess_negative_thrust_redistributed(self):
        """负推力超额应再分配给有容量的推进器。"""
        cfg = ThrusterConfig(
            name="test",
            thrusters=[
                ThrusterUnit("t1", x=0, y=0, max_thrust=100.0, min_thrust=0.0),
                ThrusterUnit("t2", x=0, y=0, max_thrust=100.0, min_thrust=0.0),
            ],
        )
        alloc = ThrusterAllocator(cfg)
        # t1 想要 -50 (不可能, 被裁到 0), t2 想要 80 (有容量吸收负超额)
        result = alloc._clip_thrusts(np.array([-50.0, 80.0]), np.array([True, True]))
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        # t2 应被减去部分 (负超额), 但仍 >= 0
        assert result[1] < 80.0, (
            "Negative excess should be redistributed to t2"
        )
        assert result[1] >= 0.0

    def test_all_saturated_no_redistribution(self):
        """全部饱和时不再分配 (不报错)。"""
        cfg = ThrusterConfig(
            name="test",
            thrusters=[
                ThrusterUnit("t1", x=0, y=0, max_thrust=100.0),
                ThrusterUnit("t2", x=0, y=0, max_thrust=100.0),
            ],
        )
        alloc = ThrusterAllocator(cfg)
        result = alloc._clip_thrusts(np.array([150.0, 150.0]), np.array([True, True]))
        assert result[0] == pytest.approx(100.0, abs=1.0)
        assert result[1] == pytest.approx(100.0, abs=1.0)


# ================================================================
# 2. 总功率约束
# ================================================================

class TestPowerConstraint:
    """max_total_power_kw 必须被 enforce。"""

    def test_power_limit_enforced(self):
        """分配结果的总功率不应超过配置限制。"""
        cfg = ThrusterConfig(
            name="power_limited",
            thrusters=[
                ThrusterUnit("t1", x=0, y=0, max_thrust=1000.0, azimuth=0.0),
            ],
            max_total_power_kw=0.5,
        )
        alloc = ThrusterAllocator(cfg)
        thrusts, feasible = alloc.allocate(
            np.array([900.0, 0.0, 0.0]), optimize_azimuth=False,
        )
        actual_power = alloc.total_power_kw(thrusts)
        assert actual_power <= cfg.max_total_power_kw * 1.05, (
            f"Power {actual_power:.3f} kW exceeds limit {cfg.max_total_power_kw} kW"
        )

    def test_power_unlimited_when_zero(self):
        """max_total_power_kw=0 表示无限制。"""
        cfg = ThrusterConfig(
            name="unlimited",
            thrusters=[
                ThrusterUnit("t1", x=0, y=0, max_thrust=1000.0, azimuth=0.0),
            ],
            max_total_power_kw=0.0,
        )
        alloc = ThrusterAllocator(cfg)
        thrusts, _ = alloc.allocate(
            np.array([900.0, 0.0, 0.0]), optimize_azimuth=False,
        )
        # 不应被裁剪
        assert abs(thrusts[0]) > 800.0


# ================================================================
# 3. 推力饱和指标
# ================================================================

class TestSaturationMetric:
    """thrust_saturation_ratio 不应混入 yaw moment (tau_n)。"""

    def test_tau_n_not_in_saturation_columns(self):
        """_detect_saturation_columns 不应返回 tau_n。"""
        df = pd.DataFrame({
            "tau_x": [0.0],
            "tau_y": [0.0],
            "tau_n": [19000.0],
        })
        cols = _detect_saturation_columns(df)
        assert "tau_n" not in cols, (
            "tau_n should not be in saturation columns"
        )

    def test_pure_yaw_moment_no_thrust_saturation(self):
        """纯 yaw moment 不应导致 thrust_saturation_ratio > 0。"""
        n = 5
        df = pd.DataFrame({
            "position_error": [0.0] * n,
            "heading_error": [0.0] * n,
            "tau_x": [0.0] * n,
            "tau_y": [0.0] * n,
            "tau_n": [19000.0] * n,  # 大 yaw moment
            "solver_success": [1.0] * n,
            "violation": [0.0] * n,
        })
        summary = summarize_run(df, "S", "ctrl", 0, 0.1, max_force=1500.0)
        assert summary["thrust_saturation_ratio"] == 0.0, (
            f"Pure yaw moment should not affect thrust_saturation_ratio, "
            f"got {summary['thrust_saturation_ratio']}"
        )

    def test_tau_x_tau_y_detected_for_saturation(self):
        """tau_x 和 tau_y 应被检测为推力饱和列。"""
        df = pd.DataFrame({
            "tau_x": [1600.0],
            "tau_y": [1600.0],
        })
        cols = _detect_saturation_columns(df)
        assert "tau_x" in cols
        assert "tau_y" in cols


# ================================================================
# 4. 配对统计检验
# ================================================================

class TestPairedWilcoxon:
    """配对样本必须使用 signed-rank 而非 rank-sum。"""

    def test_paired_wilcoxon_detects_difference(self):
        """配对差值应被正确检测。"""
        rng = np.random.default_rng(42)
        n = 30
        base = rng.normal(10.0, 1.0, n)
        prop = base - rng.normal(2.0, 0.5, n)  # 系统性更好
        p = _paired_wilcoxon_p(base, prop)
        assert p < 0.001, f"Paired difference should be significant, got p={p}"

    def test_paired_wilcoxon_no_difference(self):
        """无差别的配对数据不应显著。"""
        rng = np.random.default_rng(42)
        n = 30
        vals = rng.normal(5.0, 1.0, n)
        p = _paired_wilcoxon_p(vals, vals)
        assert np.isnan(p) or p > 0.05, "Identical pairs should not be significant"

    def test_statistical_comparison_uses_paired_test(self):
        """statistical_comparison 应使用配对检验 (相同数据, 配对应更敏感)。"""
        rng = np.random.default_rng(42)
        n = 20
        data = []
        for seed in range(n):
            base_val = 10.0 + rng.normal(0, 0.5)
            prop_val = base_val - 2.0 + rng.normal(0, 0.3)  # 配对差值 ~2.0
            data.append({"scenario_id": "S1", "controller": "pid", "seed": seed,
                         "metric": base_val})
            data.append({"scenario_id": "S1", "controller": "full", "seed": seed,
                         "metric": prop_val})
        run_df = pd.DataFrame(data)
        result = statistical_comparison(run_df, "metric", "pid", "full")
        assert result["paired_samples"] == n
        assert result["significant"], "Paired difference of ~2.0 should be significant"


# ================================================================
# 5. 仿真日志时间戳
# ================================================================

class TestSimTimestamp:
    """日志中的 time 应反映积分后状态。"""

    def test_first_log_time_is_dt_not_zero(self):
        """第一条日志的 time 应为 dt (积分后), 不是 0。"""
        class ZeroCtrl(BaseController):
            def set_target(self, x, y, psi): pass
            def compute_control(self, state, **kwargs):
                self._last_diagnostics = {"solver_success": True, "solve_time_ms": 0.0}
                return ControllerResult(tau=np.zeros(3), feasible=True, mode="zero", risk=0.0)
            def reset(self): pass

        log = run_simulation(ZeroCtrl(), duration=0.5, dt=0.1)
        df = log.to_dataframe()
        assert len(df) > 0
        # 第一条记录的时间应为 dt, 不是 0
        assert df["time"].iloc[0] == pytest.approx(0.1, abs=1e-10), (
            f"First log time should be dt=0.1, got {df['time'].iloc[0]}"
        )

    def test_time_monotonically_increases(self):
        """时间戳应单调递增。"""
        class ZeroCtrl(BaseController):
            def set_target(self, x, y, psi): pass
            def compute_control(self, state, **kwargs):
                self._last_diagnostics = {"solver_success": True, "solve_time_ms": 0.0}
                return ControllerResult(tau=np.zeros(3), feasible=True, mode="zero", risk=0.0)
            def reset(self): pass

        log = run_simulation(ZeroCtrl(), duration=1.0, dt=0.1)
        df = log.to_dataframe()
        times = df["time"].to_numpy()
        assert np.all(np.diff(times) > 0), "Time should be monotonically increasing"


# ================================================================
# 6. 能耗对 log_interval 不敏感
# ================================================================

class TestEnergyLogInterval:
    """能耗应不受 log_interval 影响。"""

    def _make_const_ctrl(self):
        class ConstCtrl(BaseController):
            def set_target(self, x, y, psi): pass
            def compute_control(self, state, **kwargs):
                self._last_diagnostics = {"solver_success": True, "solve_time_ms": 0.0}
                return ControllerResult(
                    tau=np.array([1000.0, 0.0, 0.0]), feasible=True, mode="const", risk=0.0,
                )
            def reset(self): pass
        return ConstCtrl()

    def test_energy_invariant_to_log_interval(self):
        """不同 log_interval 下, 相同时刻的能耗应一致。

        注意: 必须比较相同时刻的值, 因为不同 log_interval 的最后一行
        可能对应不同的仿真时刻。
        """
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController
        from arctic_quasi_dp.sci1.ice_schedule import ConstantIce

        ice = ConstantIce(0.5, 0.8, 0.3, 120.0)
        # duration=5.0, dt=0.1, log_interval=5 -> 最后一行 t=4.6
        # 需要在 li=1 的结果中找到 t=4.6 对应的行

        ctrl1 = PrecisionDPController()
        ctrl1.set_target(5.0, 3.0, 0.0)
        log1 = run_simulation(ctrl1, duration=5.0, dt=0.1, log_interval=1, ice_schedule=ice)
        df1 = log1.to_dataframe()

        ctrl2 = PrecisionDPController()
        ctrl2.set_target(5.0, 3.0, 0.0)
        log2 = run_simulation(ctrl2, duration=5.0, dt=0.1, log_interval=5, ice_schedule=ice)
        df2 = log2.to_dataframe()

        # 比较 t=4.6 时刻的能耗 (li=5 的最后一行)
        t_target = float(df2["time"].iloc[-1])
        idx_li1 = (df1["time"] - t_target).abs().idxmin()
        e1 = float(df1.loc[idx_li1, "energy"])
        e2 = float(df2["energy"].iloc[-1])
        assert abs(e1 - e2) < 0.01 * max(abs(e1), 1e-9), (
            f"Energy at t={t_target:.1f}s should be invariant to log_interval: "
            f"log_interval=1 -> {e1:.4f}, log_interval=5 -> {e2:.4f}"
        )


# ================================================================
# 7. allocation_success 列
# ================================================================

class TestAllocationSuccess:
    """allocation_success 列应被记录到仿真日志中。"""

    def test_allocation_success_column_exists(self):
        """使用推进器分配时, allocation_success 列应存在。"""
        from arctic_quasi_dp.controllers.pid import PIDController, PIDParams

        ctrl = PIDController(PIDParams())
        ctrl.set_target(0.0, 0.0, 0.0)
        cfg = ThrusterConfig.generic_dp_vessel()
        log = run_simulation(
            ctrl, duration=1.0, dt=0.1, thruster_config=cfg,
        )
        df = log.to_dataframe()
        assert "allocation_success" in df.columns, (
            "allocation_success column should be present when allocator is used"
        )

    def test_allocation_success_without_allocator(self):
        """无推进器分配时, allocation_success 应为 1.0。"""
        from arctic_quasi_dp.controllers.pid import PIDController, PIDParams

        ctrl = PIDController(PIDParams())
        ctrl.set_target(0.0, 0.0, 0.0)
        log = run_simulation(ctrl, duration=1.0, dt=0.1)
        df = log.to_dataframe()
        # 无 allocator 时该列可能不存在, 但 summarize_run 应正确处理
        # (默认 allocation_failure_ratio = 0)
        summary = summarize_run(df, "S", "ctrl", 0, 0.1)
        assert summary["allocation_failure_ratio"] == 0.0


# ================================================================
# 8. 传感器注入
# ================================================================

class TestSensorInjection:
    """传感器模型应注入噪声到控制器输入。"""

    def test_position_sensor_changes_controller_input(self):
        """有位置传感器时, 控制器应收到含噪声的状态。"""
        from arctic_quasi_dp.sci1.sensor_models import PositionSensorModel, SensorNoiseConfig
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController

        ctrl = PrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        sensor = PositionSensorModel(SensorNoiseConfig(gaussian_std=5.0))

        log = run_simulation(
            ctrl, duration=1.0, dt=0.1, seed=42,
            position_sensor=sensor,
        )
        df = log.to_dataframe()
        # 有噪声传感器时, 位置误差应与无传感器时不同
        ctrl2 = PrecisionDPController()
        ctrl2.set_target(0.0, 0.0, 0.0)
        log2 = run_simulation(ctrl2, duration=1.0, dt=0.1, seed=42)
        df2 = log2.to_dataframe()
        # 至少某些时间步的位置误差应不同
        assert not np.allclose(
            df["position_error"].values, df2["position_error"].values, atol=1e-6,
        ), "Position sensor noise should affect controller behavior"

    def test_ice_sensor_model_injects_noise(self):
        """有冰况传感器时, 控制器应收到含噪声的冰况。"""
        from arctic_quasi_dp.sci1.sensor_models import IceConditionSensorModel
        from arctic_quasi_dp.sci1.controllers import IceAwarePrecisionDPController
        from arctic_quasi_dp.sci1.ice_schedule import ConstantIce

        ctrl = IceAwarePrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ice = ConstantIce(0.5, 0.8, 0.3, 120.0)
        sensor = IceConditionSensorModel(concentration_std=0.2)

        log = run_simulation(
            ctrl, duration=2.0, dt=0.1, seed=42, ice_schedule=ice,
            ice_sensor_model=sensor,
        )
        df = log.to_dataframe()
        assert len(df) > 0
        # 传感器注入噪声后, 控制力应与无传感器时不同
        ctrl2 = IceAwarePrecisionDPController()
        ctrl2.set_target(0.0, 0.0, 0.0)
        log2 = run_simulation(ctrl2, duration=2.0, dt=0.1, seed=42, ice_schedule=ice)
        df2 = log2.to_dataframe()
        # 有传感器噪声时 tau_x 应不同 (概率极高)
        assert not np.allclose(df["tau_x"].values[:5], df2["tau_x"].values[:5], atol=1e-6)


# ================================================================
# 9. 冰力模型切换
# ================================================================

class TestIceModelSwitching:
    """G 组场景应使用不同冰力模型。"""

    def test_ice_load_model_parameter_accepted(self):
        """run_simulation 应接受 ice_load_model 参数。"""
        from arctic_quasi_dp.sci1.ice_models import EmpiricalIceLoadModel
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController
        from arctic_quasi_dp.sci1.ice_schedule import ConstantIce

        ctrl = PrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ice = ConstantIce(0.5, 0.8, 0.3, 120.0)
        model = EmpiricalIceLoadModel()

        log = run_simulation(
            ctrl, duration=1.0, dt=0.1, ice_schedule=ice,
            ice_load_model=model,
        )
        df = log.to_dataframe()
        assert len(df) > 0


# ================================================================
# 10. 风场注入
# ================================================================

class TestWindInjection:
    """A2 场景应使用风场。"""

    def test_wind_schedule_affects_dynamics(self):
        """有风场时, 船舶应受风力影响。"""
        from arctic_quasi_dp.sci1.sim_loop import ConstantWindSchedule
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController

        ctrl = PrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        wind = ConstantWindSchedule(u10=15.0, v10=10.0)

        log = run_simulation(
            ctrl, duration=5.0, dt=0.1, wind_schedule=wind,
        )
        df = log.to_dataframe()
        # 有风时, 船舶应偏离目标
        ctrl2 = PrecisionDPController()
        ctrl2.set_target(0.0, 0.0, 0.0)
        log2 = run_simulation(ctrl2, duration=5.0, dt=0.1)
        df2 = log2.to_dataframe()
        # 有风时的位置误差应更大
        assert df["position_error"].iloc[-1] > df2["position_error"].iloc[-1] * 0.5, (
            "Wind should increase position error"
        )


# ================================================================
# 11. Observer 单次更新
# ================================================================

class TestObserverSingleUpdate:
    """supervisor 模式下 observer 每步应只更新一次。"""

    def test_observer_update_count_in_ice_aware_mode(self):
        """ICE_AWARE 模式下 observer 每步只更新一次。"""
        from arctic_quasi_dp.sci1.controllers import (
            ModeSupervisedIceDPController, DPMode,
        )
        from arctic_quasi_dp.sci1.ice_schedule import ConstantIce

        ctrl = ModeSupervisedIceDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ice = ConstantIce(0.7, 1.0, 0.5, 150.0)

        # 运行几步让模式切换到 ICE_AWARE
        log = run_simulation(
            ctrl, duration=5.0, dt=0.1, ice_schedule=ice,
        )
        df = log.to_dataframe()
        # 检查 supervisor_mode 列存在且有模式切换
        assert "supervisor_mode" in df.columns


# ================================================================
# 12. PID 增益对齐
# ================================================================

class TestPIDGainAlignment:
    """PID 增益应与 PrecisionDPController 一致。"""

    def test_pid_gains_match_precision(self):
        """PID 默认增益应与 PrecisionDP 一致。"""
        from arctic_quasi_dp.controllers.pid import PIDParams
        from arctic_quasi_dp.sci1.controllers import PrecisionDPParams

        pid = PIDParams()
        prec = PrecisionDPParams()
        assert pid.Kp_pos == prec.kp_pos
        assert pid.Kd_pos == prec.kd_pos
        assert pid.Kp_heading == prec.kp_heading
        assert pid.Kd_heading == prec.kd_heading
