"""自定义仿真循环 — 支持时变冰况、推进器分配和 NMPC。

当父包的 Simulator 不支持 per-step 冰况更新时，使用本模块的仿真循环。

本模块实现：
- 3-DOF 船舶动力学积分 (RK4)
- 每 timestep 更新冰况 (通过 IceSchedule)
- 推进器分配 (通过 ThrusterAllocator)
- 完整的日志记录
- 与父包相同的 DataFrame 输出格式

使用：
    from arctic_quasi_dp.sci1.sim_loop import run_simulation
    from arctic_quasi_dp.sci1.ice_schedule import LinearRampIce
    from arctic_quasi_dp.sci1.controllers import ModeSupervisedIceDPController

    schedule = LinearRampIce(0.3, 0.7, 0.15, 120.0, 0.8, 1.1, 0.55, 160.0, 300.0)
    ctrl = ModeSupervisedIceDPController()
    ctrl.set_target(0.0, 0.0, 0.0)
    df = run_simulation(ctrl, duration=300.0, dt=0.1, ice_schedule=schedule)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import math
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .ice_schedule import IceSchedule, ConstantIce
from .thruster import ThrusterAllocator, ThrusterConfig, ThrusterDegradationProfile
from .logging_config import get_logger as _get_logger

_sim_logger = _get_logger(__name__)

# ---------- 物理常量 ----------
_LEVER_FRACTION = 0.18          # 冰力力臂 = 0.18 * 船长
_RHO_AIR = 1.225                # 空气密度 (kg/m³)
_CD_AIR = 0.8                   # 风阻力系数
_WIND_AREA_BEAM_FACTOR = 10.0   # 迎风面积 = 船宽 × 10m
_WIND_AREA_LENGTH_FACTOR = 10.0 # 侧风面积 = 船长 × 10m
_WIND_MOMENT_ARM_FACTOR = 0.3   # 风力矩臂 = 0.3 × 船长
_ENERGY_SCALE = 0.001           # 能耗简化系数


@dataclass
class WindState:
    """风场状态。"""
    u10: float = 0.0  # 10m 风 u 分量 (m/s)
    v10: float = 0.0  # 10m 风 v 分量 (m/s)

    @property
    def speed(self) -> float:
        """风速标量 (m/s)。"""
        return math.sqrt(self.u10 ** 2 + self.v10 ** 2)

    @property
    def direction_deg(self) -> float:
        """风向 (度, 0-360)。"""
        return math.degrees(math.atan2(self.v10, self.u10)) % 360


class ConstantWindSchedule:
    """常数风场调度器。"""

    def __init__(self, u10: float, v10: float):
        self._state = WindState(u10=u10, v10=v10)

    def at(self, t: float) -> WindState:
        return self._state


def _apply_rate_limits(
    thrusts: NDArray[np.float64],
    prev_thrusts: NDArray[np.float64],
    allocator: ThrusterAllocator,
    dt: float,
    max_thrust_rate: float,
    max_azimuth_rate: float,
) -> NDArray[np.float64]:
    """对推进器推力和方位角施加速率限制。"""
    result = thrusts.copy()
    for i, t in enumerate(allocator.config.thrusters):
        if t.faulty:
            continue
        # 推力速率限制
        if max_thrust_rate > 0:
            max_delta = max_thrust_rate * dt
            delta = np.clip(result[i] - prev_thrusts[i], -max_delta, max_delta)
            result[i] = prev_thrusts[i] + delta
        # 方位角速率限制 (仅对可旋转推进器)
        # 注意: 方位角速率限制已在 ThrusterAllocator.allocate() 内部实现
        # (通过 max_azimuth_rate 参数), 此处不需要重复限制。
        # 这里仅对推力幅度施加速率限制。
    return result


@dataclass
class VesselState:
    """船舶 3-DOF 状态。"""
    x: float = 0.0          # NED x (m)
    y: float = 0.0          # NED y (m)
    psi: float = 0.0        # yaw (rad)
    u: float = 0.0          # surge velocity in body (m/s)
    v: float = 0.0          # sway velocity in body (m/s)
    r: float = 0.0          # yaw rate (rad/s)

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.psi, self.u, self.v, self.r], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "VesselState":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]),
                   float(arr[3]), float(arr[4]), float(arr[5]))


@dataclass
class VesselParams:
    """船舶物理参数。

    注意: ice_crushing_strength_mpa 使用代理值 (0.004 MPa) 而非物理值 (2.0 MPa)，
    使冰力输出 (~1 kN) 与控制器力限和推进器容量匹配。
    结构因子和水线角保持 Lindqvist 1989 文献值。
    """
    mass: float = 500000.0         # kg (雪龙2号 ~14000t, 简化为 500t)
    Izz: float = 5e8               # kg·m²
    Xu: float = 500.0              # surge linear damping (N·s/m)
    Yv: float = 800.0              # sway linear damping (N·s/m)
    Nr: float = 2e5                # yaw linear damping (N·m·s/rad)
    Xu_abs: float = 200.0          # surge quadratic damping (N·s²/m²)
    Yv_abs: float = 300.0          # sway quadratic damping (N·s²/m²)
    Nr_abs: float = 5e4            # yaw quadratic damping (N·m·s²/rad²)
    length: float = 122.5          # vessel length (m)
    beam: float = 22.0             # vessel beam (m)
    ice_crushing_strength_mpa: float = 0.0003   # 代理值: 使中等冰况冰力 ~1 kN (物理值 ~2 MPa)
    ice_structure_factor: float = 0.45         # Lindqvist 1989
    waterline_angle_deg: float = 30.0         # 水线角 (度)


def _ice_force_body(
    ice: Dict[str, float],
    psi: float,
    params: VesselParams,
) -> NDArray[np.float64]:
    """计算冰力 (船体坐标系)。使用共享 Lindqvist 简化模型。"""
    from .ice_force_common import compute_ice_force_body_from_dict
    return compute_ice_force_body_from_dict(
        ice=ice,
        vessel_psi=psi,
        crushing_strength_mpa=params.ice_crushing_strength_mpa,
        vessel_beam_m=params.beam,
        vessel_length_m=params.length,
        structure_factor=params.ice_structure_factor,
        waterline_angle_rad=math.radians(params.waterline_angle_deg),
    )


def compute_dynamics_derivatives(
    psi: float, u: float, v: float, r: float,
    tau_control: NDArray[np.float64],
    tau_ice: NDArray[np.float64],
    mass: float, Izz: float,
    Xu: float, Yv: float, Nr: float,
    Xu_abs: float, Yv_abs: float, Nr_abs: float,
) -> NDArray[np.float64]:
    """计算 3-DOF 状态导数 (公共动力学核心)。

    统一的动力学函数，sim_loop 和 simulator 共用。
    阻尼参数为正值，公式为 (force - Xu*u - Xu_abs*|u|*u) / mass。
    """
    cpsi, spsi = math.cos(psi), math.sin(psi)
    xdot = cpsi * u - spsi * v
    ydot = spsi * u + cpsi * v
    psidot = r

    total_fx = tau_control[0] + tau_ice[0]
    total_fy = tau_control[1] + tau_ice[1]
    total_mz = tau_control[2] + tau_ice[2]

    udot = (total_fx - Xu * u - Xu_abs * abs(u) * u) / mass
    vdot = (total_fy - Yv * v - Yv_abs * abs(v) * v) / mass
    rdot = (total_mz - Nr * r - Nr_abs * abs(r) * r) / Izz

    return np.array([xdot, ydot, psidot, udot, vdot, rdot], dtype=np.float64)


def _dynamics(
    state: VesselState,
    tau_control: NDArray[np.float64],
    tau_ice: NDArray[np.float64],
    params: VesselParams,
) -> NDArray[np.float64]:
    """计算状态导数 (连续时间)。调用公共动力学核心。"""
    return compute_dynamics_derivatives(
        state.psi, state.u, state.v, state.r,
        tau_control, tau_ice,
        params.mass, params.Izz,
        params.Xu, params.Yv, params.Nr,
        params.Xu_abs, params.Yv_abs, params.Nr_abs,
    )


def _rk4_step(
    state: VesselState,
    tau_control: NDArray[np.float64],
    tau_ice: NDArray[np.float64],
    params: VesselParams,
    dt: float,
) -> VesselState:
    """RK4 积分一步。"""
    s = state.to_array()
    k1 = _dynamics(state, tau_control, tau_ice, params)

    s2 = s + 0.5 * dt * k1
    k2 = _dynamics(VesselState.from_array(s2), tau_control, tau_ice, params)

    s3 = s + 0.5 * dt * k2
    k3 = _dynamics(VesselState.from_array(s3), tau_control, tau_ice, params)

    s4 = s + dt * k3
    k4 = _dynamics(VesselState.from_array(s4), tau_control, tau_ice, params)

    s_new = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # 角度归一化
    s_new[2] = (s_new[2] + math.pi) % (2 * math.pi) - math.pi
    return VesselState.from_array(s_new)


@dataclass
class SimLog:
    """仿真日志。"""
    rows: List[Dict[str, float]] = field(default_factory=list)

    def append(self, row: Dict[str, float]) -> None:
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def run_simulation(
    controller,
    duration: float = 300.0,
    dt: float = 0.1,
    ice_schedule: Optional[IceSchedule] = None,
    wind_schedule: Optional[Any] = None,
    vessel_params: Optional[VesselParams] = None,
    thruster_config: Optional[ThrusterConfig] = None,
    degradation_profile: Optional[ThrusterDegradationProfile] = None,
    target_x: float = 0.0,
    target_y: float = 0.0,
    target_psi: float = 0.0,
    safe_region_radius: float = 10.0,
    seed: int = 2026,
    log_interval: int = 1,
    verbose: bool = False,
    # 传感器退化注入 (E 组场景)
    position_sensor: Optional[Any] = None,
    heading_sensor: Optional[Any] = None,
    ice_sensor_model: Optional[Any] = None,
    # 冰力模型选择 (G 组场景)
    ice_load_model: Optional[Any] = None,
    # 扰动观测器 (可选)
    disturbance_observer: Optional[Any] = None,
    # 动态目标 (A5 场景)
    target_x_final: Optional[float] = None,
    target_y_final: Optional[float] = None,
    target_psi_final: Optional[float] = None,
    target_change_time: float = 0.0,
    # 推进器速率限制 (C5/C6 场景)
    max_azimuth_rate: float = 0.0,
    max_thrust_rate: float = 0.0,
) -> SimLog:
    """运行完整仿真。

    Args:
        controller: 控制器 (需有 compute_control, set_target, set_ice_conditions 方法)
        duration: 仿真时长 (s)
        dt: 时间步长 (s)
        ice_schedule: 冰况调度器 (None=常数零冰)
        vessel_params: 船舶参数 (None=默认)
        thruster_config: 推进器配置 (None=不使用推进器分配)
        degradation_profile: 推进器退化场景 (None=无退化)
        target_x, target_y, target_psi: 目标位置/艏向
        safe_region_radius: 安全区域半径 (m)
        seed: 随机种子
        log_interval: 日志记录间隔 (每 N 步记录一次)
        verbose: 是否打印进度
        position_sensor: 位置传感器模型 (None=使用真值)
        heading_sensor: 航向传感器模型 (None=使用真值)
        ice_sensor_model: 冰况传感器模型 (None=使用真值)
        ice_load_model: 冰力模型 (None=使用内置 Lindqvist)
        max_azimuth_rate: 方位角速率限制 (rad/s, 0=无限制)
        max_thrust_rate: 推力速率限制 (N/s, 0=无限制)

    Returns:
        SimLog 对象，可调用 to_dataframe() 转为 DataFrame
    """
    params = vessel_params or VesselParams()

    # 冰况调度
    if ice_schedule is None:
        ice_schedule = ConstantIce(0.0, 0.0, 0.0, 0.0)

    # 推进器分配
    allocator = None
    if thruster_config is not None:
        allocator = ThrusterAllocator(thruster_config, max_azimuth_rate=max_azimuth_rate)
        if degradation_profile is not None:
            degradation_profile.apply(allocator)

    # 初始化
    # 注意: controller.set_target 期望 psi 为角度 (度), 内部会转弧度
    controller.set_target(target_x, target_y, target_psi)
    if hasattr(controller, 'set_safe_region_radius'):
        controller.set_safe_region_radius(safe_region_radius)

    # 仿真内部使用弧度计算 heading error (target_psi 始终为度)
    target_psi_rad = math.radians(target_psi)

    state = VesselState()
    n_steps = int(duration / dt)
    log = SimLog()
    rng = np.random.default_rng(seed)
    cumulative_energy = 0.0
    prev_thrusts: Optional[Any] = None  # 用于速率限制

    for step in range(n_steps):
        t = step * dt

        # 0. 动态目标切换
        if target_x_final is not None and t >= target_change_time:
            _new_x = target_x_final
            _new_y = target_y_final if target_y_final is not None else target_y
            _new_psi = target_psi_final if target_psi_final is not None else target_psi
            if _new_x != target_x or _new_y != target_y or _new_psi != target_psi:
                target_x, target_y, target_psi = _new_x, _new_y, _new_psi
                target_psi_rad = math.radians(target_psi)
                controller.set_target(target_x, target_y, target_psi)

        # 1. 获取当前冰况 (真值)
        ice = ice_schedule.at(t)

        # 1b. 冰况传感器注入 (E 组场景)
        if ice_sensor_model is not None:
            ice_est = ice_sensor_model.update(
                ice.concentration, ice.thickness,
                ice.drift_speed, ice.drift_direction,
                rng, dt,
            )
            ice_for_ctrl = ice_est  # IceEstimate 有 .concentration, .thickness_m 等
        else:
            ice_for_ctrl = ice

        # 2. 更新控制器冰况 (使用传感器观测值或真值)
        if hasattr(controller, 'set_ice_conditions'):
            if ice_sensor_model is not None:
                # IceEstimate.drift_direction_deg 已经是角度 (度)
                # 与 IceState.drift_direction 单位一致
                controller.set_ice_conditions(
                    ice_for_ctrl.concentration, ice_for_ctrl.thickness_m,
                    ice_for_ctrl.drift_speed_mps, ice_for_ctrl.drift_direction_deg,
                )
            else:
                controller.set_ice_conditions(
                    ice.concentration, ice.thickness,
                    ice.drift_speed, ice.drift_direction,
                )

        # 3. 计算控制 (使用传感器观测状态或真值)
        state_arr = state.to_array()
        if position_sensor is not None or heading_sensor is not None:
            noisy_state = state_arr.copy()
            if position_sensor is not None:
                noisy_pos = position_sensor.measure(state_arr[:2].copy(), rng)
                noisy_state[0] = noisy_pos[0]
                noisy_state[1] = noisy_pos[1]
            if heading_sensor is not None:
                noisy_state[2] = heading_sensor.measure(float(state_arr[2]), rng)
            result = controller.compute_control(noisy_state, dt=dt)
        else:
            result = controller.compute_control(state_arr, dt=dt)
        tau_cmd = np.asarray(result.tau, dtype=np.float64).reshape(3,)

        # 4. 推进器分配
        tau_actual = tau_cmd
        thrust_saturation = 0.0
        if allocator is not None:
            thrusts, feasible = allocator.allocate(tau_cmd, dt=dt)
            # 4b. 速率限制 (C5/C6 场景)
            if (max_thrust_rate > 0 or max_azimuth_rate > 0) and prev_thrusts is not None:
                thrusts = _apply_rate_limits(
                    thrusts, prev_thrusts, allocator, dt,
                    max_thrust_rate, max_azimuth_rate,
                )
            prev_thrusts = thrusts.copy()
            tau_actual = allocator.resulting_tau(thrusts)
            thrust_saturation = allocator.thrust_saturation_ratio(thrusts)

        # 4c. 扰动观测器更新 (估计冰力, 不使用真值)
        observer_ice_estimate = np.zeros(3, dtype=np.float64)
        if disturbance_observer is not None:
            observer_ice_estimate = disturbance_observer.update(
                tau_actual, np.array([state.u, state.v, state.r], dtype=np.float64),
                params.mass, params.Izz,
                params.Xu, params.Yv, params.Nr,
                params.Xu_abs, params.Yv_abs, params.Nr_abs,
                dt,
            )
            # 将观测器估计传递给控制器 (如果有接口)
            if hasattr(controller, 'set_observer_estimate'):
                controller.set_observer_estimate(observer_ice_estimate)

        # 5. 计算冰力 (使用真值, 可选冰力模型)
        if ice_load_model is not None:
            # ice_models.compute() 期望弧度; 使用 IceState.drift_direction_rad 属性
            ice_result = ice_load_model.compute(
                state.psi, ice.concentration, ice.thickness,
                ice.drift_speed, ice.drift_direction_rad,
                params.length, params.beam, rng,
            )
            tau_ice = ice_result.force_body
        else:
            tau_ice = _ice_force_body(ice.to_dict(), state.psi, params)

        # 5b. 风场强迫 (如果有)
        tau_wind = np.zeros(3, dtype=np.float64)
        if wind_schedule is not None:
            wind = wind_schedule.at(t)
            # 风力代理: F = 0.5 * rho_air * Cd * A * V^2
            A_front = params.beam * _WIND_AREA_BEAM_FACTOR
            A_side = params.length * _WIND_AREA_LENGTH_FACTOR
            # ERA5 convention: u10=eastward, v10=northward
            # NED convention: North=x, East=y → v_ned = [v10, u10]
            # R_ned2body @ [v10, u10] = [cpsi*v10 + spsi*u10, -spsi*v10 + cpsi*u10]
            cpsi_w, spsi_w = math.cos(state.psi), math.sin(state.psi)
            u_wind_body = cpsi_w * wind.v10 + spsi_w * wind.u10  # v10=north, u10=east
            v_wind_body = -spsi_w * wind.v10 + cpsi_w * wind.u10
            u_wind = u_wind_body - state.u  # 相对风速 (体坐标系)
            v_wind = v_wind_body - state.v
            fx_wind = 0.5 * _RHO_AIR * _CD_AIR * A_front * u_wind * abs(u_wind)
            fy_wind = 0.5 * _RHO_AIR * _CD_AIR * A_side * v_wind * abs(v_wind)
            # 偏航力矩: 横向力 × 纵向力臂 + 纵向力 × 横向力臂
            mz_wind = (fy_wind * params.length * _WIND_MOMENT_ARM_FACTOR
                       + fx_wind * params.beam * 0.25)
            tau_wind = np.array([fx_wind, fy_wind, mz_wind])

        # 6. 积分动力学 (控制力 + 冰力 + 风力)
        state = _rk4_step(state, tau_actual, tau_ice + tau_wind, params, dt)

        # 6b. 累积能耗 (每步计算，不受 log_interval 影响)
        tau_mag = math.sqrt(tau_actual[0] ** 2 + tau_actual[1] ** 2 + (tau_actual[2] / max(params.length, 1.0)) ** 2)
        cumulative_energy += tau_mag * dt * _ENERGY_SCALE

        # 7. 记录日志
        if step % log_interval == 0:
            pos_err = math.sqrt((state.x - target_x) ** 2 + (state.y - target_y) ** 2)
            head_err = abs((state.psi - target_psi_rad + math.pi) % (2 * math.pi) - math.pi)
            dist_from_target = pos_err
            violation = 1.0 if dist_from_target > safe_region_radius else 0.0

            diag = controller.get_diagnostics() if hasattr(controller, 'get_diagnostics') else {}

            # 分配残差和执行器证据
            alloc_residual = 0.0
            power_kw = 0.0
            actuator_trace: Dict[str, float] = {}
            if allocator is not None:
                alloc_residual = float(np.linalg.norm(tau_cmd - tau_actual))
                power_kw = allocator.total_power_kw(thrusts)
                actuator_trace = allocator.get_actuator_trace(thrusts, tau_cmd, tau_actual)

            row = {
                "time": (step + 1) * dt,  # 积分后时间
                "x": state.x,
                "y": state.y,
                "psi": state.psi,
                "u": state.u,
                "v": state.v,
                "r": state.r,
                "target_x": target_x,
                "target_y": target_y,
                "target_psi": target_psi_rad,
                "position_error": pos_err,
                "heading_error": head_err,
                "tau_x": tau_actual[0],
                "tau_y": tau_actual[1],
                "tau_n": tau_actual[2],
                "tau_cmd_x": tau_cmd[0],
                "tau_cmd_y": tau_cmd[1],
                "tau_cmd_n": tau_cmd[2],
                "allocation_residual": alloc_residual,
                "ice_concentration": ice.concentration,
                "ice_thickness": ice.thickness,
                "ice_drift_speed": ice.drift_speed,
                "ice_drift_direction": ice.drift_direction,
                "violation": violation,
                "boundary_violation": violation,
                "thrust_saturation": thrust_saturation,
                "allocation_success": 1.0 if allocator is None else float(feasible),
                "power_kw": power_kw,
                "risk_total": diag.get("risk_total", 0.0),
                "risk_ice": diag.get("risk_ice", 0.0),
                "risk_cvar": diag.get("risk_cvar", 0.0),
                "solver_time_ms": diag.get("solve_time_ms", 0.0),
                "solver_success": 1.0 if diag.get("solver_success", True) else 0.0,
                "cbf_active": 1.0 if diag.get("cbf_active", False) else 0.0,
                "cbf_slack": diag.get("cbf_slack", 0.0),
                "supervisor_mode": diag.get("supervisor_mode", 0),
                "energy": cumulative_energy,
                "observer_ice_fx": float(observer_ice_estimate[0]),
                "observer_ice_fy": float(observer_ice_estimate[1]),
                "observer_ice_mz": float(observer_ice_estimate[2]),
            }
            # 合并执行器证据字段
            row.update(actuator_trace)
            # 合并 safety filter 证据字段 (从 controller diagnostics)
            for sf_key in [
                "tau_des_x", "tau_des_y", "tau_des_n",
                "tau_safe_x", "tau_safe_y", "tau_safe_n",
                "safety_filter_active", "safety_filter_qp_success", "safety_filter_infeasible",
                "safety_filter_status", "safety_filter_solver_backend", "safety_filter_solve_time_ms",
                "safety_filter_iter", "safety_filter_slack",
                "safety_filter_slack_active", "safety_set_h", "safety_set_h_dot",
                "safety_filter_hocbf_margin", "safety_filter_correction_norm",
                "hocbf_constraint_margin", "hocbf_nominal_constraint_margin",
                "hocbf_robust_disturbance_margin", "hocbf_disturbance_accel_bound_mps2",
                "hocbf_a_norm", "hocbf_b", "hocbf_b_nominal", "hocbf_soft_certificate",
                "safety_filter_risk_level", "safety_filter_risk_scale",
                "safety_filter_alpha1", "safety_filter_alpha2",
                "safety_filter_constraint_mode", "safety_filter_feasible_set_type",
                "actuator_feasible_mode", "safety_filter_solver_setup_count",
                "safety_filter_solver_update_count", "safety_filter_osqp_reused_factorization",
            ]:
                if sf_key in diag:
                    row[sf_key] = diag[sf_key]
            log.append(row)

        if verbose and step % (n_steps // 10 + 1) == 0:
            _sim_logger.info("step %d/%d (t=%.1fs)", step, n_steps, t)

    return log
