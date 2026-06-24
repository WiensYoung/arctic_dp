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
    """船舶物理参数。"""
    mass: float = 500000.0         # kg (雪龙2号 ~14000t, 简化为 500t)
    Izz: float = 5e8               # kg·m²
    Xu: float = -500.0             # surge linear damping
    Yv: float = -800.0             # sway linear damping
    Nr: float = -2e5               # yaw linear damping
    Xu_abs: float = -200.0         # surge quadratic damping
    Yv_abs: float = -300.0         # sway quadratic damping
    Nr_abs: float = -5e4           # yaw quadratic damping
    length: float = 122.5          # vessel length (m)
    beam: float = 22.0             # vessel beam (m)
    ice_crushing_strength_mpa: float = 2.0
    ice_structure_factor: float = 0.45
    waterline_angle_deg: float = 30.0        # 水线角 (度)


def _ice_force_body(
    ice: Dict[str, float],
    psi: float,
    params: VesselParams,
) -> NDArray[np.float64]:
    """计算冰力 (船体坐标系)。使用 Lindqvist 简化模型。"""
    c = float(np.clip(ice.get("concentration", 0.0), 0.0, 1.0))
    h = max(0.0, float(ice.get("thickness", 0.0)))
    v = max(0.0, float(ice.get("drift_speed", 0.0)))
    direction = math.radians(float(ice.get("drift_direction", 0.0)))

    speed_factor = 1.0 + 0.5 * v / (v + 0.5) if v > 0 else 1.0
    # 水线角因子 (与 controllers.py 和 simulator.py 一致)
    alpha = math.radians(params.waterline_angle_deg)
    angle_factor = 1.0 + 0.3 * math.tan(min(alpha, math.pi / 3))
    base_force = (
        params.ice_crushing_strength_mpa * 1000.0
        * h * params.beam * params.ice_structure_factor
        * speed_factor * angle_factor * c
    )

    force_ned = base_force * np.array([math.cos(direction), math.sin(direction)])
    cpsi, spsi = math.cos(psi), math.sin(psi)
    R = np.array([[cpsi, spsi], [-spsi, cpsi]])
    force_body = R @ force_ned

    lever = 0.18 * params.length
    moment = lever * force_body[1]

    return np.array([force_body[0], force_body[1], moment], dtype=np.float64)


def _dynamics(
    state: VesselState,
    tau_control: NDArray[np.float64],
    tau_ice: NDArray[np.float64],
    params: VesselParams,
) -> NDArray[np.float64]:
    """计算状态导数 (连续时间)。"""
    psi = state.psi
    u, v, r = state.u, state.v, state.r

    cpsi, spsi = math.cos(psi), math.sin(psi)
    xdot = cpsi * u - spsi * v
    ydot = spsi * u + cpsi * v
    psidot = r

    total_fx = tau_control[0] + tau_ice[0]
    total_fy = tau_control[1] + tau_ice[1]
    total_mz = tau_control[2] + tau_ice[2]

    udot = (total_fx - params.Xu * u - params.Xu_abs * abs(u) * u) / params.mass
    vdot = (total_fy - params.Yv * v - params.Yv_abs * abs(v) * v) / params.mass
    rdot = (total_mz - params.Nr * r - params.Nr_abs * abs(r) * r) / params.Izz

    return np.array([xdot, ydot, psidot, udot, vdot, rdot], dtype=np.float64)


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
        allocator = ThrusterAllocator(thruster_config)
        if degradation_profile is not None:
            degradation_profile.apply(allocator)

    # 初始化
    controller.set_target(target_x, target_y, target_psi)
    if hasattr(controller, 'set_safe_region_radius'):
        controller.set_safe_region_radius(safe_region_radius)

    state = VesselState()
    n_steps = int(duration / dt)
    log = SimLog()
    rng = np.random.default_rng(seed)

    for step in range(n_steps):
        t = step * dt

        # 1. 获取当前冰况
        ice = ice_schedule.at(t)
        ice_dict = ice.to_dict()

        # 2. 更新控制器冰况
        if hasattr(controller, 'set_ice_conditions'):
            controller.set_ice_conditions(
                ice.concentration, ice.thickness,
                ice.drift_speed, ice.drift_direction,
            )

        # 3. 计算控制
        state_arr = state.to_array()
        result = controller.compute_control(state_arr, dt=dt)
        tau_cmd = np.asarray(result.tau, dtype=np.float64).reshape(3,)

        # 4. 推进器分配
        tau_actual = tau_cmd
        thrust_saturation = 0.0
        if allocator is not None:
            thrusts, feasible = allocator.allocate(tau_cmd)
            tau_actual = allocator.resulting_tau(thrusts)
            thrust_saturation = allocator.thrust_saturation_ratio(thrusts)

        # 5. 计算冰力
        tau_ice = _ice_force_body(ice_dict, state.psi, params)

        # 5b. 风场强迫 (如果有)
        tau_wind = np.zeros(3, dtype=np.float64)
        if wind_schedule is not None:
            wind = wind_schedule.at(t)
            # 风力代理: F = 0.5 * rho_air * Cd * A * V^2
            rho_air = 1.225  # kg/m³
            Cd = 0.8         # 风阻力系数
            A_front = params.beam * 10.0  # 迎风面积 (简化: 船宽 × 10m)
            A_side = params.length * 10.0  # 侧风面积
            u_wind = wind.u10 - state.u  # 相对风速
            v_wind = wind.v10 - state.v
            fx_wind = 0.5 * rho_air * Cd * A_front * u_wind * abs(u_wind)
            fy_wind = 0.5 * rho_air * Cd * A_side * v_wind * abs(v_wind)
            tau_wind = np.array([fx_wind, fy_wind, fy_wind * params.length * 0.3])

        # 6. 积分动力学 (控制力 + 冰力 + 风力)
        state = _rk4_step(state, tau_actual, tau_ice + tau_wind, params, dt)

        # 7. 记录日志
        if step % log_interval == 0:
            pos_err = math.sqrt((state.x - target_x) ** 2 + (state.y - target_y) ** 2)
            head_err = abs((state.psi - target_psi + math.pi) % (2 * math.pi) - math.pi)
            dist_from_target = pos_err
            violation = 1.0 if dist_from_target > safe_region_radius else 0.0

            diag = controller.get_diagnostics() if hasattr(controller, 'get_diagnostics') else {}

            row = {
                "time": t,
                "x": state.x,
                "y": state.y,
                "psi": state.psi,
                "u": state.u,
                "v": state.v,
                "r": state.r,
                "position_error": pos_err,
                "heading_error": head_err,
                "tau_x": tau_actual[0],
                "tau_y": tau_actual[1],
                "tau_n": tau_actual[2],
                "tau_cmd_x": tau_cmd[0],
                "tau_cmd_y": tau_cmd[1],
                "tau_cmd_n": tau_cmd[2],
                "ice_concentration": ice.concentration,
                "ice_thickness": ice.thickness,
                "ice_drift_speed": ice.drift_speed,
                "ice_drift_direction": ice.drift_direction,
                "violation": violation,
                "boundary_violation": violation,
                "thrust_saturation": thrust_saturation,
                "risk_total": diag.get("risk_total", 0.0),
                "risk_ice": diag.get("risk_ice", 0.0),
                "risk_cvar": diag.get("risk_cvar", 0.0),
                "solver_time_ms": diag.get("solve_time_ms", 0.0),
                "solver_success": 1.0 if diag.get("solver_success", True) else 0.0,
                "cbf_active": 1.0 if diag.get("cbf_active", False) else 0.0,
                "supervisor_mode": diag.get("supervisor_mode", 0),
                "energy": 0.0,  # 累积能耗 (下面计算)
            }
            log.append(row)

        if verbose and step % (n_steps // 10 + 1) == 0:
            print(f"  step {step}/{n_steps} (t={t:.1f}s)")

    # 计算累积能耗
    if log.rows:
        cumulative_energy = 0.0
        for row in log.rows:
            tau_mag = math.sqrt(row["tau_x"] ** 2 + row["tau_y"] ** 2)
            cumulative_energy += tau_mag * dt * 0.001  # 简化能耗
            row["energy"] = cumulative_energy

    return log
