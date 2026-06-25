"""仿真器 — 3-DOF 船舶动力学仿真。

实现与控制器接口兼容的仿真循环，支持：
- 3-DOF 动力学积分 (RK4)
- 冰况参数注入
- 控制器调用
- 完整日志记录
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..sci1.sim_loop import compute_dynamics_derivatives


@dataclass
class SimulationConfig:
    """仿真配置。"""
    duration: float = 100.0
    dt: float = 0.1
    target_x: float = 0.0
    target_y: float = 0.0
    target_psi: float = 0.0
    ice_concentration: float = 0.0
    ice_thickness: float = 0.0
    ice_drift_speed: float = 0.0
    ice_drift_direction: float = 0.0
    verbose: bool = False
    seed: int = 2026
    trial: int = 0
    # 扩展字段 (由 sci1 runner 使用)
    ice_schedule: Any = None


@dataclass
class SimulationLog:
    """仿真日志。"""
    rows: List[Dict[str, float]] = field(default_factory=list)

    def append(self, row: Dict[str, float]) -> None:
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class Simulator:
    """3-DOF 船舶动力学仿真器。"""

    def __init__(self, safe_region_radius: float = 10.0):
        self.safe_region_radius = safe_region_radius
        # 船舶参数 (与 sim_loop.VesselParams 一致)
        self.mass = 500000.0     # 500 tonnes (简化)
        self.Izz = 5e8           # kg·m²
        self.Xu = 500.0
        self.Yv = 800.0
        self.Nr = 2e5
        self.Xu_abs = 200.0
        self.Yv_abs = 300.0
        self.Nr_abs = 5e4
        self.vessel_length = 122.5
        self.vessel_beam = 22.0

    def _ice_force(
        self, ice: Dict[str, float], psi: float,
    ) -> NDArray[np.float64]:
        """Lindqvist 简化冰力模型。"""
        c = float(np.clip(ice.get("concentration", 0.0), 0.0, 1.0))
        h = max(0.0, float(ice.get("thickness", 0.0)))
        v = max(0.0, float(ice.get("drift_speed", 0.0)))
        direction = math.radians(float(ice.get("drift_direction", 0.0)))

        speed_factor = 1.0 + 0.5 * v / (v + 0.5) if v > 0 else 1.0
        alpha = math.radians(30.0)
        angle_factor = 1.0 + 0.3 * math.tan(min(alpha, math.pi / 3))
        base_force = 2.0 * 1000.0 * h * self.vessel_beam * 0.45 * speed_factor * angle_factor * c

        force_ned = base_force * np.array([math.cos(direction), math.sin(direction)])
        cpsi, spsi = math.cos(psi), math.sin(psi)
        R = np.array([[cpsi, spsi], [-spsi, cpsi]])
        force_body = R @ force_ned
        lever = 0.18 * self.vessel_length
        moment = lever * force_body[1]
        return np.array([force_body[0], force_body[1], moment], dtype=np.float64)

    def _dynamics(
        self, state: NDArray, tau_ctrl: NDArray, tau_ice: NDArray,
    ) -> NDArray:
        """调用公共动力学核心。"""
        return compute_dynamics_derivatives(
            state[2], state[3], state[4], state[5],
            tau_ctrl, tau_ice,
            self.mass, self.Izz,
            self.Xu, self.Yv, self.Nr,
            self.Xu_abs, self.Yv_abs, self.Nr_abs,
        )

    def _rk4(self, state: NDArray, tau: NDArray, tau_ice: NDArray, dt: float) -> NDArray:
        k1 = self._dynamics(state, tau, tau_ice)
        k2 = self._dynamics(state + 0.5 * dt * k1, tau, tau_ice)
        k3 = self._dynamics(state + 0.5 * dt * k2, tau, tau_ice)
        k4 = self._dynamics(state + dt * k3, tau, tau_ice)
        s = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s[2] = (s[2] + math.pi) % (2 * math.pi) - math.pi
        return s

    def run(
        self,
        cfg: SimulationConfig,
        controller,
        log_interval: int = 1,
        config_hash: str = "",
    ) -> SimulationLog:
        """运行仿真。"""
        controller.set_target(cfg.target_x, cfg.target_y, cfg.target_psi)
        if hasattr(controller, 'set_safe_region_radius'):
            controller.set_safe_region_radius(self.safe_region_radius)
        if hasattr(controller, 'set_ice_conditions'):
            controller.set_ice_conditions(
                cfg.ice_concentration, cfg.ice_thickness,
                cfg.ice_drift_speed, cfg.ice_drift_direction,
            )

        n_steps = int(cfg.duration / cfg.dt)
        state = np.zeros(6, dtype=np.float64)
        log = SimulationLog()

        for step in range(n_steps):
            t = step * cfg.dt
            # 时变冰况
            if cfg.ice_schedule is not None and hasattr(controller, 'set_ice_conditions'):
                ice = cfg.ice_schedule(t) if callable(cfg.ice_schedule) else cfg.ice_schedule
                controller.set_ice_conditions(
                    ice["concentration"], ice["thickness"],
                    ice["drift_speed"], ice["drift_direction"],
                )

            result = controller.compute_control(state, dt=cfg.dt)
            tau = np.asarray(result.tau, dtype=np.float64).reshape(3,)

            ice_dict = {
                "concentration": cfg.ice_concentration,
                "thickness": cfg.ice_thickness,
                "drift_speed": cfg.ice_drift_speed,
                "drift_direction": cfg.ice_drift_direction,
            }
            if cfg.ice_schedule is not None:
                ice_dict = cfg.ice_schedule(t) if callable(cfg.ice_schedule) else cfg.ice_schedule

            tau_ice = self._ice_force(ice_dict, state[2])
            state = self._rk4(state, tau, tau_ice, cfg.dt)

            if step % log_interval == 0:
                pos_err = math.sqrt(
                    (state[0] - cfg.target_x) ** 2 + (state[1] - cfg.target_y) ** 2
                )
                head_err = abs((state[2] - cfg.target_psi + math.pi) % (2 * math.pi) - math.pi)
                violation = 1.0 if pos_err > self.safe_region_radius else 0.0
                diag = controller.get_diagnostics() if hasattr(controller, 'get_diagnostics') else {}
                log.append({
                    "time": t,
                    "x": float(state[0]), "y": float(state[1]), "psi": float(state[2]),
                    "u": float(state[3]), "v": float(state[4]), "r": float(state[5]),
                    "position_error": pos_err, "heading_error": head_err,
                    "tau_x": tau[0], "tau_y": tau[1], "tau_n": tau[2],
                    "violation": violation, "boundary_violation": violation,
                    "risk_total": diag.get("risk_total", 0.0),
                    "risk_cvar": diag.get("risk_cvar", 0.0),
                    "solver_time_ms": diag.get("solve_time_ms", 0.0),
                    "solver_success": 1.0 if diag.get("solver_success", True) else 0.0,
                    "energy": 0.0,
                })

        # 累积能耗
        cum_e = 0.0
        for row in log.rows:
            cum_e += math.sqrt(row["tau_x"] ** 2 + row["tau_y"] ** 2) * cfg.dt * 0.001
            row["energy"] = cum_e

        return log
