"""基于 CasADi 的非线性模型预测控制 (NMPC) 冰区 DP 控制器。

实现：
- 3-DOF 船舶动力学模型 (surge, sway, yaw)
- 冰载荷前馈扰动模型
- Control Barrier Function (CBF) 安全约束
- 推进器饱和约束
- 二次代价函数 (位置跟踪 + 控制 effort + 状态约束)

依赖：
    pip install casadi

使用：
    nmpc = NMPCIceController(NMPCParams())
    nmpc.set_target(0.0, 0.0, 0.0)
    nmpc.set_ice_conditions(0.5, 0.8, 0.3, 120.0)
    result = nmpc.compute_control(state, dt=0.1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import time
import math

import numpy as np
from numpy.typing import NDArray

from ..controllers.base import BaseController, ControllerResult
from ..utils.math_utils import deg2rad, wrap_to_pi
from .thruster import ThrusterAllocator, ThrusterConfig

# CasADi 延迟导入 (允许在未安装时仍能 import 本模块)
_casadi_available = False
try:
    import casadi as cs
    _casadi_available = True
except ImportError:
    pass


@dataclass
class NMPCParams:
    """NMPC 参数。"""
    # 预测步数和时间步长
    N: int = 20                      # 预测步数
    dt_nmpc: float = 0.2             # NMPC 内部时间步长 (s)

    # 船舶物理参数 (与 sim_loop.VesselParams 一致)
    mass: float = 500000.0           # 船舶质量 (kg) — 简化为 500t
    Izz: float = 5e8                 # 偏航转动惯量 (kg·m²)
    Xu: float = -500.0               # surge 线性阻尼 (N·s/m)
    Yv: float = -800.0               # sway 线性阻尼 (N·s/m)
    Nr: float = -2e5                 # yaw 线性阻尼 (N·m·s/rad)
    Xu_abs: float = -200.0           # surge 二次阻尼 (N·s²/m²)
    Yv_abs: float = -300.0           # sway 二次阻尼 (N·s²/m²)
    Nr_abs: float = -5e4             # yaw 二次阻尼 (N·m·s²/rad²)
    # 与 sim_loop.VesselParams 保持一致

    # 代价权重
    Q_pos: float = 100.0             # 位置误差权重
    Q_heading: float = 50.0          # 艏向误差权重
    Q_vel: float = 1.0               # 速度权重
    R_force: float = 0.001           # 控制力权重
    R_moment: float = 0.0001         # 控制力矩权重
    dR_force: float = 0.01           # 控制力变化率权重
    dR_moment: float = 0.001         # 控制力矩变化率权重

    # 约束
    max_force: float = 1500.0        # 最大合力 (N)
    max_moment: float = 20000.0      # 最大力矩 (N·m)
    safe_radius: float = 10.0        # CBF 安全半径 (m)
    cbf_gamma: float = 1.0           # CBF 收敛速率

    # 冰力模型参数
    ice_crushing_strength_mpa: float = 2.0
    ice_structure_factor: float = 0.45
    vessel_beam_m: float = 22.0
    vessel_length_m: float = 122.5
    waterline_angle_deg: float = 30.0

    solver_label: str = "nmpc_casadi"


def _rotation_body_from_ned(psi: float):
    """返回旋转矩阵 (符号或数值)。"""
    c = cs.cos(psi) if _casadi_available else math.cos(psi)
    s = cs.sin(psi) if _casadi_available else math.sin(psi)
    return cs.vertcat(
        cs.horzcat(c, s),
        cs.horzcat(-s, c),
    ) if _casadi_available else np.array([[c, s], [-s, c]])


def _ice_force_symbolic(
    concentration,
    thickness,
    drift_speed,
    drift_dir,
    psi,
    params: NMPCParams,
):
    """符号冰力模型 (CasADi MX/SX 兼容)。"""
    c = cs.fmax(0.0, cs.fmin(1.0, concentration))
    h = cs.fmax(0.0, thickness)
    v = cs.fmax(0.0, drift_speed)

    # Lindqvist 简化 (含水线角因子)
    speed_factor = 1.0 + 0.5 * v / (v + 0.5)
    alpha_val = params.waterline_angle_deg * 3.141592653589793 / 180.0
    angle_factor = 1.0 + 0.3 * cs.tan(cs.fmin(alpha_val, 3.141592653589793 / 3.0))
    base_force = (
        params.ice_crushing_strength_mpa * 1000.0
        * h * params.vessel_beam_m * params.ice_structure_factor
        * speed_factor * angle_factor * c
    )

    force_ned_x = base_force * cs.cos(drift_dir)
    force_ned_y = base_force * cs.sin(drift_dir)

    # 旋转到船体坐标系
    cpsi = cs.cos(psi)
    spsi = cs.sin(psi)
    fx = cpsi * force_ned_x + spsi * force_ned_y
    fy = -spsi * force_ned_x + cpsi * force_ned_y

    # 力矩
    lever = 0.18 * params.vessel_length_m
    mz = lever * fy

    return fx, fy, mz


class NMPCIceController(BaseController):
    """基于 CasADi 的 NMPC 冰区 DP 控制器。"""

    def __init__(
        self,
        params: Optional[NMPCParams] = None,
        thruster_config: Optional[ThrusterConfig] = None,
    ):
        super().__init__()
        if not _casadi_available:
            raise ImportError(
                "CasADi is required for NMPC controller. "
                "Install with: pip install casadi"
            )
        self.params = params or NMPCParams()
        self._solver_label = self.params.solver_label
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_psi: float = 0.0
        self._raw_ice = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
        self._ice_est = dict(self._raw_ice)
        self._rng = np.random.default_rng(2026)

        # 推进器分配器
        if thruster_config is not None:
            self._allocator = ThrusterAllocator(thruster_config)
        else:
            self._allocator = ThrusterAllocator(ThrusterConfig.generic_dp_vessel())

        # 上一次控制输入 (用于变化率惩罚)
        self._prev_tau = np.zeros(3, dtype=np.float64)

        # 构建 NMPC 问题
        self._build_nlp()

    def _build_nlp(self) -> None:
        """构建 CasADi NLP 问题。"""
        p = self.params
        N = p.N
        dt = p.dt_nmpc

        # 符号变量
        # 参数: [x_ref, y_ref, psi_ref, ice_c, ice_h, ice_v, ice_dir,
        #        prev_Fx, prev_Fy, prev_Mz,
        #        eta_x, eta_y, psi, u, v, r]
        param = cs.MX.sym("param", 16)

        # 状态和控制的符号序列
        X = cs.MX.sym("X", 6, N + 1)
        U = cs.MX.sym("U", 3, N)

        # 从参数提取冰况 (常量符号)
        ice_c = param[3]
        ice_h = param[4]
        ice_v = param[5]
        ice_dir = param[6]

        # 动力学函数 (连续), 冰力按当前状态 psi 计算
        def dynamics(xk, tau_k):
            psi = xk[2]
            u_body = xk[3]
            v_body = xk[4]
            r = xk[5]

            cpsi = cs.cos(psi)
            spsi = cs.sin(psi)

            # NED 速度
            xdot_ned = cpsi * u_body - spsi * v_body
            ydot_ned = spsi * u_body + cpsi * v_body

            # 冰力 (基于当前步的 heading)
            ice_fx, ice_fy, ice_mz = _ice_force_symbolic(
                ice_c, ice_h, ice_v, ice_dir, psi, p
            )

            # 船体动力学 (含阻尼和冰力)
            u_dot = (tau_k[0] + ice_fx - p.Xu * u_body - p.Xu_abs * cs.fabs(u_body) * u_body) / p.mass
            v_dot = (tau_k[1] + ice_fy - p.Yv * v_body - p.Yv_abs * cs.fabs(v_body) * v_body) / p.mass
            r_dot = (tau_k[2] + ice_mz - p.Nr * r - p.Nr_abs * cs.fabs(r) * r) / p.Izz

            return cs.vertcat(xdot_ned, ydot_ned, r, u_dot, v_dot, r_dot)

        # 离散化 (RK4)
        def rk4_step(xk, uk):
            k1 = dynamics(xk, uk)
            k2 = dynamics(xk + 0.5 * dt * k1, uk)
            k3 = dynamics(xk + 0.5 * dt * k2, uk)
            k4 = dynamics(xk + dt * k3, uk)
            return xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 构建 NLP
        # 目标函数
        obj = 0.0
        # 约束
        g = []  # 等式/不等式约束
        lbg = []
        ubg = []

        # 初始状态约束 (从参数向量提取当前状态)
        x0_param = param[10:16]
        g.append(X[:, 0] - x0_param)
        lbg.extend([0.0] * 6)
        ubg.extend([0.0] * 6)

        for k in range(N):
            # 动力学约束 (通过 RK4)
            x_next = rk4_step(X[:, k], U[:, k])
            g.append(X[:, k + 1] - x_next)
            lbg.extend([0.0] * 6)
            ubg.extend([0.0] * 6)

            # 代价: 位置跟踪
            pos_err_x = X[0, k] - param[0]
            pos_err_y = X[1, k] - param[1]
            heading_err = cs.sin(X[2, k] - param[2])  # 用 sin 避免角度不连续
            obj += p.Q_pos * (pos_err_x ** 2 + pos_err_y ** 2)
            obj += p.Q_heading * heading_err ** 2

            # 代价: 速度阻尼
            obj += p.Q_vel * (X[3, k] ** 2 + X[4, k] ** 2 + X[5, k] ** 2)

            # 代价: 控制 effort
            obj += p.R_force * (U[0, k] ** 2 + U[1, k] ** 2)
            obj += p.R_moment * U[2, k] ** 2

            # 代价: 控制变化率
            if k == 0:
                prev = cs.vertcat(param[7], param[8], param[9])
            else:
                prev = U[:, k - 1]
            dtau = U[:, k] - prev
            obj += p.dR_force * (dtau[0] ** 2 + dtau[1] ** 2)
            obj += p.dR_moment * dtau[2] ** 2

            # 控制约束: 推进器饱和
            g.append(cs.sqrt(U[0, k] ** 2 + U[1, k] ** 2))
            lbg.append(0.0)
            ubg.append(p.max_force)
            g.append(cs.fabs(U[2, k]))
            lbg.append(0.0)
            ubg.append(p.max_moment)

            # CBF 安全约束: h(x) = R^2 - ||pos - pos_ref||^2 >= 0
            # h_dot + gamma * h >= 0
            h_val = p.safe_radius ** 2 - (pos_err_x ** 2 + pos_err_y ** 2)
            # 近似 h_dot ≈ -2 * (pos_err * vel_ned)
            vel_ned_x = cs.cos(X[2, k]) * X[3, k] - cs.sin(X[2, k]) * X[4, k]
            vel_ned_y = cs.sin(X[2, k]) * X[3, k] + cs.cos(X[2, k]) * X[4, k]
            h_dot = -2.0 * (pos_err_x * vel_ned_x + pos_err_y * vel_ned_y)
            g.append(h_dot + p.cbf_gamma * h_val)
            lbg.append(0.0)
            ubg.append(1e20)

        # 终端代价
        pos_err_x_T = X[0, N] - param[0]
        pos_err_y_T = X[1, N] - param[1]
        heading_err_T = cs.sin(X[2, N] - param[2])
        obj += 2.0 * p.Q_pos * (pos_err_x_T ** 2 + pos_err_y_T ** 2)
        obj += 2.0 * p.Q_heading * heading_err_T ** 2

        # NLP 变量
        w = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
        nlp = {"f": obj, "x": w, "p": param, "g": cs.vertcat(*g)}

        # 求解器选项
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "print_time": 0,
            "verbose": False,
        }

        try:
            self._solver = cs.nlpsol("nmpc_solver", "ipopt", nlp, opts)
        except Exception:
            # 如果 IPOPT 不可用，尝试使用 SLSQP
            opts_slsqp = {"print_time": 0, "verbose": False}
            self._solver = cs.nlpsol("nmpc_solver", "sqpmethod", nlp, opts_slsqp)

        self._N = N
        self._n_x = 6
        self._n_u = 3
        self._n_w = 6 * (N + 1) + 3 * N
        self._n_g = len(lbg)
        self._lbg = lbg
        self._ubg = ubg

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        self._target_pos = (float(x), float(y))
        self._target_psi = deg2rad(float(psi_deg))
        self.target_position = self._target_pos
        self.target_heading = self._target_psi

    def set_ice_conditions(self, ice_concentration: float, ice_thickness: float, ice_drift_speed: float, ice_drift_direction: float = 0.0) -> None:
        self._raw_ice = {
            "concentration": float(np.clip(ice_concentration, 0.0, 1.0)),
            "thickness": max(0.0, float(ice_thickness)),
            "drift_speed": max(0.0, float(ice_drift_speed)),
            "drift_direction": float(ice_drift_direction),
        }
        self._ice_est = dict(self._raw_ice)

    def set_safe_region_radius(self, radius: float) -> None:
        self.params.safe_radius = float(radius)

    def set_cvar_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def compute_control(
        self,
        state: NDArray[np.float64],
        reference: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ControllerResult:
        tic = time.perf_counter()
        dt = float(kwargs.get("dt", 0.1))

        if self._target_pos is None:
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="nmpc", risk=0.0)

        p = self.params

        # 构建参数向量
        # [x_ref, y_ref, psi_ref, ice_c, ice_h, ice_v, ice_dir,
        #  prev_Fx, prev_Fy, prev_Mz,
        #  eta_x, eta_y, psi, u, v, r]
        ice = self._ice_est
        x0 = np.asarray(state, dtype=np.float64).reshape(6,)
        param_val = np.array([
            self._target_pos[0],
            self._target_pos[1],
            self._target_psi,
            ice["concentration"],
            ice["thickness"],
            ice["drift_speed"],
            deg2rad(ice["drift_direction"]),
            self._prev_tau[0],
            self._prev_tau[1],
            self._prev_tau[2],
            x0[0], x0[1], x0[2], x0[3], x0[4], x0[5],
        ], dtype=np.float64)

        # 初始猜测: 状态保持 + 零控制
        x0 = np.asarray(state, dtype=np.float64).reshape(6,)
        X_init = np.tile(x0.reshape(6, 1), (1, p.N + 1))
        U_init = np.zeros((3, p.N))
        w0 = np.concatenate([X_init.flatten(), U_init.flatten()])

        # 上下界
        lbw = np.full(self._n_w, -np.inf)
        ubw = np.full(self._n_w, np.inf)

        # 控制变量的界 (在 w 中的位置: 6*(N+1) 开始)
        u_offset = 6 * (p.N + 1)
        for k in range(p.N):
            # Fx, Fy: ±max_force
            lbw[u_offset + 3 * k] = -p.max_force
            ubw[u_offset + 3 * k] = p.max_force
            lbw[u_offset + 3 * k + 1] = -p.max_force
            ubw[u_offset + 3 * k + 1] = p.max_force
            # Mz: ±max_moment
            lbw[u_offset + 3 * k + 2] = -p.max_moment
            ubw[u_offset + 3 * k + 2] = p.max_moment

        # 求解
        feasible = True
        try:
            sol = self._solver(
                x0=w0,
                p=param_val,
                lbx=lbw, ubx=ubw,
                lbg=np.array(self._lbg),
                ubg=np.array(self._ubg),
            )
            w_opt = np.array(sol["x"]).flatten()

            # 提取第一步控制
            u_opt = w_opt[u_offset:u_offset + 3]
            tau = u_opt.copy()

            # 推进器分配
            thrusts, thrust_feasible = self._allocator.allocate(tau)
            actual_tau = self._allocator.resulting_tau(thrusts)
            if not thrust_feasible:
                tau = actual_tau  # 使用实际可分配的力

        except Exception:
            # 求解失败: 回退到 PD 控制
            tau = self._fallback_pd(state)
            feasible = False

        tau[0:2] = np.clip(tau[0:2], -p.max_force, p.max_force)
        tau[2] = np.clip(tau[2], -p.max_moment, p.max_moment)

        self._prev_tau = tau.copy()

        # 风险估计 (与 IceAware/supervisor 三因子公式一致)
        pos_err = float(np.linalg.norm(np.asarray(state[:2]) - np.asarray(self._target_pos)))
        ice_risk = float(np.clip(
            ice["concentration"] * (0.3 + ice["thickness"]) * (0.4 + ice["drift_speed"]),
            0.0, 1.0,
        ))
        cvar_est = float(np.clip(0.15 * ice["concentration"] * (0.25 + ice["thickness"]), 0.0, 1.0))
        risk = float(np.clip(0.35 * min(1.0, pos_err / 15.0) + 0.35 * ice_risk + 0.30 * cvar_est, 0.0, 1.0))

        elapsed = (time.perf_counter() - tic) * 1000.0
        self._last_diagnostics = {
            "solver_status": "nmpc_casadi" if feasible else "nmpc_fallback_pd",
            "solver_success": feasible,
            "solve_time_ms": elapsed,
            "objective_value": float(tau @ tau),
            "constraint_violation": max(0.0, pos_err - p.safe_radius),
            "risk_total": risk,
            "risk_ice": ice_risk,
            "risk_position": min(1.0, pos_err / 15.0),
            "risk_cvar": 0.0,
            "risk_model_status": "nmpc",
            "cbf_active": pos_err > 0.75 * p.safe_radius,
            "cbf_status": "nmpc_constrained",
            "cbf_slack": p.safe_radius - pos_err,
        }

        return ControllerResult(
            tau=tau, feasible=feasible, mode="nmpc", risk=risk,
            cost_estimate=pos_err * pos_err,
        )

    def _fallback_pd(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """NMPC 求解失败时的 PD 回退控制。"""
        if self._target_pos is None:
            return np.zeros(3)
        p = self.params
        x, y, psi = float(state[0]), float(state[1]), float(state[2])
        u, v, r = float(state[3]), float(state[4]), float(state[5])

        pos_err_ned = np.array([x - self._target_pos[0], y - self._target_pos[1]])
        e_psi = wrap_to_pi(psi - self._target_psi)

        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        R = np.array([[cpsi, spsi], [-spsi, cpsi]])
        pos_err_body = R @ pos_err_ned

        # 与 PrecisionDPParams 增益一致
        Kp = 180.0
        Kd = 90.0
        Kh = 650.0
        Kr = 280.0

        fx = -Kp * pos_err_body[0] - Kd * u
        fy = -Kp * pos_err_body[1] - Kd * v
        mz = -Kh * e_psi - Kr * r

        tau = np.array([fx, fy, mz])
        tau[0:2] = np.clip(tau[0:2], -p.max_force, p.max_force)
        tau[2] = np.clip(tau[2], -p.max_moment, p.max_moment)
        return tau

    def reset(self) -> None:
        self._prev_tau = np.zeros(3, dtype=np.float64)
        self._ice_est = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
        self._last_diagnostics = {}


def check_casadi_available() -> bool:
    """检查 CasADi 是否可用。"""
    return _casadi_available
