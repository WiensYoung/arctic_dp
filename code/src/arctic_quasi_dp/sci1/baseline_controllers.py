"""SCI一区投稿补充基线控制器。

实现两个经典控制方法作为对比基线:
1. LQGController — 线性二次高斯控制 (LQR + Kalman Filter)
2. DOBMPCController — 基于扰动观测器的 NMPC

这些控制器遵循 BaseController 接口，可直接被 runner 和 sim_loop 调用。

参考:
- Fossen (2011) "Handbook of Marine Craft Hydrodynamics and Motion Control"
- LQR/Kalman: Anderson & Moore (1971) "Linear Optimal Control"
- DOB-MPC: Kim et al. (2023) "Disturbance Observer-Based MPC for Marine Vehicles"
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
from .sensor_models import IceLoadObserver


# ============================================================
# LQG 控制器 (LQR + Kalman Filter)
# ============================================================

@dataclass
class LQGParams:
    """LQG 控制器参数。"""
    # LQR 权重矩阵 (对角元素)
    Q_pos: float = 100.0       # 位置误差权重
    Q_heading: float = 50.0    # 艏向误差权重
    Q_vel: float = 1.0         # 速度权重
    R_force: float = 0.001     # 控制力权重
    R_moment: float = 0.0001   # 控制力矩权重

    # Kalman 滤波器参数
    process_noise_pos: float = 0.1       # 位置过程噪声 (m)
    process_noise_heading: float = 0.01  # 艏向过程噪声 (rad)
    process_noise_vel: float = 0.5       # 速度过程噪声 (m/s)
    measurement_noise_pos: float = 0.5   # 位置测量噪声 (m)
    measurement_noise_heading: float = 0.02  # 艏向测量噪声 (rad)

    # 船舶参数 (与 VesselParams 一致)
    mass: float = 500000.0
    Izz: float = 5e8
    Xu: float = 500.0
    Yv: float = 800.0
    Nr: float = 2e5

    max_force: float = 3000.0
    max_moment: float = 100000.0
    solver_label: str = "lqg"


class LQGController(BaseController):
    """线性二次高斯控制器 (LQR + Kalman Filter)。

    LQR 在平衡点附近线性化 3-DOF DP 动力学:
        ẋ = Ax + Bu + d (d = 扰动/冰力)

    Kalman 滤波器估计状态 [x, y, ψ, u, v, r]。
    LQR 增益通过求解代数 Riccati 方程获得。

    注意: LQG 不包含冰力感知、CBF 安全约束或模式降级——
    它是经典最优控制基线，用于对比增强方法的增量价值。
    """

    def __init__(self, params: Optional[LQGParams] = None):
        super().__init__()
        self.params = params or LQGParams()
        self._solver_label = self.params.solver_label
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_psi: float = 0.0

        # Kalman 滤波器状态
        self._x_hat = np.zeros(6, dtype=np.float64)  # 状态估计
        self._P = np.eye(6, dtype=np.float64) * 10.0  # 协方差矩阵

        # 构建 LQR 增益
        self._K = self._compute_lqr_gain()

    def _linearize_dynamics(self) -> Tuple[NDArray, NDArray]:
        """在零状态处线性化 3-DOF DP 动力学。

        Returns:
            A: 6×6 状态矩阵
            B: 6×3 控制矩阵
        """
        p = self.params
        # 状态: [x, y, ψ, u, v, r]
        # 在平衡点 (u=0, v=0, r=0, ψ=0) 线性化
        A = np.zeros((6, 6), dtype=np.float64)
        # 位置动力学: ẋ = u, ẏ = v, ψ̇ = r
        A[0, 3] = 1.0  # ẋ = u
        A[1, 4] = 1.0  # ẏ = v
        A[2, 5] = 1.0  # ψ̇ = r
        # 速度动力学 (线性化后)
        A[3, 3] = -p.Xu / p.mass    # u̇ = -Xu/m * u
        A[4, 4] = -p.Yv / p.mass    # v̇ = -Yv/m * v
        A[5, 5] = -p.Nr / p.Izz     # ṙ = -Nr/Iz * r

        # 控制矩阵
        B = np.zeros((6, 3), dtype=np.float64)
        B[3, 0] = 1.0 / p.mass   # Fx → u̇
        B[4, 1] = 1.0 / p.mass   # Fy → v̇
        B[5, 2] = 1.0 / p.Izz    # Mz → ṙ

        return A, B

    def _compute_lqr_gain(self) -> NDArray:
        """求解代数 Riccati 方程获得 LQR 增益矩阵 K。

        使用 scipy.linalg.solve_continuous_are 如果可用,
        否则使用简单的对角近似。
        """
        A, B = self._linearize_dynamics()
        p = self.params

        # 权重矩阵
        Q = np.diag([p.Q_pos, p.Q_pos, p.Q_heading, p.Q_vel, p.Q_vel, p.Q_vel])
        R = np.diag([p.R_force, p.R_force, p.R_moment])

        try:
            from scipy.linalg import solve_continuous_are
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            return K
        except ImportError:
            pass

        # 回退: 对角近似增益 (PD 形式)
        K = np.zeros((3, 6), dtype=np.float64)
        K[0, 0] = np.sqrt(p.Q_pos * p.R_force)     # x → Fx
        K[0, 3] = np.sqrt(p.Q_vel * p.R_force) * 2  # u → Fx
        K[1, 1] = np.sqrt(p.Q_pos * p.R_force)      # y → Fy
        K[1, 4] = np.sqrt(p.Q_vel * p.R_force) * 2  # v → Fy
        K[2, 2] = np.sqrt(p.Q_heading * p.R_moment)  # ψ → Mz
        K[2, 5] = np.sqrt(p.Q_vel * p.R_moment) * 2  # r → Mz
        return K

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        self._target_pos = (float(x), float(y))
        self._target_psi = deg2rad(float(psi_deg))
        self.target_position = self._target_pos
        self.target_heading = self._target_psi

    def set_ice_conditions(self, ice_concentration: float, ice_thickness: float,
                          ice_drift_speed: float, ice_drift_direction: float = 0.0) -> None:
        # LQG 不使用冰况信息 (经典控制基线)
        pass

    def set_safe_region_radius(self, radius: float) -> None:
        pass

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
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="lqg", risk=0.0)

        state = np.asarray(state, dtype=np.float64).reshape(6,)

        # Kalman 预测步
        A, B = self._linearize_dynamics()
        p = self.params
        # 过程噪声
        Q_k = np.diag([
            p.process_noise_pos ** 2, p.process_noise_pos ** 2,
            p.process_noise_heading ** 2,
            p.process_noise_vel ** 2, p.process_noise_vel ** 2,
            p.process_noise_vel ** 2,
        ])
        # 测量噪声
        R_k = np.diag([
            p.measurement_noise_pos ** 2, p.measurement_noise_pos ** 2,
            p.measurement_noise_heading ** 2,
            1e6, 1e6, 1e6,  # 速度不可直接测量 → 高噪声
        ])

        # 预测 (正确离散化: F = I + A*dt, Q_d = Q*dt)
        F = np.eye(6) + A * dt
        x_pred = F @ self._x_hat
        P_pred = F @ self._P @ F.T + Q_k * dt

        # 测量更新: 位置和艏向直接测量, 速度通过动力学模型间接估计
        z = np.zeros(6)
        z[:3] = state[:3]  # 仅位置/艏向作为测量值 (速度不可直接测量)
        H = np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # 仅位置/艏向可观测
        S = H @ P_pred @ H.T + R_k
        try:
            # H2 fix: use solve instead of inv for numerical stability
            K_kal = P_pred @ H.T @ np.linalg.solve(S, np.eye(6))
        except np.linalg.LinAlgError:
            K_kal = np.zeros((6, 6))
        innovation = z - H @ x_pred
        self._x_hat = x_pred + K_kal @ innovation
        # H2 fix: Joseph form for numerically stable covariance update
        I_KH = np.eye(6) - K_kal @ H
        self._P = I_KH @ P_pred @ I_KH.T + K_kal @ R_k @ K_kal.T

        # 计算误差 (NED 坐标系)
        x_est, y_est, psi_est = float(self._x_hat[0]), float(self._x_hat[1]), float(self._x_hat[2])
        u_est, v_est, r_est = float(self._x_hat[3]), float(self._x_hat[4]), float(self._x_hat[5])

        pos_err = np.array([
            x_est - self._target_pos[0],
            y_est - self._target_pos[1],
        ], dtype=np.float64)
        e_psi = wrap_to_pi(psi_est - self._target_psi)

        # LQR 控制律: u = -K @ x_error
        x_error = np.array([pos_err[0], pos_err[1], e_psi, u_est, v_est, r_est])
        tau = -self._K @ x_error

        # 裁剪
        tau[0:2] = np.clip(tau[0:2], -p.max_force, p.max_force)
        tau[2] = np.clip(tau[2], -p.max_moment, p.max_moment)

        pos_err_norm = float(np.linalg.norm(pos_err))
        from .controllers import compute_total_risk
        risk = compute_total_risk(pos_err_norm, 0.0, 0.0, 0.0, 0.0)

        elapsed = (time.perf_counter() - tic) * 1000.0
        self._last_diagnostics = {
            "solver_status": "lqg_lqr_kalman",
            "solver_success": True,
            "solve_time_ms": elapsed,
            "objective_value": pos_err_norm * pos_err_norm,
            "constraint_violation": 0.0,
            "risk_total": risk,
            "risk_ice": 0.0,
            "risk_position": min(1.0, pos_err_norm / 15.0),
            "risk_cvar": 0.0,
            "risk_model_status": "lqg",
            "cbf_active": False,
            "cbf_status": "not_available",
            "cbf_slack": 0.0,
            "cvar_alpha": 0.0,
            "cvar_sample_count": 0,
            "cvar_quantile": 0.0,
            "cvar_tail_sample_count": 0,
            "input_constraint_active": False,
        }

        return ControllerResult(tau=tau, feasible=True, mode="lqg", risk=risk,
                               cost_estimate=pos_err_norm ** 2)

    def set_cvar_seed(self, seed: int) -> None:
        """No-op for controllers without stochastic components."""
        pass

    def reset(self) -> None:
        self._x_hat = np.zeros(6, dtype=np.float64)
        self._P = np.eye(6, dtype=np.float64) * 10.0
        self._last_diagnostics = {}


# ============================================================
# DOB-MPC 控制器 (Disturbance Observer + NMPC)
# ============================================================

@dataclass
class DOBMPCParams:
    """DOB-MPC 参数。"""
    # NMPC 参数 (与 NMPCParams 一致)
    N: int = 20
    dt_nmpc: float = 0.2
    mass: float = 500000.0
    Izz: float = 5e8
    Xu: float = 500.0
    Yv: float = 800.0
    Nr: float = 2e5
    Xu_abs: float = 200.0
    Yv_abs: float = 300.0
    Nr_abs: float = 5e4
    Q_pos: float = 100.0
    Q_heading: float = 50.0
    Q_vel: float = 1.0
    R_force: float = 0.001
    R_moment: float = 0.0001
    dR_force: float = 0.01
    dR_moment: float = 0.001
    max_force: float = 3000.0
    max_moment: float = 100000.0
    safe_radius: float = 10.0
    cbf_gamma: float = 1.0
    ice_crushing_strength_mpa: float = 0.0003
    ice_structure_factor: float = 0.45
    vessel_beam_m: float = 22.0
    vessel_length_m: float = 122.5
    waterline_angle_deg: float = 30.0
    # DOB 参数
    observer_alpha: float = 0.15
    solver_label: str = "dob_nmpc"


class DOBMPCController(BaseController):
    """基于扰动观测器的 NMPC 控制器。

    架构: NMPC + 扰动观测器 (DOB)
    - DOB 从动力学残差估计冰力 (不使用真值冰况)
    - NMPC 将 DOB 估计作为已知扰动前馈纳入预测模型
    - NMPC 求解器失败时回退到 LQR + DOB 前馈

    这是比纯 NMPC 更鲁棒的方案, 因为 DOB 可以捕捉
    NMPC 内部模型未建模的扰动 (如冰力模型误差)。
    """

    def __init__(self, params: Optional[DOBMPCParams] = None):
        super().__init__()
        self.params = params or DOBMPCParams()
        self._solver_label = self.params.solver_label
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_psi: float = 0.0
        self._raw_ice = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
        self._ice_est = dict(self._raw_ice)
        self._prev_tau = np.zeros(3, dtype=np.float64)
        self._rng = np.random.default_rng(2026)

        # 扰动观测器 (通过外部 set_observer_estimate 接口注入)
        self._has_observer_estimate = False
        self._observer_force_estimate = np.zeros(3, dtype=np.float64)

        # LQR 回退增益
        self._lqr_K = self._compute_lqr_fallback()

        # NMPC 求解器 (延迟构建, 需要 casadi)
        self._nmpc_solver = None
        self._nmpc_built = False

    def set_observer_estimate(self, ice_force_estimate: "NDArray[np.float64]") -> None:
        """接收来自外部扰动观测器的冰力估计。"""
        self._observer_force_estimate = np.asarray(ice_force_estimate, dtype=np.float64).reshape(3,)
        self._has_observer_estimate = True

    def _compute_lqr_fallback(self) -> NDArray:
        """计算 LQR 回退增益。"""
        p = self.params
        A = np.zeros((6, 6))
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0
        A[3, 3] = -p.Xu / p.mass
        A[4, 4] = -p.Yv / p.mass
        A[5, 5] = -p.Nr / p.Izz
        B = np.zeros((6, 3))
        B[3, 0] = 1.0 / p.mass
        B[4, 1] = 1.0 / p.mass
        B[5, 2] = 1.0 / p.Izz
        Q = np.diag([p.Q_pos, p.Q_pos, p.Q_heading, p.Q_vel, p.Q_vel, p.Q_vel])
        R = np.diag([p.R_force, p.R_force, p.R_moment])
        try:
            from scipy.linalg import solve_continuous_are
            P = solve_continuous_are(A, B, Q, R)
            return np.linalg.inv(R) @ B.T @ P
        except ImportError:
            K = np.zeros((3, 6))
            K[0, 0] = 180.0
            K[0, 3] = 90.0
            K[1, 1] = 180.0
            K[1, 4] = 90.0
            K[2, 2] = 650.0
            K[2, 5] = 280.0
            return K

    def _build_nmpc(self) -> None:
        """延迟构建 NMPC 求解器 (需要 casadi)。"""
        if self._nmpc_built:
            return
        try:
            import casadi as cs
        except ImportError:
            self._nmpc_built = True
            return

        p = self.params
        N = p.N
        dt = p.dt_nmpc

        param = cs.MX.sym("param", 17)
        X = cs.MX.sym("X", 6, N + 1)
        U = cs.MX.sym("U", 3, N)

        ice_c = param[3]
        ice_h = param[4]
        ice_v = param[5]
        ice_dir = param[6]

        def dynamics(xk, tau_k):
            psi = xk[2]
            u_body, v_body, r = xk[3], xk[4], xk[5]
            cpsi, spsi = cs.cos(psi), cs.sin(psi)
            xdot = cpsi * u_body - spsi * v_body
            ydot = spsi * u_body + cpsi * v_body

            speed_factor = 1.5 - 0.25 / (ice_v + 0.5)
            alpha_val = p.waterline_angle_deg * math.pi / 180.0
            alpha_clamped = cs.fmin(alpha_val, math.pi / 3.0)  # 与 ice_force_common._ANGLE_ALPHA_MAX 一致
            tan_approx = alpha_clamped + alpha_clamped * alpha_clamped * alpha_clamped / 3.0
            angle_factor = 1.0 + 0.3 * tan_approx
            base_f = p.ice_crushing_strength_mpa * 1e6 * ice_h * p.vessel_beam_m * p.ice_structure_factor * speed_factor * angle_factor * ice_c
            ice_fx = base_f * cs.cos(ice_dir)
            ice_fy = base_f * cs.sin(ice_dir)
            # 体坐标系冰力分量 + 完整力矩 (对齐 ice_force_common)
            ice_fx_body = cpsi * ice_fx + spsi * ice_fy
            ice_fy_body = -spsi * ice_fx + cpsi * ice_fy
            ice_mz = (0.18 * p.vessel_length_m * ice_fy_body
                      - 0.05 * p.vessel_beam_m * ice_fx_body)

            u_dot = (tau_k[0] + ice_fx - p.Xu * u_body - p.Xu_abs * cs.fabs(u_body) * u_body) / p.mass
            v_dot = (tau_k[1] + ice_fy - p.Yv * v_body - p.Yv_abs * cs.fabs(v_body) * v_body) / p.mass
            r_dot = (tau_k[2] + ice_mz - p.Nr * r - p.Nr_abs * cs.fabs(r) * r) / p.Izz
            return cs.vertcat(xdot, ydot, r, u_dot, v_dot, r_dot)

        def rk4(xk, uk):
            k1 = dynamics(xk, uk)
            k2 = dynamics(xk + 0.5 * dt * k1, uk)
            k3 = dynamics(xk + 0.5 * dt * k2, uk)
            k4 = dynamics(xk + dt * k3, uk)
            return xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        obj = 0.0
        g_list = []
        lbg = []
        ubg = []

        g_list.append(X[:, 0] - param[10:16])
        lbg.extend([0.0] * 6)
        ubg.extend([0.0] * 6)

        for k in range(N):
            x_next = rk4(X[:, k], U[:, k])
            g_list.append(X[:, k + 1] - x_next)
            lbg.extend([0.0] * 6)
            ubg.extend([0.0] * 6)

            pos_err_x = X[0, k] - param[0]
            pos_err_y = X[1, k] - param[1]
            heading_err = 1.0 - cs.cos(X[2, k] - param[2])
            obj += p.Q_pos * (pos_err_x ** 2 + pos_err_y ** 2)
            obj += p.Q_heading * heading_err  # 不平方: (1-cos) 在 180° 为全局最大, 无虚假极小
            obj += p.Q_vel * (X[3, k] ** 2 + X[4, k] ** 2 + X[5, k] ** 2)
            obj += p.R_force * (U[0, k] ** 2 + U[1, k] ** 2)
            obj += p.R_moment * U[2, k] ** 2

            g_list.append(cs.sqrt(U[0, k] ** 2 + U[1, k] ** 2))
            lbg.append(0.0)
            ubg.append(p.max_force)
            g_list.append(cs.fabs(U[2, k]))
            lbg.append(0.0)
            ubg.append(p.max_moment)

        w = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
        nlp = {"f": obj, "x": w, "p": param, "g": cs.vertcat(*g_list)}
        opts = {"ipopt.print_level": 0, "ipopt.max_iter": 100, "ipopt.tol": 1e-3,
                "ipopt.max_wall_time": 5.0, "ipopt.max_cpu_time": 3.0,
                "print_time": 0, "verbose": False}
        try:
            import io, contextlib
            _devnull = io.StringIO()
            with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
                self._nmpc_solver = cs.nlpsol("dob_nmpc_solver", "ipopt", nlp, opts)
        except Exception:
            try:
                self._nmpc_solver = cs.nlpsol("dob_nmpc_solver", "sqpmethod", nlp, {"print_time": 0, "verbose": False})
            except Exception:
                self._nmpc_solver = None

        self._nmpc_N = N
        self._nmpc_lbg = lbg
        self._nmpc_ubg = ubg
        self._nmpc_built = True

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        self._target_pos = (float(x), float(y))
        self._target_psi = deg2rad(float(psi_deg))
        self.target_position = self._target_pos
        self.target_heading = self._target_psi

    def set_ice_conditions(self, ice_concentration: float, ice_thickness: float,
                          ice_drift_speed: float, ice_drift_direction: float = 0.0) -> None:
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
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="dob_nmpc", risk=0.0)

        state = np.asarray(state, dtype=np.float64).reshape(6,)
        p = self.params

        # 使用 DOB 估计的冰力作为前馈
        # C7 fix: NMPC 已包含参数化冰力模型, DOB 仅补偿模型误差
        # tau_ff = -gain * (observer_estimate - modelled_ice_estimate)
        tau_ff = np.zeros(3)
        if hasattr(self, '_has_observer_estimate') and self._has_observer_estimate:
            # Compute modelled ice force from current ice conditions
            ice = self._ice_est
            psi = float(state[2])
            cpsi, spsi = np.cos(psi), np.sin(psi)
            v = max(0.0, ice["drift_speed"])
            speed_factor = 1.0 + 0.5 * v / (v + 0.5) if v > 0 else 1.0
            alpha_val = p.waterline_angle_deg * math.pi / 180.0
            angle_factor = 1.0 + 0.3 * math.tan(min(alpha_val, math.pi / 3.0))
            base_f = p.ice_crushing_strength_mpa * 1e6 * ice["thickness"] * p.vessel_beam_m * p.ice_structure_factor * speed_factor * angle_factor * ice["concentration"]
            ice_dir_rad = math.radians(ice["drift_direction"])
            ice_fx_ned = base_f * math.cos(ice_dir_rad)
            ice_fy_ned = base_f * math.sin(ice_dir_rad)
            # Rotate to body frame
            ice_fx_body = cpsi * ice_fx_ned + spsi * ice_fy_ned
            ice_fy_body = -spsi * ice_fx_ned + cpsi * ice_fy_ned
            # 完整力矩 (对齐 ice_force_common: Mz = x_cp*Fy - y_cp*Fx)
            ice_mz_body = (0.18 * p.vessel_length_m * ice_fy_body
                           - 0.05 * p.vessel_beam_m * ice_fx_body)
            modelled_ice = np.array([ice_fx_body, ice_fy_body, ice_mz_body])
            # DOB compensates only the model error, not the full estimate
            ice_error = self._observer_force_estimate - modelled_ice
            tau_ff = -0.55 * ice_error  # 前馈增益

        # 尝试 NMPC 求解
        self._build_nmpc()
        feasible = True

        if self._nmpc_solver is not None:
            try:
                ice = self._ice_est
                param_val = np.array([
                    self._target_pos[0], self._target_pos[1], self._target_psi,
                    ice["concentration"], ice["thickness"], ice["drift_speed"],
                    math.radians(ice["drift_direction"]),  # deg → rad (canonical: ice_schedule.drift_dir_deg_to_rad)
                    self._prev_tau[0], self._prev_tau[1], self._prev_tau[2],
                    state[0], state[1], state[2], state[3], state[4], state[5],
                    p.safe_radius,
                ])

                x0 = state.copy()
                X_init = np.tile(x0.reshape(6, 1), (1, p.N + 1))
                U_init = np.zeros((3, p.N))
                w0 = np.concatenate([X_init.flatten(), U_init.flatten()])

                u_offset = 6 * (p.N + 1)
                lbw = np.full(len(w0), -np.inf)
                ubw = np.full(len(w0), np.inf)
                for k in range(p.N):
                    lbw[u_offset + 3 * k] = -p.max_force
                    ubw[u_offset + 3 * k] = p.max_force
                    lbw[u_offset + 3 * k + 1] = -p.max_force
                    ubw[u_offset + 3 * k + 1] = p.max_force
                    lbw[u_offset + 3 * k + 2] = -p.max_moment
                    ubw[u_offset + 3 * k + 2] = p.max_moment

                import io, contextlib
                _devnull = io.StringIO()
                with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
                    sol = self._nmpc_solver(x0=w0, p=param_val, lbx=lbw, ubx=ubw,
                                            lbg=np.array(self._nmpc_lbg), ubg=np.array(self._nmpc_ubg))
                w_opt = np.array(sol["x"]).flatten()
                tau = w_opt[u_offset:u_offset + 3].copy()  # 广义力, 由 sim_loop 统一分配
            except Exception:
                tau = self._fallback_lqr(state)
                feasible = False
        else:
            tau = self._fallback_lqr(state)
            feasible = False

        # 加入 DOB 前馈
        tau += tau_ff

        tau[0:2] = np.clip(tau[0:2], -p.max_force, p.max_force)
        tau[2] = np.clip(tau[2], -p.max_moment, p.max_moment)
        self._prev_tau = tau.copy()

        pos_err = float(np.linalg.norm(state[:2] - np.array(self._target_pos)))
        from .controllers import compute_total_risk, _ice_risk_standardized
        ice_risk = _ice_risk_standardized(self._ice_est["concentration"], self._ice_est["thickness"], self._ice_est["drift_speed"])
        risk = compute_total_risk(
            pos_err, self._ice_est["concentration"], self._ice_est["thickness"], self._ice_est["drift_speed"]
        )

        elapsed = (time.perf_counter() - tic) * 1000.0
        self._last_diagnostics = {
            "solver_status": "dob_nmpc" if feasible else "dob_nmpc_fallback_lqr",
            "solver_success": feasible,
            "solve_time_ms": elapsed,
            "objective_value": float(tau @ tau),
            "constraint_violation": 0.0,
            "risk_total": risk,
            "risk_ice": ice_risk,
            "risk_position": min(1.0, pos_err / 15.0),
            "risk_cvar": 0.0,
            "risk_model_status": "dob_nmpc",
            "cbf_active": False,
            "cbf_status": "not_available",
            "cbf_slack": 0.0,
            "cvar_alpha": 0.0,
            "cvar_sample_count": 0,
            "cvar_quantile": 0.0,
            "cvar_tail_sample_count": 0,
            "input_constraint_active": False,
        }

        return ControllerResult(tau=tau, feasible=feasible, mode="dob_nmpc", risk=risk,
                               cost_estimate=pos_err ** 2)

    def _fallback_lqr(self, state: NDArray) -> NDArray:
        """LQR 回退控制 + DOB 前馈。"""
        if self._target_pos is None:
            return np.zeros(3)
        p = self.params
        pos_err = np.array([state[0] - self._target_pos[0], state[1] - self._target_pos[1]])
        e_psi = wrap_to_pi(state[2] - self._target_psi)
        x_error = np.array([pos_err[0], pos_err[1], e_psi, state[3], state[4], state[5]])
        tau = -self._lqr_K @ x_error
        tau[0:2] = np.clip(tau[0:2], -p.max_force, p.max_force)
        tau[2] = np.clip(tau[2], -p.max_moment, p.max_moment)
        return tau

    def reset(self) -> None:
        self._prev_tau = np.zeros(3, dtype=np.float64)
        self._ice_est = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
        self._has_observer_estimate = False
        self._last_diagnostics = {}

# ============================================================
# ADRC controller (proxy strong baseline)
# ============================================================

@dataclass
class ADRCParams:
    """Active disturbance rejection control parameters.

    This is a lightweight proxy-scale ADRC baseline for the SCI1 benchmark:
    a PD nominal law plus an extended-state-observer-style total-disturbance
    estimate. It is not advertised as a fully tuned marine ADRC design.
    """
    kp_pos: float = 260.0
    kd_pos: float = 120.0
    kp_heading: float = 850.0
    kd_heading: float = 360.0
    observer_gain: float = 0.18
    disturbance_compensation: float = 0.75
    max_force: float = 3000.0
    # 船舶物理参数 (与 VesselParams 一致)
    mass: float = 500000.0
    Izz: float = 5e8
    Xu: float = 500.0
    Yv: float = 800.0
    Nr: float = 2e5
    Xu_abs: float = 200.0
    Yv_abs: float = 300.0
    Nr_abs: float = 5e4
    max_moment: float = 100000.0
    solver_label: str = "adrc_proxy"


class ADRCController(BaseController):
    """Proxy-scale ADRC baseline with an ESO-like disturbance estimate."""

    def __init__(self, params: Optional[ADRCParams] = None):
        super().__init__()
        self.params = params or ADRCParams()
        self._solver_label = self.params.solver_label
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_psi: float = 0.0
        self._last_state: Optional[NDArray[np.float64]] = None
        self._last_tau = np.zeros(3, dtype=np.float64)
        self._disturbance_hat = np.zeros(3, dtype=np.float64)
        self._raw_ice = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        self._target_pos = (float(x), float(y))
        self._target_psi = deg2rad(float(psi_deg))
        self.target_position = self._target_pos
        self.target_heading = self._target_psi

    def set_ice_conditions(self, ice_concentration: float, ice_thickness: float,
                          ice_drift_speed: float, ice_drift_direction: float = 0.0) -> None:
        self._raw_ice = {
            "concentration": float(np.clip(ice_concentration, 0.0, 1.0)),
            "thickness": max(0.0, float(ice_thickness)),
            "drift_speed": max(0.0, float(ice_drift_speed)),
            "drift_direction": float(ice_drift_direction),
        }

    def compute_control(self, state: NDArray[np.float64], reference: Optional[Dict[str, Any]] = None,
                        environment: Optional[Dict[str, Any]] = None, **kwargs) -> ControllerResult:
        tic = time.perf_counter()
        dt = float(kwargs.get("dt", 0.1))
        state = np.asarray(state, dtype=np.float64).reshape(6,)
        p = self.params
        if self._target_pos is None:
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="adrc", risk=0.0)

        # ESO-like residual estimate from velocity-rate mismatch. This is a
        # proxy estimator for benchmark comparison, not a certified ADRC proof.
        # H1 fix: subtract known model terms (including damping) so the ESO
        # estimates only the unknown disturbance, not the stabilizing damping.
        if self._last_state is not None and dt > 1e-9:
            dv = (state[3:6] - self._last_state[3:6]) / dt
            # Predicted acceleration = (tau - damping) / inertia
            u_last = float(self._last_state[3])
            v_last = float(self._last_state[4])
            r_last = float(self._last_state[5])
            predicted = np.array([
                (self._last_tau[0] - p.Xu * u_last - p.Xu_abs * abs(u_last) * u_last) / p.mass,
                (self._last_tau[1] - p.Yv * v_last - p.Yv_abs * abs(v_last) * v_last) / p.mass,
                (self._last_tau[2] - p.Nr * r_last - p.Nr_abs * abs(r_last) * r_last) / p.Izz,
            ])
            residual_acc = dv - predicted
            residual_tau = np.array([residual_acc[0] * p.mass, residual_acc[1] * p.mass, residual_acc[2] * p.Izz])
            self._disturbance_hat = (1.0 - p.observer_gain) * self._disturbance_hat + p.observer_gain * residual_tau

        pos_err = np.array([self._target_pos[0] - state[0], self._target_pos[1] - state[1]], dtype=np.float64)
        heading_err = wrap_to_pi(self._target_psi - state[2])
        yaw_rate = state[5]

        # 将速度转换到 NED 帧，与位置误差在同一坐标系下计算 PD 力
        psi = float(state[2])
        cpsi, spsi = np.cos(psi), np.sin(psi)
        vel_ned = np.array([cpsi * state[3] - spsi * state[4],
                            spsi * state[3] + cpsi * state[4]])
        fx_ned = p.kp_pos * pos_err[0] - p.kd_pos * vel_ned[0]
        fy_ned = p.kp_pos * pos_err[1] - p.kd_pos * vel_ned[1]
        # NED 力旋转到体坐标系
        fx = cpsi * fx_ned + spsi * fy_ned
        fy = -spsi * fx_ned + cpsi * fy_ned
        mz = p.kp_heading * heading_err - p.kd_heading * yaw_rate
        tau = np.array([fx, fy, mz], dtype=np.float64)
        tau -= p.disturbance_compensation * self._disturbance_hat
        # ESO 使用裁剪前的 tau (避免将饱和误差当作扰动)
        self._last_state = state.copy()
        self._last_tau = tau.copy()
        tau[:2] = np.clip(tau[:2], -p.max_force, p.max_force)
        tau[2] = np.clip(tau[2], -p.max_moment, p.max_moment)

        err_norm = float(np.linalg.norm(pos_err))
        from .controllers import compute_total_risk, _ice_risk_standardized
        ice_risk = _ice_risk_standardized(self._raw_ice.get("concentration", 0.0), self._raw_ice.get("thickness", 0.0), self._raw_ice.get("drift_speed", 0.0))
        risk = compute_total_risk(err_norm, self._raw_ice.get("concentration", 0.0), self._raw_ice.get("thickness", 0.0), self._raw_ice.get("drift_speed", 0.0), 0.0)
        elapsed = (time.perf_counter() - tic) * 1000.0
        self._last_diagnostics = {
            "solver_status": "adrc_proxy_eso",
            "solver_success": True,
            "solve_time_ms": elapsed,
            "objective_value": err_norm * err_norm,
            "constraint_violation": 0.0,
            "risk_total": risk,
            "risk_ice": ice_risk,
            "risk_position": min(1.0, err_norm / 15.0),
            "risk_cvar": 0.0,
            "risk_model_status": "adrc",
            "cbf_active": False,
            "cbf_status": "not_available",
            "cbf_slack": 0.0,
            "cvar_alpha": 0.0,
            "cvar_sample_count": 0,
            "cvar_quantile": 0.0,
            "cvar_tail_sample_count": 0,
            "input_constraint_active": False,
            "adrc_disturbance_norm": float(np.linalg.norm(self._disturbance_hat)),
        }
        return ControllerResult(tau=tau, feasible=True, mode="adrc", risk=risk, cost_estimate=err_norm ** 2)

    def set_cvar_seed(self, seed: int) -> None:
        """No-op for controllers without stochastic components."""
        pass

    def reset(self) -> None:
        self._last_state = None
        self._last_tau = np.zeros(3, dtype=np.float64)
        self._disturbance_hat = np.zeros(3, dtype=np.float64)
        self._last_diagnostics = {}


# ============================================================
# Robust / Tube MPC proxy baselines
# ============================================================

@dataclass
class RobustMPCParams:
    """Lightweight robust MPC proxy parameters.

    The implementation uses a one-step receding-horizon approximation with
    tightened force/moment limits and an ice-disturbance margin. It is a runnable
    robust-MPC-style baseline, not a full invariant-tube MPC theorem.
    """
    kp_pos: float = 240.0
    kd_pos: float = 110.0
    kp_heading: float = 780.0
    kd_heading: float = 320.0
    max_force: float = 3000.0
    max_moment: float = 100000.0
    disturbance_margin: float = 0.20
    tube_margin: float = 0.0
    solver_label: str = "robust_mpc_proxy"


class RobustMPCController(BaseController):
    """Baseline: PD controller with conservatively tightened force/moment limits.

    IMPORTANT NAMING NOTE: This is NOT a true robust MPC. It does NOT implement
    uncertainty set propagation, tube invariance, or any robust feasibility guarantee.
    It is a PD controller whose saturation limits are scaled down by a
    `disturbance_margin` factor. The name "robust_mpc" is retained for backward
    compatibility with configuration files; in the paper and figures it should be
    referred to as "Conservative PD (tightened limits)."
    """

    def __init__(self, params: Optional[RobustMPCParams] = None, mode_name: str = "robust_mpc"):
        super().__init__()
        self.params = params or RobustMPCParams()
        self._solver_label = self.params.solver_label
        self._mode_name = mode_name
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_psi: float = 0.0
        self._raw_ice = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
        self._prev_tau = np.zeros(3, dtype=np.float64)

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        self._target_pos = (float(x), float(y))
        self._target_psi = deg2rad(float(psi_deg))
        self.target_position = self._target_pos
        self.target_heading = self._target_psi

    def set_ice_conditions(self, ice_concentration: float, ice_thickness: float,
                          ice_drift_speed: float, ice_drift_direction: float = 0.0) -> None:
        self._raw_ice = {
            "concentration": float(np.clip(ice_concentration, 0.0, 1.0)),
            "thickness": max(0.0, float(ice_thickness)),
            "drift_speed": max(0.0, float(ice_drift_speed)),
            "drift_direction": float(ice_drift_direction),
        }

    def _margin_scale(self) -> float:
        ice = self._raw_ice
        from .controllers import _ice_risk_standardized
        risk = _ice_risk_standardized(ice["concentration"], ice["thickness"], ice["drift_speed"])
        return float(np.clip(1.0 - self.params.disturbance_margin * risk - self.params.tube_margin, 0.45, 1.0))

    def compute_control(self, state: NDArray[np.float64], reference: Optional[Dict[str, Any]] = None,
                        environment: Optional[Dict[str, Any]] = None, **kwargs) -> ControllerResult:
        tic = time.perf_counter()
        state = np.asarray(state, dtype=np.float64).reshape(6,)
        p = self.params
        if self._target_pos is None:
            return ControllerResult(tau=np.zeros(3), feasible=True, mode=self._mode_name, risk=0.0)
        pos_err = np.array([self._target_pos[0] - state[0], self._target_pos[1] - state[1]], dtype=np.float64)
        heading_err = wrap_to_pi(self._target_psi - state[2])
        # NED 帧计算 PD 力, 再旋转到体坐标系 (与 ADRC 修复一致)
        psi = float(state[2])
        cpsi, spsi = np.cos(psi), np.sin(psi)
        vel_ned = np.array([cpsi * state[3] - spsi * state[4],
                            spsi * state[3] + cpsi * state[4]])
        fx_ned = p.kp_pos * pos_err[0] - p.kd_pos * vel_ned[0]
        fy_ned = p.kp_pos * pos_err[1] - p.kd_pos * vel_ned[1]
        fx = cpsi * fx_ned + spsi * fy_ned
        fy = -spsi * fx_ned + cpsi * fy_ned
        tau_nom = np.array([
            fx,
            fy,
            p.kp_heading * heading_err - p.kd_heading * state[5],
        ], dtype=np.float64)
        # Receding-horizon proxy smoothing: penalize fast moves as a dU term.
        tau = 0.72 * tau_nom + 0.28 * self._prev_tau
        scale = self._margin_scale()
        tau[:2] = np.clip(tau[:2], -p.max_force * scale, p.max_force * scale)
        tau[2] = np.clip(tau[2], -p.max_moment * scale, p.max_moment * scale)
        self._prev_tau = tau.copy()
        err_norm = float(np.linalg.norm(pos_err))
        from .controllers import compute_total_risk, _ice_risk_standardized
        risk = compute_total_risk(err_norm, self._raw_ice.get("concentration", 0.0), self._raw_ice.get("thickness", 0.0), self._raw_ice.get("drift_speed", 0.0), 0.0)
        elapsed = (time.perf_counter() - tic) * 1000.0
        self._last_diagnostics = {
            "solver_status": self._mode_name + "_tightened_proxy",
            "solver_success": True,
            "solve_time_ms": elapsed,
            "objective_value": float(err_norm * err_norm + 1e-5 * tau @ tau),
            "constraint_violation": 0.0,
            "risk_total": risk,
            "risk_ice": _ice_risk_standardized(self._raw_ice.get("concentration", 0.0), self._raw_ice.get("thickness", 0.0), self._raw_ice.get("drift_speed", 0.0)),
            "risk_position": min(1.0, err_norm / 15.0),
            "risk_cvar": 0.0,
            "risk_model_status": self._mode_name,
            "cbf_active": False,
            "cbf_status": "not_available",
            "cbf_slack": 0.0,
            "cvar_alpha": 0.0,
            "cvar_sample_count": 0,
            "cvar_quantile": 0.0,
            "cvar_tail_sample_count": 0,
            "input_constraint_active": False,
            "constraint_tightening_scale": scale,
            "tube_margin": p.tube_margin,
        }
        return ControllerResult(tau=tau, feasible=True, mode=self._mode_name, risk=risk, cost_estimate=err_norm ** 2)

    def set_cvar_seed(self, seed: int) -> None:
        """No-op for controllers without stochastic components."""
        pass

    def reset(self) -> None:
        self._prev_tau = np.zeros(3, dtype=np.float64)
        self._last_diagnostics = {}


class TubeMPCController(RobustMPCController):
    """Baseline: PD controller with an additional static `tube_margin` constraint tightening.

    IMPORTANT NAMING NOTE: This is NOT a tube MPC. It applies a fixed-factor
    tightening to the force/moment limits (on top of RobustMPCController's margins).
    A true Tube MPC requires: bounded disturbance set, invariant tube (RPI set),
    tube-based constraint tightening from disturbance bounds, and a nominal
    controller within the tube. None of these are implemented.

    In the paper, refer to this as "Margin PD (double-tightened limits)" or
    "Double-Margin PD" to avoid misleading MPC reviewers.
    The code name "tube_mpc" is retained for backward YAML compatibility."""

    def __init__(self, params: Optional[RobustMPCParams] = None):
        p = params or RobustMPCParams(tube_margin=0.12, solver_label="tube_mpc_proxy")
        if p.tube_margin <= 0.0:
            p.tube_margin = 0.12
        super().__init__(p, mode_name="tube_mpc")


# ============================================================
# LESO-ADRC (Linear Extended State Observer ADRC)
# ============================================================

@dataclass
class LESOADRCParams:
    """Linear Extended State Observer ADRC parameters.

    Implements a standard LESO-based ADRC for the surge/sway channels
    with a PD-like yaw controller. The LESO estimates total disturbance
    (model mismatch + external forces) which is then cancelled in the
    control law.

    Reference: Gao (2006) Scaling and bandwidth-parameterization based
    controller tuning; Han (1998) From PID to Active Disturbance
    Rejection Control.
    """
    # Observer bandwidth (higher = faster estimation, more noise-sensitive)
    omega_o: float = 5.0        # rad/s
    # Controller bandwidth (higher = faster tracking)
    omega_c: float = 2.0        # rad/s
    # Plant gain estimate: tau/mass for surge/sway, tau_moment/Izz for yaw
    b0_x: float = 1.0 / 500000.0   # 1/mass for surge
    b0_y: float = 1.0 / 500000.0   # 1/mass for sway
    b0_psi: float = 1.0 / 5e8      # 1/Izz for yaw

    max_force: float = 3000.0
    max_moment: float = 100000.0
    solver_label: str = "leso_adrc"


class LESOADRCController(BaseController):
    r"""Linear Extended State Observer ADRC controller.

    Standard third-order LESO for each channel (x, y, psi):
        z1_dot = z2 + 3*omega_o*(y - z1)
        z2_dot = z3 + b0*u + 3*omega_o^2*(y - z1)
        z3_dot = omega_o^3*(y - z1)

    Control law:
        u0 = omega_c^2 * (r - z1) - 2*omega_c * z2
        u = (u0 - z3) / b0

    This is a proper linear ADRC implementation, not a proxy.
    """

    def __init__(self, params: Optional[LESOADRCParams] = None):
        super().__init__()
        self.params = params or LESOADRCParams()
        self._solver_label = self.params.solver_label
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_psi: float = 0.0

        # LESO states: [z1, z2, z3] for each channel
        self._z_x = np.zeros(3, dtype=np.float64)
        self._z_y = np.zeros(3, dtype=np.float64)
        self._z_psi = np.zeros(3, dtype=np.float64)
        self._initialized = False

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        self._target_pos = (float(x), float(y))
        self._target_psi = deg2rad(float(psi_deg))
        self.target_position = self._target_pos
        self.target_heading = self._target_psi

    def set_ice_conditions(self, ice_concentration: float, ice_thickness: float,
                          ice_drift_speed: float, ice_drift_direction: float = 0.0) -> None:
        pass  # ADRC estimates disturbance internally

    def set_safe_region_radius(self, radius: float) -> None:
        pass

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
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="leso_adrc", risk=0.0)

        state = np.asarray(state, dtype=np.float64).reshape(6,)
        p = self.params

        # Initialize LESO states on first call
        if not self._initialized:
            psi0 = float(state[2])
            c0, s0 = np.cos(psi0), np.sin(psi0)
            vx_ned0 = c0 * state[3] - s0 * state[4]
            vy_ned0 = s0 * state[3] + c0 * state[4]
            self._z_x = np.array([state[0], vx_ned0, 0.0])
            self._z_y = np.array([state[1], vy_ned0, 0.0])
            self._z_psi = np.array([state[2], state[5], 0.0])
            self._initialized = True

        # LESO update (3rd order for each channel)
        o = p.omega_o
        o2 = o * o
        o3 = o2 * o

        # LESO: z1_dot = z2 + 3*o*(y-z1), z2_dot = z3 + b0*u + 3*o^2*(y-z1), z3_dot = o^3*(y-z1)
        # Note: b0*u is added AFTER the control law computation (lines ~1035)

        # x channel
        y_x = state[0]
        e_x = y_x - self._z_x[0]
        self._z_x[0] += dt * (self._z_x[1] + 3 * o * e_x)
        self._z_x[1] += dt * (self._z_x[2] + 3 * o2 * e_x)  # b0*u added later
        self._z_x[2] += dt * (o3 * e_x)

        # y channel
        y_y = state[1]
        e_y = y_y - self._z_y[0]
        self._z_y[0] += dt * (self._z_y[1] + 3 * o * e_y)
        self._z_y[1] += dt * (self._z_y[2] + 3 * o2 * e_y)  # b0*u added later
        self._z_y[2] += dt * (o3 * e_y)

        # psi channel (wrap heading innovation to [-pi, pi])
        y_psi = state[2]
        e_psi_obs = wrap_to_pi(y_psi - self._z_psi[0])
        self._z_psi[0] += dt * (self._z_psi[1] + 3 * o * e_psi_obs)
        self._z_psi[0] = wrap_to_pi(self._z_psi[0])  # 防止长期漂移
        self._z_psi[1] += dt * (self._z_psi[2] + 3 * o2 * e_psi_obs)  # b0*u added later
        self._z_psi[2] += dt * (o3 * e_psi_obs)

        # Control law: u0 = omega_c^2 * (r - z1) - 2*omega_c * z2
        #              u = (u0 - z3) / b0
        c = p.omega_c
        c2 = c * c

        # Reference errors (NED frame)
        r_x = self._target_pos[0]
        r_y = self._target_pos[1]
        r_psi = self._target_psi

        u0_x = c2 * (r_x - self._z_x[0]) - 2 * c * self._z_x[1]
        u0_y = c2 * (r_y - self._z_y[0]) - 2 * c * self._z_y[1]
        u0_psi = c2 * wrap_to_pi(r_psi - self._z_psi[0]) - 2 * c * self._z_psi[1]

        # Disturbance cancellation
        fx_ned = (u0_x - self._z_x[2]) / max(p.b0_x, 1e-12)
        fy_ned = (u0_y - self._z_y[2]) / max(p.b0_y, 1e-12)
        mz = (u0_psi - self._z_psi[2]) / max(p.b0_psi, 1e-12)

        # Rotate NED forces to body frame
        psi = float(state[2])
        cpsi, spsi = np.cos(psi), np.sin(psi)
        fx = cpsi * fx_ned + spsi * fy_ned
        fy = -spsi * fx_ned + cpsi * fy_ned

        tau = np.array([fx, fy, mz], dtype=np.float64)
        tau[0:2] = np.clip(tau[0:2], -p.max_force, p.max_force)
        tau[2] = np.clip(tau[2], -p.max_moment, p.max_moment)

        # Update LESO with the control input in the same frame as the LESO
        # states.  x/y LESOs run in NED coordinates, so use fx_ned/fy_ned.
        # Using body-frame fx/fy here would reintroduce the frame-mixing bug
        # that the ADRC baseline audit found.
        self._z_x[1] += dt * p.b0_x * fx_ned
        self._z_y[1] += dt * p.b0_y * fy_ned
        self._z_psi[1] += dt * p.b0_psi * mz

        pos_err = float(np.linalg.norm(state[:2] - np.array(self._target_pos)))
        from .controllers import compute_total_risk
        risk = compute_total_risk(pos_err, 0.0, 0.0, 0.0)

        elapsed = (time.perf_counter() - tic) * 1000.0
        dist_norm = float(np.sqrt(self._z_x[2]**2 + self._z_y[2]**2 + self._z_psi[2]**2))
        self._last_diagnostics = {
            "solver_status": "leso_adrc",
            "solver_success": True,
            "solve_time_ms": elapsed,
            "objective_value": pos_err * pos_err,
            "constraint_violation": 0.0,
            "risk_total": risk,
            "risk_ice": 0.0,
            "risk_position": min(1.0, pos_err / 15.0),
            "risk_cvar": 0.0,
            "risk_model_status": "leso_adrc",
            "cbf_active": False,
            "cbf_status": "not_available",
            "cbf_slack": 0.0,
            "cvar_alpha": 0.0,
            "cvar_sample_count": 0,
            "cvar_quantile": 0.0,
            "cvar_tail_sample_count": 0,
            "input_constraint_active": False,
            "leso_disturbance_norm": dist_norm,
            "leso_omega_o": p.omega_o,
            "leso_omega_c": p.omega_c,
        }
        return ControllerResult(tau=tau, feasible=True, mode="leso_adrc", risk=risk,
                               cost_estimate=pos_err ** 2)

    def set_cvar_seed(self, seed: int) -> None:
        """No-op for controllers without stochastic components."""
        pass

    def reset(self) -> None:
        self._z_x = np.zeros(3, dtype=np.float64)
        self._z_y = np.zeros(3, dtype=np.float64)
        self._z_psi = np.zeros(3, dtype=np.float64)
        self._initialized = False
        self._last_diagnostics = {}
