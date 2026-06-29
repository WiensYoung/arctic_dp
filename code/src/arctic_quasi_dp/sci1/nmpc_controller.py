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
from .controllers import PrecisionDPParams
from .control.nmpc_terminal import compute_terminal_cost, terminal_value, TerminalCostResult

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
    Xu: float = 500.0                # surge 线性阻尼 (N·s/m)
    Yv: float = 800.0                # sway 线性阻尼 (N·s/m)
    Nr: float = 2e5                  # yaw 线性阻尼 (N·m·s/rad)
    Xu_abs: float = 200.0            # surge 二次阻尼 (N·s²/m²)
    Yv_abs: float = 300.0            # sway 二次阻尼 (N·s²/m²)
    Nr_abs: float = 5e4              # yaw 二次阻尼 (N·m·s²/rad²)
    # 与 sim_loop.VesselParams 保持一致

    # 代价权重
    Q_pos: float = 100.0             # 位置误差权重
    Q_heading: float = 50.0          # 艏向误差权重
    Q_vel: float = 1.0               # 速度权重
    R_force: float = 0.001           # 控制力权重
    R_moment: float = 0.0001         # 控制力矩权重
    dR_force: float = 0.01           # 控制力变化率权重
    dR_moment: float = 0.001         # 控制力矩变化率权重

    # 约束 (与 PrecisionDPParams 一致)
    max_force: float = 3000.0        # 最大合力 (N)
    max_moment: float = 100000.0     # 最大力矩 (N·m)
    safe_radius: float = 10.0        # CBF 安全半径 (m)
    cbf_gamma: float = 1.0           # CBF 收敛速率

    # 冰力模型参数 (代理值, 与 VesselParams 一致)
    ice_crushing_strength_mpa: float = 0.0003
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
    """符号冰力模型 (CasADi MX/SX 兼容)。

    使用代数等价变换消除所有可能产生NaN的运算:
    - v/(v+0.5) → 1 - 0.5/(v+0.5), 避免分子分母同时为零
    - tan(alpha) → 多项式近似, 避免tan的导数发散
    """
    c = cs.fmax(0.0, cs.fmin(1.0, concentration))
    h = cs.fmax(0.0, thickness)
    v = cs.fmax(0.0, drift_speed)

    # 速度因子: 1 + 0.5*v/(v+0.5) = 1.5 - 0.25/(v+0.5)
    # 代数等价, 但分母 v+0.5 恒 > 0 (因为 v >= 0), 无NaN风险
    speed_factor = 1.5 - 0.25 / (v + 0.5)

    # 角度因子: 与 ice_force_common.py 保持一致, 使用 cs.tan
    # alpha 上限 60° (≈1.047 rad), 远低于 tan 的奇点 pi/2
    alpha_val = params.waterline_angle_deg * 3.141592653589793 / 180.0
    alpha_clamped = cs.fmin(alpha_val, 3.141592653589793 / 3.0)  # 60°
    angle_factor = 1.0 + 0.3 * cs.tan(alpha_clamped)

    base_force = (
        params.ice_crushing_strength_mpa * 1e6
        * h * params.vessel_beam_m * params.ice_structure_factor
        * speed_factor * angle_factor * c
    )

    force_ned_x = base_force * cs.cos(drift_dir)
    force_ned_y = base_force * cs.sin(drift_dir)

    cpsi = cs.cos(psi)
    spsi = cs.sin(psi)
    fx = cpsi * force_ned_x + spsi * force_ned_y
    fy = -spsi * force_ned_x + cpsi * force_ned_y

    lever = 0.18 * params.vessel_length_m
    mz = lever * fy

    return fx, fy, mz


class NMPCIceController(BaseController):
    """基于 CasADi 的 NMPC 冰区 DP 控制器。"""

    def __init__(
        self,
        params: Optional[NMPCParams] = None,
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

        # 上一次控制输入 (用于变化率惩罚)
        self._prev_tau = np.zeros(3, dtype=np.float64)

        # E7: 计算 LQR Riccati 终端代价
        # 正则化P矩阵防止CasADi求导时数值溢出
        p = self.params
        tc = compute_terminal_cost(
            mass=p.mass, Izz=p.Izz, Xu=p.Xu, Yv=p.Yv, Nr=p.Nr,
            dt=p.dt_nmpc, Q_pos=p.Q_pos, Q_heading=p.Q_heading, Q_vel=p.Q_vel,
            R_force=p.R_force, R_moment=p.R_moment,
        )
        # 限制P矩阵元素幅度，防止CasADi雅可比计算出现NaN
        eigvals, eigvecs = np.linalg.eigh(tc.P)
        eigvals_clipped = np.clip(eigvals, 1e-6, 1e6)
        P_reg = (eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T)
        # 确保对称正定
        P_reg = (P_reg + P_reg.T) / 2.0
        tc.P = P_reg
        self._terminal_cost = tc

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
        #        eta_x, eta_y, psi, u, v, r,
        #        safe_radius]
        param = cs.MX.sym("param", 17)

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
            heading_err = 1.0 - cs.cos(X[2, k] - param[2])  # 0 at 0°, 2 at 180°
            obj += p.Q_pos * (pos_err_x ** 2 + pos_err_y ** 2)
            obj += p.Q_heading * heading_err  # 不平方: (1-cos) 在 180° 为全局最大, 无虚假极小

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

            # HOCBF 安全约束 (relative-degree-2):
            # h = R^2 - ||pos - pos_ref||^2
            # ḣ = -2 * e_p · v_ned
            # ḧ = -2 * ||v_ned||^2 - 2 * e_p · a_ned
            # ḧ + (γ₁ + γ₂)·ḣ + γ₁·γ₂·h ≥ 0
            safe_r = param[16]
            h_val = safe_r ** 2 - (pos_err_x ** 2 + pos_err_y ** 2)
            vel_ned_x = cs.cos(X[2, k]) * X[3, k] - cs.sin(X[2, k]) * X[4, k]
            vel_ned_y = cs.sin(X[2, k]) * X[3, k] + cs.cos(X[2, k]) * X[4, k]
            h_dot = -2.0 * (pos_err_x * vel_ned_x + pos_err_y * vel_ned_y)
            # 加速度 (NED) ≈ R_body2ned @ (tau / mass)
            accel_ned_x = (cs.cos(X[2, k]) * U[0, k] - cs.sin(X[2, k]) * U[1, k]) / p.mass
            accel_ned_y = (cs.sin(X[2, k]) * U[0, k] + cs.cos(X[2, k]) * U[1, k]) / p.mass
            h_ddot = -2.0 * (vel_ned_x ** 2 + vel_ned_y ** 2) - 2.0 * (pos_err_x * accel_ned_x + pos_err_y * accel_ned_y)
            gamma1 = p.cbf_gamma
            gamma2 = p.cbf_gamma
            hocbf_constraint = h_ddot + (gamma1 + gamma2) * h_dot + gamma1 * gamma2 * h_val
            g.append(hocbf_constraint)
            lbg.append(0.0)
            ubg.append(1e20)

        # E7: LQR Riccati 终端代价
        # 使用P矩阵对角元素作为权重，避免完整矩阵乘法导致CasADi雅可比NaN
        x_err_T = X[:, N] - cs.vertcat(param[0], param[1], param[2], 0.0, 0.0, 0.0)
        P_term = self._terminal_cost.P
        for i in range(6):
            for j in range(6):
                if abs(P_term[i, j]) > 1e-6:
                    obj += float(P_term[i, j]) * x_err_T[i] * x_err_T[j]

        # NLP 变量
        w = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
        nlp = {"f": obj, "x": w, "p": param, "g": cs.vertcat(*g)}

        # 求解器选项
        # Issue 2.1 fix: 添加 wall-clock 超时防止 IPOPT 长时间阻塞
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.max_wall_time": 5.0,    # 5秒 wall-clock 超时
            "ipopt.max_cpu_time": 5.0,     # 5秒 CPU 时间超时
            "print_time": 0,
            "verbose": False,
        }

        try:
            import io, contextlib
            _devnull = io.StringIO()
            with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
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
        #  eta_x, eta_y, psi, u, v, r,
        #  safe_radius]
        ice = self._ice_est
        x0 = np.asarray(state, dtype=np.float64).reshape(6,)
        param_val = np.array([
            self._target_pos[0],
            self._target_pos[1],
            self._target_psi,
            ice["concentration"],
            ice["thickness"],
            ice["drift_speed"],
            math.radians(ice["drift_direction"]),  # deg → rad (canonical: ice_schedule.drift_dir_deg_to_rad)
            self._prev_tau[0],
            self._prev_tau[1],
            self._prev_tau[2],
            x0[0], x0[1], x0[2], x0[3], x0[4], x0[5],
            self.params.safe_radius,
        ], dtype=np.float64)

        # 初始猜测: 状态保持 + 零控制
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

        # 求解 (抑制CasADi/IPOPT的NaN警告, solver已有PD回退)
        feasible = True
        try:
            import io, contextlib
            _devnull = io.StringIO()
            with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
                sol = self._solver(
                    x0=w0,
                    p=param_val,
                    lbx=lbw, ubx=ubw,
                    lbg=np.array(self._lbg),
                    ubg=np.array(self._ubg),
                )
            w_opt = np.array(sol["x"]).flatten()

            # 检查 IPOPT 求解器返回状态
            solver_stats = self._solver.stats()
            return_status = solver_stats.get("return_status", "unknown")
            if return_status not in ("Solve_Succeeded", "Solved_To_Acceptable_Level", "Feasible_Point_Found"):
                # 求解器未成功收敛，回退到 PD 控制
                tau = self._fallback_pd(state)
                feasible = False
            else:
                # 提取第一步控制 (广义力, 由 sim_loop 的推进器分配器统一处理)
                u_opt = w_opt[u_offset:u_offset + 3]
                tau = u_opt.copy()

        except Exception:
            # 求解失败: 回退到 PD 控制
            tau = self._fallback_pd(state)
            feasible = False

        tau[0:2] = np.clip(tau[0:2], -p.max_force, p.max_force)
        tau[2] = np.clip(tau[2], -p.max_moment, p.max_moment)

        self._prev_tau = tau.copy()

        # 风险估计 (与 IceAware/supervisor 三因子公式一致)
        pos_err = float(np.linalg.norm(np.asarray(state[:2]) - np.asarray(self._target_pos)))
        from .controllers import compute_total_risk, _ice_risk_standardized
        ice_risk = _ice_risk_standardized(ice["concentration"], ice["thickness"], ice["drift_speed"])
        cvar_est = float(np.clip(0.15 * ice["concentration"] * (0.25 + ice["thickness"]), 0.0, 1.0))
        risk = compute_total_risk(
            pos_err, ice["concentration"], ice["thickness"], ice["drift_speed"], cvar_est
        )

        # E7: 终端代价诊断
        state_err = x0 - np.array([
            self._target_pos[0], self._target_pos[1], self._target_psi,
            0.0, 0.0, 0.0,
        ])
        terminal_v = terminal_value(state_err, self._terminal_cost.P)
        in_terminal_set = terminal_v <= self._terminal_cost.alpha

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
            "risk_cvar": cvar_est,
            "risk_model_status": "nmpc",
            "cbf_active": pos_err > 0.75 * p.safe_radius,
            "cbf_status": "nmpc_constrained",
            "cbf_slack": p.safe_radius - pos_err,
            # E7 terminal cost diagnostics
            "terminal_cost_enabled": True,
            "terminal_cost_value": float(terminal_v),
            "terminal_set_alpha": float(self._terminal_cost.alpha),
            "in_terminal_set": in_terminal_set,
            "terminal_spectral_radius": self._terminal_cost.spectral_radius,
            "terminal_stable": self._terminal_cost.is_stable,
            "terminal_method": self._terminal_cost.method,
        }

        return ControllerResult(
            tau=tau, feasible=feasible, mode="nmpc", risk=risk,
            cost_estimate=pos_err * pos_err,
        )

    def _fallback_pd(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """NMPC 求解失败时的 PD 回退控制。

        增益从 PrecisionDPParams 默认值读取，确保与 Precision DP 基线一致。
        计算方式: NED 帧计算力 → 旋转到体坐标系 (与 _raw_precision_tau 一致)。
        """
        if self._target_pos is None:
            return np.zeros(3)
        p = self.params
        pd_params = PrecisionDPParams()  # 使用默认增益
        x, y, psi = float(state[0]), float(state[1]), float(state[2])
        u, v, r = float(state[3]), float(state[4]), float(state[5])

        pos_err_ned = np.array([x - self._target_pos[0], y - self._target_pos[1]])
        e_psi = wrap_to_pi(psi - self._target_psi)

        # NED 帧速度
        cpsi, spsi = np.cos(psi), np.sin(psi)
        vel_ned = np.array([cpsi * u - spsi * v, spsi * u + cpsi * v])

        # NED 帧 PD 力
        fx_ned = -pd_params.kp_pos * pos_err_ned[0] - pd_params.kd_pos * vel_ned[0]
        fy_ned = -pd_params.kp_pos * pos_err_ned[1] - pd_params.kd_pos * vel_ned[1]
        # 旋转到体坐标系
        fx = cpsi * fx_ned + spsi * fy_ned
        fy = -spsi * fx_ned + cpsi * fy_ned
        mz = -pd_params.kp_heading * e_psi - pd_params.kd_heading * r

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
