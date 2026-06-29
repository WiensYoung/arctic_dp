"""投稿级冰区 DP 增强控制器族。

这些控制器遵循 BaseController 接口，可直接被现有 Simulator 调用。
设计目标：
1) Precision DP 是主模式，不放松目标点；
2) Ice-aware Precision DP 使用观测/估计冰情，不直接读取未来真实冰载荷；
3) Risk-aware Quasi-DP 只作为推力饱和/高风险/不可行时的安全降级；
4) Ice-vaning/Escape 是极端冰况保护；
5) ModeSupervisedIceDPController 以 hysteresis 和 minimum dwell time 连接四种模式。

说明：本文件实现的是可复现实验基线/投稿 scaffold。若要冲 TOP 期刊，建议后续将
IceAwarePrecisionDPController 的解析控制律替换为 CasADi/acados NMPC，但保留相同接口和指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
import time
import math

import numpy as np
from numpy.typing import NDArray

from ..controllers.base import BaseController, ControllerResult
from ..utils.math_utils import deg2rad, wrap_to_pi
from .control.hocbf_activation import HOCBFActivationStateMachine, HOCBFActivationConfig, SafetyMode


class DPMode(IntEnum):
    PRECISION = 0
    ICE_AWARE = 1
    QUASI_DP = 2
    ESCAPE = 3


@dataclass
class PrecisionDPParams:
    kp_pos: float = 180.0
    kd_pos: float = 90.0
    ki_pos: float = 8.0       # 启用积分: 消除冰扰动下的稳态偏差
    kp_heading: float = 650.0
    kd_heading: float = 280.0
    ki_heading: float = 5.0   # 启用航向积分: 消除航向稳态偏差
    max_force: float = 3000.0      # 匹配推进器总容量 (~4 kN)
    max_moment: float = 100000.0   # 匹配推进器布局力臂 (~22 m × 4 kN)
    max_integral: float = 20.0
    solver_label: str = "precision_pd"


@dataclass
class IceAwareParams(PrecisionDPParams):
    ice_feedforward_gain: float = 0.80   # 提高到 80%: 更积极的冰力补偿, 减少未补偿扰动
    observer_alpha: float = 0.25         # 提高到 0.25: 更快的冰况跟踪, 代价是略多噪声
    observer_noise_std: float = 0.02
    observer_direction_noise_deg: float = 5.0  # 冰漂移方向观测噪声 (度)
    cvar_alpha: float = 0.90
    cvar_samples: int = 64
    cvar_sigma_force: float = 150.0
    cbf_radius: float = 10.0
    cbf_gain: float = 180.0   # 提高: 更积极的安全修正, 更好的边界保持
    vessel_mass_kg: float = 500000.0  # 用于 HOCBF 加速度估计
    thrust_margin_soft_limit: float = 0.72
    risk_gain: float = 0.75
    solver_label: str = "ice_aware_risk_pd"
    # --- 冰力模型参数 (代理值, 与 VesselParams 一致) ---
    ice_crushing_strength_mpa: float = 0.0003   # 代理值: 使中等冰况冰力 ~1 kN
    ice_structure_factor: float = 0.45          # Lindqvist 1989
    vessel_beam_m: float = 22.0                 # 船宽 (m)
    vessel_length_m: float = 122.5              # 船长 (m), 用于冰力力臂
    vessel_waterline_angle_deg: float = 30.0    # 水线角 (度)


@dataclass
class QuasiDPParams(IceAwareParams):
    watch_radius: float = 10.0
    relax_gain: float = 0.45   # 降低: 减少目标放松幅度, 保持更好的位置精度
    pos_weight_scale_inside: float = 0.35
    max_force: float = 2500.0
    solver_label: str = "quasi_dp_safety"


@dataclass
class EscapeParams(IceAwareParams):
    safe_retreat_force: float = 1500.0
    heading_to_ice_weight: float = 1.0
    max_force: float = 2000.0
    solver_label: str = "ice_vaning_escape"


@dataclass
class SupervisorParams:
    ice_enter: float = 0.28           # 进入 ICE_AWARE 的风险阈值
    ice_exit: float = 0.20            # 退出 ICE_AWARE 的风险阈值
    high_risk_enter: float = 0.70     # 提高: 减少不必要的 QUASI_DP 切换
    high_risk_exit: float = 0.50      # 提高: 与 high_risk_enter 保持滞后
    extreme_risk_enter: float = 0.88  # 提高: 只在真正极端时进入 ESCAPE
    extreme_risk_exit: float = 0.72   # 提高: 与 extreme_risk_enter 保持滞后
    pos_error_quasi_enter: float = 12.0  # 提高: 位置误差更大才降级
    pos_error_quasi_exit: float = 8.0    # 提高: 与 enter 保持更大滞后
    dwell_time: float = 5.0           # 降低: 允许更快的模式恢复
    solver_label: str = "mode_supervised_ice_dp"


def _rotation_body_from_ned(psi: float) -> NDArray[np.float64]:
    c = float(np.cos(psi))
    s = float(np.sin(psi))
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def _clip_tau(tau: NDArray[np.float64], max_force: float, max_moment: float) -> NDArray[np.float64]:
    tau = np.array(tau, dtype=np.float64, copy=True).reshape(3,)
    tau[0] = np.clip(tau[0], -max_force, max_force)
    tau[1] = np.clip(tau[1], -max_force, max_force)
    tau[2] = np.clip(tau[2], -max_moment, max_moment)
    return tau


def _ice_risk_standardized(concentration: float, thickness: float, drift_speed: float) -> float:
    """标准化冰风险计算 — 所有控制器共用，保证 supervisor 与子控制器一致。

    使用物理归一化的乘积形式，输出自然落在 [0, 1]：
    - concentration ∈ [0, 1] (已是无量纲)
    - thickness / H_REF，H_REF = 2.5m (多年冰典型厚度上限)
    - drift_speed / V_REF，V_REF = 1.0 m/s (快速冰漂移上限)

    典型值:
    - 开阔水域: 0.0
    - 中等冰况 (c=0.45, h=0.7, v=0.25): ~0.07
    - 重冰况 (c=0.72, h=1.25, v=0.5): ~0.29
    - 极端冰况 (c=0.86, h=1.55, v=0.65): ~0.51
    """
    H_REF = 2.5   # 参考冰厚 (m)
    V_REF = 1.0   # 参考漂移速度 (m/s)
    c = float(np.clip(concentration, 0.0, 1.0))
    h = max(0.0, float(thickness)) / H_REF
    v = max(0.0, float(drift_speed)) / V_REF
    return float(np.clip(c * h * (0.3 + v), 0.0, 1.0))


def compute_total_risk(
    pos_err: float,
    concentration: float,
    thickness: float,
    drift_speed: float,
    cvar: float = 0.0,
    max_pos_err: float = 15.0,
) -> float:
    """三因子风险公式 — 所有控制器共用。

    risk = 0.35 * pos_risk + 0.35 * ice_risk + 0.30 * cvar

    Args:
        pos_err: 位置误差 (m)
        concentration: 冰浓度 [0, 1]
        thickness: 冰厚 (m)
        drift_speed: 冰漂移速度 (m/s)
        cvar: CVaR 尾部风险 [0, 1]
        max_pos_err: 位置风险归一化参考值 (m)

    Returns:
        总风险 [0, 1]
    """
    pos_risk = min(1.0, pos_err / max_pos_err)
    ice_risk = _ice_risk_standardized(concentration, thickness, drift_speed)
    return float(np.clip(0.35 * pos_risk + 0.35 * ice_risk + 0.30 * cvar, 0.0, 1.0))


def _ice_force_lindqvist_proxy(
    concentration: float,
    thickness: float,
    drift_speed: float,
    drift_direction_deg: float,
    psi: float,
    crushing_strength_mpa: float = 0.0003,
    structure_factor: float = 0.45,
    vessel_beam_m: float = 22.0,
    vessel_length_m: float = 122.5,
    waterline_angle_deg: float = 30.0,
) -> NDArray[np.float64]:
    """基于 Lindqvist (1989) 简化的冰力代理模型。使用共享模块。"""
    from .ice_force_common import compute_ice_force_body
    return compute_ice_force_body(
        concentration=concentration,
        thickness=thickness,
        drift_speed=drift_speed,
        drift_direction_deg=drift_direction_deg,
        vessel_psi=psi,
        crushing_strength_mpa=crushing_strength_mpa,
        vessel_beam_m=vessel_beam_m,
        vessel_length_m=vessel_length_m,
        structure_factor=structure_factor,
        waterline_angle_rad=deg2rad(float(waterline_angle_deg)),
    )


class PrecisionDPController(BaseController):
    """高精度定位主模式：严格追踪目标点和艏向，不做准定位放松。"""

    def __init__(self, params: Optional[PrecisionDPParams] = None):
        super().__init__()
        self.params = params or PrecisionDPParams()
        self._solver_label = self.params.solver_label
        self._vel_filtered = np.zeros(2, dtype=np.float64)  # 低通滤波后的速度
        self._r_filtered = 0.0  # 低通滤波后的艏摇角速度
        self._vel_initialized = False  # 首次调用时直接使用真值
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_psi: float = 0.0
        self._int_pos = np.zeros(2, dtype=np.float64)
        self._int_psi = 0.0
        # 阻尼系数: 默认值与 VesselParams 一致, 可通过 set_vessel_params 覆盖
        self._damp_Xu = 500.0; self._damp_Yv = 800.0; self._damp_Nr = 200000.0
        self._damp_Xu_abs = 200.0; self._damp_Yv_abs = 300.0; self._damp_Nr_abs = 50000.0
        self._vessel_mass = 500000.0
        self._vessel_Izz = 5e8
        self._vessel_length = 122.5

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        self._target_pos = (float(x), float(y))
        self._target_psi = deg2rad(float(psi_deg))
        self.target_position = self._target_pos
        self.target_heading = self._target_psi
        self._int_pos[:] = 0.0
        self._int_psi = 0.0

    def set_safe_region_radius(self, radius: float) -> None:
        """设置安全区域半径，同步更新 CBF 参数。"""
        if hasattr(self.params, 'cbf_radius'):
            self.params.cbf_radius = float(radius)

    def set_vessel_params(self, vessel_params) -> None:
        """从 VesselParams 更新阻尼系数和船舶参数, 消除硬编码不匹配。"""
        self._damp_Xu = vessel_params.Xu
        self._damp_Yv = vessel_params.Yv
        self._damp_Nr = vessel_params.Nr
        self._damp_Xu_abs = vessel_params.Xu_abs
        self._damp_Yv_abs = vessel_params.Yv_abs
        self._damp_Nr_abs = vessel_params.Nr_abs
        self._vessel_mass = vessel_params.mass
        self._vessel_Izz = vessel_params.Izz
        self._vessel_length = vessel_params.length

    def _raw_precision_tau(self, state: NDArray[np.float64], target_pos: Tuple[float, float], target_psi: float, dt: float) -> NDArray[np.float64]:
        p = self.params
        x, y, psi = float(state[0]), float(state[1]), float(state[2])
        u, v, r = float(state[3]), float(state[4]), float(state[5])
        pos_err_ned = np.array([x - target_pos[0], y - target_pos[1]], dtype=np.float64)
        vel_ned = np.array([np.cos(psi) * u - np.sin(psi) * v, np.sin(psi) * u + np.cos(psi) * v])
        e_psi = wrap_to_pi(psi - target_psi)

        # 微分项低通滤波: 抑制速度噪声, 截止频率 ~1.5 Hz
        if not self._vel_initialized:
            self._vel_filtered = vel_ned.copy()
            self._r_filtered = r
            self._vel_initialized = True
        else:
            alpha_vel = min(1.0, 2.0 * dt)  # dt=0.1 时 alpha=0.2, 截止 ~1.5 Hz
            self._vel_filtered = (1.0 - alpha_vel) * self._vel_filtered + alpha_vel * vel_ned
            self._r_filtered = (1.0 - alpha_vel) * self._r_filtered + alpha_vel * r

        # 增益调度: 根据冰况风险动态调整增益
        # 轻冰 (risk < 0.2): 使用标称增益
        # 重冰 (risk > 0.5): 增益提高 40%, 提供更强的位置保持力
        _ice = getattr(self, '_ice_est', {})
        ice_risk = _ice_risk_standardized(
            _ice.get("concentration", 0.0),
            _ice.get("thickness", 0.0),
            _ice.get("drift_speed", 0.0),
        )
        gain_scale = 1.0 + 0.4 * min(1.0, ice_risk / 0.35)  # risk=0.35 时达到最大 1.4×
        kp = p.kp_pos * gain_scale
        kd = p.kd_pos * gain_scale
        kp_h = p.kp_heading * gain_scale
        kd_h = p.kd_heading * gain_scale

        # 积分抗饱和: 只在误差较小时累积积分, 防止饱和时积分继续增长
        pos_err_norm = float(np.linalg.norm(pos_err_ned))
        anti_windup_gate = 1.0 if pos_err_norm < 15.0 else 0.0  # 超过安全半径时停止积分
        self._int_pos = np.clip(self._int_pos + pos_err_ned * dt * anti_windup_gate, -p.max_integral, p.max_integral)
        self._int_psi = float(np.clip(self._int_psi + e_psi * dt, -p.max_integral, p.max_integral))

        # 计算力矩控制 (Computed Torque):
        # 1. PD 计算期望加速度: a_des = -(kp*e + kd*e_dot + ki*integral)
        # 2. 动力学补偿: tau = M*a_des + D*v (精确抵消已知动力学)
        # 3. 效果: 系统表现为纯双积分器, PD 控制器可精确调参
        M_pos = self._vessel_mass   # 船舶质量 (从 set_vessel_params 更新)
        M_yaw = self._vessel_Izz    # 偏航惯量 (从 set_vessel_params 更新)

        # 自适应死区: 大误差时增益倍增
        if pos_err_norm > 5.0:
            urgency = min(1.0, (pos_err_norm - 5.0) / 10.0)
            emergency_scale = 1.0 + urgency
            kp *= emergency_scale
            kd *= emergency_scale

        # 期望加速度 (NED 坐标系)
        a_des_ned = -(kp * pos_err_ned + kd * self._vel_filtered + p.ki_pos * self._int_pos)
        a_des_psi = -(kp_h * e_psi + kd_h * self._r_filtered + p.ki_heading * self._int_psi)

        # 动力学补偿: tau = M * a_des + D * v
        # 阻尼系数 (从 set_vessel_params 更新, 默认与 VesselParams 一致)
        damping_body = np.array([
            self._damp_Xu * u + self._damp_Xu_abs * abs(u) * u,
            self._damp_Yv * v + self._damp_Yv_abs * abs(v) * v,
            self._damp_Nr * r + self._damp_Nr_abs * abs(r) * r,
        ])

        # 计算力矩: 将期望加速度转换为体力
        force_ned = M_pos * a_des_ned
        force_body = _rotation_body_from_ned(psi) @ force_ned + damping_body[:2]
        moment = M_yaw * a_des_psi + damping_body[2]
        tau = _clip_tau(np.array([force_body[0], force_body[1], moment]), p.max_force, p.max_moment)

        # 积分回退: 如果输出饱和, 缩小积分项防止 windup
        if np.any(np.abs(tau[:2]) >= p.max_force * 0.99) or abs(tau[2]) >= p.max_moment * 0.99:
            self._int_pos *= 0.95  # 每步缩小 5%
            self._int_psi *= 0.95

        return tau

    def compute_control(self, state: NDArray[np.float64], reference: Optional[Dict[str, Any]] = None, environment: Optional[Dict[str, Any]] = None, **kwargs) -> ControllerResult:
        tic = time.perf_counter()
        dt = float(kwargs.get("dt", 0.1))
        if reference is not None and {"x", "y"}.issubset(reference):
            target_pos = (float(reference["x"]), float(reference["y"]))
            target_psi = float(reference.get("psi", self._target_psi))
        elif self._target_pos is not None:
            target_pos = self._target_pos
            target_psi = self._target_psi
        else:
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="precision", risk=0.0)
        tau = self._raw_precision_tau(state, target_pos, target_psi, dt)
        pos_err = float(np.linalg.norm(np.asarray(state[:2]) - np.asarray(target_pos)))
        risk = compute_total_risk(pos_err, 0.0, 0.0, 0.0, 0.0)
        elapsed = (time.perf_counter() - tic) * 1000.0
        self._last_diagnostics = {
            "solver_status": "analytical",
            "solver_success": True,
            "solve_time_ms": elapsed,
            "objective_value": pos_err * pos_err,
            "constraint_violation": 0.0,
            "risk_cvar": 0.0,
            "risk_total": risk,
            "risk_ice": 0.0,
            "risk_position": min(1.0, pos_err / 15.0),
            "cvar_alpha": 0.0,
            "cvar_sample_count": 0,
            "cvar_quantile": 0.0,
            "cvar_tail_sample_count": 0,
            "risk_model_status": "not_used",
            "cbf_active": False,
            "cbf_status": "inactive",
            "cbf_slack": 0.0,
            "safety_set_h": 0.0,
            "safety_set_h_dot": 0.0,
            "safety_filter_hocbf_margin": 0.0,
            "hocbf_constraint_margin": 0.0,
            "safety_filter_correction_norm": 0.0,
            "safety_filter_slack": 0.0,
            "safety_filter_slack_active": 0.0,
            "safety_filter_qp_success": 1.0,
            "safety_filter_infeasible": 0.0,
            "safety_filter_status": "inactive",
            "safety_filter_solver_backend": "none",
            "safety_filter_active": 0.0,
            "hocbf_a_norm": 0.0,
            "hocbf_soft_certificate": False,
            "safety_filter_alpha1": 0.0,
            "safety_filter_alpha2": 0.0,
        }
        return ControllerResult(tau=tau, feasible=True, mode="precision", risk=risk, cost_estimate=pos_err * pos_err)

    def reset(self) -> None:
        self._int_pos[:] = 0.0
        self._int_psi = 0.0
        self._vel_filtered[:] = 0.0
        self._r_filtered = 0.0
        self._vel_initialized = False
        self._last_diagnostics = {}


class IceAwarePrecisionDPController(PrecisionDPController):
    """冰载荷感知精确 DP：保留精确目标，增加冰情观测、CVaR proxy 和 CBF 安全修正。"""

    def __init__(self, params: Optional[IceAwareParams] = None, use_cbf: bool = True, use_cvar: bool = True, use_observer: bool = True):
        super().__init__(params or IceAwareParams())
        self.params: IceAwareParams = self.params  # type: ignore[assignment]
        self._solver_label = self.params.solver_label
        self.use_cbf = use_cbf
        self.use_cvar = use_cvar
        self.use_observer = use_observer
        self._raw_ice = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
        self._ice_est = dict(self._raw_ice)
        self._rng = np.random.default_rng(2026)
        self._observer_updated_this_step = False
        self._observer_force_estimate = np.zeros(3, dtype=np.float64)
        self._has_observer_estimate = False
        self._last_hocbf_diag: Dict[str, float] = {}

    def set_cvar_seed(self, seed: int) -> None:
        """设置 CVaR 随机种子，应由 runner 根据场景 seed 设置。"""
        self._rng = np.random.default_rng(seed)

    def set_observer_estimate(self, ice_force_estimate: "NDArray[np.float64]") -> None:
        """接收来自扰动观测器的冰力估计 [Fx, Fy, Mz] (N, N, N·m)。

        用于增强前馈补偿——当观测器可用时，使用基于动力学残差的冰力估计
        替代纯参数级 EMA 估计。
        """
        self._observer_force_estimate = np.asarray(ice_force_estimate, dtype=np.float64).reshape(3,)
        self._has_observer_estimate = True

    def set_ice_conditions(self, ice_concentration: float, ice_thickness: float, ice_drift_speed: float, ice_drift_direction: float = 0.0) -> None:
        self._raw_ice = {
            "concentration": float(np.clip(ice_concentration, 0.0, 1.0)),
            "thickness": max(0.0, float(ice_thickness)),
            "drift_speed": max(0.0, float(ice_drift_speed)),
            "drift_direction": float(ice_drift_direction),
        }

    def _update_ice_estimate(self) -> Dict[str, float]:
        if not self.use_observer:
            self._ice_est = dict(self._raw_ice)
            return self._ice_est
        from ..utils.math_utils import angle_ema_deg
        a = float(np.clip(self.params.observer_alpha, 0.01, 1.0))
        _noise_threshold = 0.01  # 低于此值时不注入噪声, 避免零冰正偏
        _direction_noise_std = getattr(self.params, 'observer_direction_noise_deg', 5.0)
        for k, v in self._raw_ice.items():
            noise = 0.0
            if k in {"concentration", "thickness", "drift_speed"} and v > _noise_threshold:
                noise = float(self._rng.normal(0.0, self.params.observer_noise_std))
            elif k == "drift_direction":
                noise = float(self._rng.normal(0.0, _direction_noise_std))
            if k == "drift_direction":
                # 角度感知 EMA: 正确处理 ±180° 环绕
                self._ice_est[k] = angle_ema_deg(
                    self._ice_est.get(k, v), v + noise, a,
                )
            else:
                self._ice_est[k] = (1.0 - a) * self._ice_est.get(k, v) + a * (v + noise)
        self._ice_est["concentration"] = float(np.clip(self._ice_est["concentration"], 0.0, 1.0))
        return self._ice_est

    def get_ice_estimate(self) -> Dict[str, float]:
        """返回当前冰情估计的副本（公共接口，供监督器等外部调用）。"""
        return dict(self._ice_est)

    def update_ice_estimate(self) -> Dict[str, float]:
        """触发冰情估计更新（公共接口，供监督器调用）。"""
        return self._update_ice_estimate()

    def _ice_force_proxy_body(self, psi: float, ice: Dict[str, float]) -> NDArray[np.float64]:
        """冰力代理 — 使用 Lindqvist (1989) 简化模型。

        替代原来的纯经验多项式代理 (240*c*(0.2+h)*(0.25+v^2))，
        新模型基于冰抗压强度、船宽、水线角等物理参数。
        """
        return _ice_force_lindqvist_proxy(
            concentration=ice.get("concentration", 0.0),
            thickness=ice.get("thickness", 0.0),
            drift_speed=ice.get("drift_speed", 0.0),
            drift_direction_deg=float(ice.get("drift_direction", 0.0)),
            psi=psi,
            crushing_strength_mpa=self.params.ice_crushing_strength_mpa,
            structure_factor=self.params.ice_structure_factor,
            vessel_beam_m=self.params.vessel_beam_m,
            vessel_length_m=self.params.vessel_length_m,
            waterline_angle_deg=self.params.vessel_waterline_angle_deg,
        )

    def _cvar_proxy(self, state: NDArray[np.float64], base_tau: NDArray[np.float64], ice: Dict[str, float]) -> Tuple[float, float, int]:
        """CVaR 风险代理 — 物理归一化损失函数。

        损失 = 控制饱和度 + 冰力扰动 + 安全边界违反
        各项归一化到 [0, ~1] 范围, 使 CVaR 具有物理可解释性。
        """
        if not self.use_cvar:
            return 0.0, 0.0, 0
        alpha = float(np.clip(self.params.cvar_alpha, 0.5, 0.99))
        n = int(max(8, self.params.cvar_samples))
        c = float(np.clip(ice.get("concentration", 0.0), 0.0, 1.0))
        h = max(0.0, float(ice.get("thickness", 0.0)))
        v = max(0.0, float(ice.get("drift_speed", 0.0)))

        # 冰力扰动标准差: 与冰况强度成正比, 归一化到力限
        sigma = self.params.cvar_sigma_force * c * (0.3 + h) * (0.4 + v)
        draws = self._rng.normal(0.0, sigma, size=n)

        # 控制饱和度: ||tau|| / max_force ∈ [0, ~1]
        # 注: control_sat 对所有样本为常量 (基于当前控制力), 冰力扰动为随机分量。
        # 这是一个代理 CVaR: 真正的联合 CVaR 需要对控制力也进行随机采样。
        max_f = max(self.params.max_force, 1.0)
        control_sat = np.linalg.norm(base_tau[:2]) / max_f
        # 添加小量执行器不确定性使 control_sat 也参与尾部估计
        actuator_noise = self._rng.normal(0.0, 0.02 * max_f, size=n)
        control_sat_samples = np.clip((np.linalg.norm(base_tau[:2]) + np.abs(actuator_noise)) / max_f, 0.0, 1.5)

        # 冰力扰动: |F_ice| / max_force ∈ [0, ~1]
        ice_disturbance = np.abs(draws) / max_f

        # 安全边界违反: max(0, pos_err - R) / R ∈ [0, ~1]
        pos_err = np.linalg.norm(state[:2] - np.array(self._target_pos if self._target_pos else (0.0, 0.0)))
        radius = max(self.params.cbf_radius, 1.0)
        violation = max(0.0, pos_err - radius) / radius

        # 综合损失 (各项权重可调) — control_sat_samples 为 n 元素数组
        losses = 0.4 * control_sat_samples + 0.3 * ice_disturbance + 0.3 * violation
        q = float(np.quantile(losses, alpha))
        tail = losses[losses >= q]
        cvar = float(np.mean(tail)) if len(tail) else q
        cvar_norm = float(np.clip(cvar, 0.0, 1.0))
        return cvar_norm, q, int(len(tail))

    def _apply_cbf(self, state: NDArray[np.float64], tau: NDArray[np.float64]) -> Tuple[NDArray[np.float64], bool, float, str]:
        """高阶 CBF (HOCBF) 安全约束 — relative-degree-2。

        对于 DP 系统, h(x) = R² - ||p - p_ref||² 的相对度为 2
        (因为 h 依赖位置, 位置是速度的积分, 速度是加速度的积分, 加速度由 τ 直接控制)。

        HOCBF 约束:
            ḧ + (γ₁ + γ₂) · ḣ + γ₁·γ₂·h ≥ 0

        其中:
            h = R² - ||p - p_ref||²
            ḣ = -2 · e_p · v_ned
            ḧ = -2 · ||v_ned||² - 2 · e_p · a_ned
            a_ned ≈ R^T · (τ_body / m)  (忽略阻尼和冰力的快速变化)

        参考: Ames et al. (2019) "Control Barrier Functions: Theory and Applications"
        """
        if not self.use_cbf or self._target_pos is None:
            self._last_hocbf_diag = {"h_val": 0.0, "h_dot": 0.0, "psi_hocbf": 0.0, "a_norm": 0.0, "gamma1": 0.0, "gamma2": 0.0}
            return tau, False, 0.0, "inactive"

        radius = float(self.params.cbf_radius)
        pos = np.asarray(state[:2], dtype=np.float64)
        target = np.asarray(self._target_pos, dtype=np.float64)
        err = pos - target
        dist = float(np.linalg.norm(err))
        margin = radius - dist

        # 在接近安全边界时激活 (内侧 85% 以内不干预, 更早介入)
        if dist < 0.85 * radius:
            self._last_hocbf_diag = {"h_val": float(radius**2 - dist**2), "h_dot": 0.0, "psi_hocbf": 0.0, "a_norm": 0.0, "gamma1": 0.0, "gamma2": 0.0}
            return tau, False, margin, "inactive"

        # 状态分量
        psi = float(state[2])
        vel_body = np.asarray(state[3:5], dtype=np.float64)  # [u, v]
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_ned2body = np.array([[cpsi, spsi], [-spsi, cpsi]])
        R_body2ned = R_ned2body.T
        vel_ned = R_body2ned @ vel_body

        # CBF 函数: h = R² - ||e_p||²
        e_p = err  # position error in NED
        h_val = radius ** 2 - np.dot(e_p, e_p)

        # 一阶导数: ḣ = -2 · e_p^T · v_ned
        h_dot = -2.0 * np.dot(e_p, vel_ned)

        # 二阶导数: ḧ ≈ -2 · ||v_ned||² - 2 · e_p^T · a_ned
        mass_est = self._vessel_mass
        accel_body = tau[:2] / mass_est
        accel_ned = R_body2ned @ accel_body
        h_ddot = -2.0 * np.dot(vel_ned, vel_ned) - 2.0 * np.dot(e_p, accel_ned)

        # HOCBF 参数 (relative-degree-2)
        # 速度自适应: 高速冲向边界时增大 gamma, 提供更强制动
        vel_toward_boundary = max(0.0, -h_dot)  # h_dot < 0 表示接近边界
        speed_factor = 1.0 + 0.5 * vel_toward_boundary / max(radius, 1.0)
        gamma1 = self.params.cbf_gain / 50.0 * speed_factor
        gamma2 = self.params.cbf_gain / 50.0 * speed_factor
        psi_hocbf = h_ddot + (gamma1 + gamma2) * h_dot + gamma1 * gamma2 * h_val

        # 速度约束层: 当靠近边界且高速冲向边界时, 额外施加制动力
        # h_v = v_max² - v_radial² (径向速度约束)
        # v_radial = -e_p^T · v_ned / ||e_p|| (朝向边界的径向速度)
        if dist > 0.5:
            v_radial = -np.dot(e_p, vel_ned) / dist  # 正值 = 冲向边界
            # 在安全区内: v_max 随 margin 缩小; 在安全区外: 允许更快返回
            if margin > 0:
                v_max = 2.0 * max(1.0, margin)
            else:
                # 距离越远, 允许越快返回 (线性增长, 上限 10 m/s)
                v_max = 2.0 + min(8.0, abs(margin) * 2.0)
            if v_radial > v_max:
                # 径向速度超限: 添加制动力
                brake_gain = 500.0 * (v_radial - v_max)  # 制动增益
                brake_dir = e_p / dist  # 朝向中心的方向
                tau_brake = np.zeros(3)
                brake_ned = brake_gain * brake_dir
                brake_body = R_ned2body @ brake_ned
                tau_brake[:2] = brake_body
                tau = tau.copy()
                tau[:2] += tau_brake
                tau = _clip_tau(tau, self.params.max_force, self.params.max_moment)

        # ∂ψ/∂τ = -2 · e_p^T · R_body2ned / m
        dpsi_dtau_ned = -2.0 * e_p / mass_est
        dpsi_dtau_body = R_ned2body @ dpsi_dtau_ned
        a_norm = float(np.linalg.norm(dpsi_dtau_body))

        # 保存 HOCBF 诊断数据 (用于 get_diagnostics)
        hocbf_margin = psi_hocbf  # ≥0 表示约束满足
        self._last_hocbf_diag = {
            "h_val": float(h_val),
            "h_dot": float(h_dot),
            "psi_hocbf": float(psi_hocbf),
            "a_norm": a_norm,
            "gamma1": float(gamma1),
            "gamma2": float(gamma2),
            "hocbf_margin": float(hocbf_margin),
            "correction_norm": 0.0,
        }

        # 如果 HOCBF 约束已满足, 不需要修正
        if psi_hocbf >= 0.0:
            return tau, False, margin, "inactive"

        # 需要修正: 计算使 ψ_hocbf ≥ 0 的最小修正力
        grad_norm_sq = a_norm * a_norm
        if grad_norm_sq < 1e-12:
            return tau, False, margin, "inactive"

        correction_gain = -psi_hocbf / grad_norm_sq
        # 限制修正幅度
        correction_gain = np.clip(correction_gain, 0.0, self.params.cbf_gain * 10.0)
        corr_body = correction_gain * dpsi_dtau_body

        tau2 = tau.copy()
        tau2[:2] += corr_body
        tau2 = _clip_tau(tau2, self.params.max_force, self.params.max_moment)

        self._last_hocbf_diag["correction_norm"] = float(np.linalg.norm(tau2 - tau))

        return tau2, True, margin, "hocbf_active" if margin >= 0 else "hocbf_boundary_corrected"

    def compute_control(self, state: NDArray[np.float64], reference: Optional[Dict[str, Any]] = None, environment: Optional[Dict[str, Any]] = None, **kwargs) -> ControllerResult:
        tic = time.perf_counter()
        dt = float(kwargs.get("dt", 0.1))
        if self._target_pos is None:
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="ice_aware", risk=0.0)

        # 冰向优化: 在冰况严重时, 微调目标航向以减小冰力
        # 原理: 迎冰航行 (船头对冰) 产生的冰力最小
        # 只在冰况明显时应用 (concentration > 0.2, thickness > 0.3)
        original_psi = self._target_psi
        ice_conc = self._ice_est.get("concentration", 0.0)
        ice_thick = self._ice_est.get("thickness", 0.0)
        if ice_conc > 0.2 and ice_thick > 0.3:
            from .ice_schedule import drift_dir_deg_to_rad
            ice_dir = drift_dir_deg_to_rad(float(self._ice_est.get("drift_direction", 0.0)))
            # 选择最接近当前航向的迎冰方向 (ice_dir 或 ice_dir + pi)
            psi_current = float(state[2])
            candidates = [ice_dir, wrap_to_pi(ice_dir + np.pi)]
            best_ice_heading = min(candidates, key=lambda a: abs(wrap_to_pi(psi_current - a)))
            # 航向偏置: 轻微偏向迎冰方向 (权重随冰况增加)
            ice_weight = min(0.3, 0.3 * ice_conc * ice_thick)  # 最大 30% 偏置
            biased_psi = psi_current + ice_weight * wrap_to_pi(best_ice_heading - psi_current)
            self._target_psi = biased_psi

        tau = self._raw_precision_tau(state, self._target_pos, self._target_psi, dt)
        self._target_psi = original_psi  # 恢复原始目标航向
        # 如果 supervisor 已更新观测器, 跳过重复更新
        if getattr(self, '_observer_updated_this_step', False):
            ice_est = self._ice_est
        else:
            ice_est = self._update_ice_estimate()
        # Feed-forward cancellation of estimated ice load.
        # 优先使用扰动观测器的冰力估计 (基于动力学残差),
        # 回退到参数级 EMA 代理模型。
        proxy_force = self._ice_force_proxy_body(float(state[2]), ice_est)
        if self._has_observer_estimate:
            # 自适应增益: 观测器与模型一致时增益更高, 不一致时更保守
            obs_norm = float(np.linalg.norm(self._observer_force_estimate))
            proxy_norm = float(np.linalg.norm(proxy_force))
            if obs_norm > 1.0 and proxy_norm > 1.0:
                # 一致性比: 越接近 1 越一致
                consistency = min(obs_norm, proxy_norm) / max(obs_norm, proxy_norm)
                adaptive_gain = self.params.ice_feedforward_gain * (0.7 + 0.3 * consistency)
            else:
                adaptive_gain = self.params.ice_feedforward_gain
            tau_ff = -adaptive_gain * self._observer_force_estimate
        else:
            tau_ff = -self.params.ice_feedforward_gain * proxy_force
        tau = _clip_tau(tau + tau_ff, self.params.max_force, self.params.max_moment)
        cvar, quantile, tail_count = self._cvar_proxy(state, tau, ice_est)
        tau, cbf_active, cbf_slack, cbf_status = self._apply_cbf(state, tau)
        pos_err = float(np.linalg.norm(np.asarray(state[:2]) - np.asarray(self._target_pos)))
        ice_risk = _ice_risk_standardized(ice_est["concentration"], ice_est["thickness"], ice_est["drift_speed"])
        risk = compute_total_risk(
            pos_err, ice_est["concentration"], ice_est["thickness"], ice_est["drift_speed"], cvar
        )
        elapsed = (time.perf_counter() - tic) * 1000.0
        self._last_diagnostics = {
            "solver_status": "analytical_observer_feedforward",
            "solver_success": True,
            "solve_time_ms": elapsed,
            "objective_value": pos_err * pos_err + 10.0 * risk,
            "constraint_violation": max(0.0, -cbf_slack),
            "risk_total": risk,
            "risk_ice": ice_risk,
            "risk_position": min(1.0, pos_err / 15.0),
            "risk_cvar": cvar,
            "cvar_alpha": self.params.cvar_alpha if self.use_cvar else 0.0,
            "cvar_sample_count": self.params.cvar_samples if self.use_cvar else 0,
            "cvar_quantile": quantile,
            "cvar_tail_sample_count": tail_count,
            "cvar_diagnostic": "proxy_tail_loss" if self.use_cvar else "disabled",
            "risk_model_status": "observer_proxy",
            "cbf_active": cbf_active,
            "cbf_status": cbf_status,
            "cbf_slack": cbf_slack,
            "input_constraint_active": bool(np.any(np.isclose(np.abs(tau[:2]), self.params.max_force, rtol=0, atol=1e-6))),
            # HOCBF 完整诊断 (用于 hocbf_diagnostics.csv)
            "safety_set_h": self._last_hocbf_diag.get("h_val", 0.0),
            "safety_set_h_dot": self._last_hocbf_diag.get("h_dot", 0.0),
            "safety_filter_hocbf_margin": self._last_hocbf_diag.get("hocbf_margin", 0.0),
            "hocbf_constraint_margin": self._last_hocbf_diag.get("hocbf_margin", 0.0),
            "hocbf_nominal_constraint_margin": self._last_hocbf_diag.get("hocbf_margin", 0.0),
            "safety_filter_correction_norm": self._last_hocbf_diag.get("correction_norm", 0.0),
            "safety_filter_slack": max(0.0, -self._last_hocbf_diag.get("psi_hocbf", 0.0)),
            "safety_filter_slack_active": 1.0 if self._last_hocbf_diag.get("psi_hocbf", 0.0) < 0.0 else 0.0,
            "safety_filter_qp_success": 1.0,
            "safety_filter_infeasible": 0.0,
            "safety_filter_status": cbf_status,
            "safety_filter_solver_backend": "analytical_cbf",
            "safety_filter_active": 1.0 if cbf_active else 0.0,
            "hocbf_a_norm": self._last_hocbf_diag.get("a_norm", 0.0),
            "hocbf_soft_certificate": self._last_hocbf_diag.get("hocbf_margin", 0.0) >= 0.0,
            "safety_filter_alpha1": self._last_hocbf_diag.get("gamma1", 0.0),
            "safety_filter_alpha2": self._last_hocbf_diag.get("gamma2", 0.0),
        }
        return ControllerResult(tau=tau, feasible=True, mode="ice_aware", risk=risk, cost_estimate=pos_err * pos_err + risk)

    def reset(self) -> None:
        super().reset()
        self._ice_est = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
        self._observer_force_estimate = np.zeros(3, dtype=np.float64)
        self._has_observer_estimate = False
        self._last_diagnostics = {}


class QuasiDPSafetyController(IceAwarePrecisionDPController):
    """准定位安全降级：允许目标在安全圈内局部放松，但仍维持边界内安全。"""

    def __init__(self, params: Optional[QuasiDPParams] = None, **kwargs: Any):
        super().__init__(params or QuasiDPParams(), **kwargs)
        self.params: QuasiDPParams = self.params  # type: ignore[assignment]
        self._solver_label = self.params.solver_label

    def compute_control(self, state: NDArray[np.float64], reference: Optional[Dict[str, Any]] = None, environment: Optional[Dict[str, Any]] = None, **kwargs) -> ControllerResult:
        if self._target_pos is None:
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="quasi_dp", risk=0.0)
        original_target = self._target_pos
        pos = np.asarray(state[:2], dtype=np.float64)
        target = np.asarray(original_target, dtype=np.float64)
        err = pos - target
        dist = float(np.linalg.norm(err))
        # If inside watch circle, relax target toward current position to reduce actuator stress;
        # if outside, do not relax so safety recovery dominates.
        if dist < self.params.watch_radius:
            relaxed = target + self.params.relax_gain * err
            self._target_pos = (float(relaxed[0]), float(relaxed[1]))
        try:
            result = super().compute_control(state, reference, environment, **kwargs)
            result.mode = "quasi_dp"
            # keep diagnostic objective interpretable with true target error
            if self._last_diagnostics:
                self._last_diagnostics["reference_relaxed"] = True
                self._last_diagnostics["true_position_error"] = dist
            return result
        except Exception:
            # 异常时恢复 target_pos 并返回安全零力输出
            return ControllerResult(tau=np.zeros(3), feasible=False, mode="quasi_dp", risk=1.0)
        finally:
            self._target_pos = original_target


class IceVaningEscapeController(IceAwarePrecisionDPController):
    """极端冰况应急：优先把艏向调整到迎冰/顺冰方向，并向安全圈中心或撤离方向给力。"""

    def __init__(self, params: Optional[EscapeParams] = None, **kwargs: Any):
        super().__init__(params or EscapeParams(), **kwargs)
        self.params: EscapeParams = self.params  # type: ignore[assignment]
        self._solver_label = self.params.solver_label

    def compute_control(self, state: NDArray[np.float64], reference: Optional[Dict[str, Any]] = None, environment: Optional[Dict[str, Any]] = None, **kwargs) -> ControllerResult:
        tic = time.perf_counter()
        if self._target_pos is None:
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="escape", risk=1.0)
        ice = self._update_ice_estimate()
        from .ice_schedule import drift_dir_deg_to_rad
        ice_dir = drift_dir_deg_to_rad(float(ice.get("drift_direction", 0.0)))
        # Bow into ice: choose nearest of ice_dir and ice_dir + pi to minimize yaw rotation.
        psi = float(state[2])
        candidates = [ice_dir, wrap_to_pi(ice_dir + np.pi)]
        target_psi = min(candidates, key=lambda ang: abs(wrap_to_pi(psi - ang)))
        original_psi = self._target_psi
        self._target_psi = target_psi
        try:
            # Position control still aims at safe-region center, but with reduced max force.
            tau = self._raw_precision_tau(state, self._target_pos, self._target_psi, float(kwargs.get("dt", 0.1)))
            # Add small retreat force opposite to estimated ice drift.
            retreat_ned = -self.params.safe_retreat_force * np.array([np.cos(ice_dir), np.sin(ice_dir)])
            retreat_body = _rotation_body_from_ned(psi) @ retreat_ned
            tau[:2] += retreat_body
            tau = _clip_tau(tau, self.params.max_force, self.params.max_moment)
            pos_err = float(np.linalg.norm(np.asarray(state[:2]) - np.asarray(self._target_pos)))
            yaw_err = abs(wrap_to_pi(psi - target_psi))
            risk = compute_total_risk(pos_err, ice.get("concentration", 0.0), ice.get("thickness", 0.0), ice.get("drift_speed", 0.0), 0.0)
            # Escape 模式风险标记: 进入 escape 即表示冰况超出正常操作范围，
            # 但风险值使用统一公式计算，确保跨模式指标可比性。
            # 在诊断字典中额外记录 escape 触发状态。
            risk = max(risk, 0.0)  # 保底非负，不再人为抬升
            self._last_diagnostics = {
                "solver_status": "analytical_escape",
                "solver_success": True,
                "solve_time_ms": (time.perf_counter() - tic) * 1000.0,
                "objective_value": pos_err * pos_err + yaw_err * yaw_err,
                "constraint_violation": 0.0,
                "risk_total": risk,
                "risk_ice": _ice_risk_standardized(ice.get("concentration", 0.0), ice.get("thickness", 0.0), ice.get("drift_speed", 0.0)),
                "risk_position": min(1.0, pos_err / 15.0),
                "risk_cvar": 0.0,
                "cvar_alpha": 0.0,
                "cvar_sample_count": 0,
                "cvar_quantile": 0.0,
                "cvar_tail_sample_count": 0,
                "risk_model_status": "escape",
                "cbf_active": True,
                "cbf_status": "escape_active",
                "cbf_slack": 0.0,
                "input_constraint_active": False,
                "safety_set_h": 0.0,
                "safety_set_h_dot": 0.0,
                "safety_filter_hocbf_margin": 0.0,
                "hocbf_constraint_margin": 0.0,
                "safety_filter_correction_norm": 0.0,
                "safety_filter_slack": 0.0,
                "safety_filter_slack_active": 0.0,
                "safety_filter_qp_success": 1.0,
                "safety_filter_infeasible": 0.0,
                "safety_filter_status": "inactive",
                "safety_filter_solver_backend": "none",
                "safety_filter_active": 0.0,
                "hocbf_a_norm": 0.0,
                "hocbf_soft_certificate": False,
                "safety_filter_alpha1": 0.0,
                "safety_filter_alpha2": 0.0,
            }
            return ControllerResult(tau=tau, feasible=True, mode="escape", risk=risk, cost_estimate=pos_err * pos_err)
        finally:
            self._target_psi = original_psi


class ModeSupervisedIceDPController(BaseController):
    """四模式监督控制器：投稿主方法。"""

    def __init__(
        self,
        precision: Optional[PrecisionDPController] = None,
        ice_aware: Optional[IceAwarePrecisionDPController] = None,
        quasi: Optional[QuasiDPSafetyController] = None,
        escape: Optional[IceVaningEscapeController] = None,
        params: Optional[SupervisorParams] = None,
    ):
        super().__init__()
        self.params = params or SupervisorParams()
        self._solver_label = self.params.solver_label
        self.precision = precision or PrecisionDPController()
        self.ice_aware = ice_aware or IceAwarePrecisionDPController()
        self.quasi = quasi or QuasiDPSafetyController()
        self.escape = escape or IceVaningEscapeController()
        self._controllers = {
            DPMode.PRECISION: self.precision,
            DPMode.ICE_AWARE: self.ice_aware,
            DPMode.QUASI_DP: self.quasi,
            DPMode.ESCAPE: self.escape,
        }
        self._mode = DPMode.PRECISION
        self._last_switch_t = -1e9
        self._raw_ice = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
        self._ice_est = dict(self._raw_ice)
        self._t = 0.0

        # E9: HOCBF 迟滞激活状态机
        self._activation_sm = HOCBFActivationStateMachine(HOCBFActivationConfig(
            h_activate=4.0, h_deactivate=9.0,
            risk_activate=0.65, risk_deactivate=0.45,
            min_dwell_time_s=self.params.dwell_time,
        ))

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        for c in self._controllers.values():
            c.set_target(x, y, psi_deg)
        self.target_position = (float(x), float(y))
        self.target_heading = deg2rad(float(psi_deg))

    def set_safe_region_radius(self, radius: float) -> None:
        """设置安全区域半径，同步到所有子控制器。"""
        for c in self._controllers.values():
            c.set_safe_region_radius(radius)

    def set_vessel_params(self, vessel_params) -> None:
        """将船舶参数传播到所有子控制器。"""
        for c in self._controllers.values():
            if hasattr(c, 'set_vessel_params'):
                c.set_vessel_params(vessel_params)

    def set_cvar_seed(self, seed: int) -> None:
        """设置 CVaR 随机种子，传播到所有支持该方法的子控制器。"""
        for c in [self.ice_aware, self.quasi, self.escape]:
            if hasattr(c, 'set_cvar_seed'):
                c.set_cvar_seed(seed)

    def set_ice_conditions(self, ice_concentration: float, ice_thickness: float, ice_drift_speed: float, ice_drift_direction: float = 0.0) -> None:
        self._raw_ice = {
            "concentration": float(np.clip(ice_concentration, 0.0, 1.0)),
            "thickness": max(0.0, float(ice_thickness)),
            "drift_speed": max(0.0, float(ice_drift_speed)),
            "drift_direction": float(ice_drift_direction),
        }
        # 传播到所有有 set_ice_conditions 的子控制器 (precision 没有该方法，跳过)
        for c in [self.ice_aware, self.quasi, self.escape]:
            c.set_ice_conditions(ice_concentration, ice_thickness, ice_drift_speed, ice_drift_direction)
        # 更新 supervisor 自身的冰况估计 (使用 ice_aware 的观测器输出)
        self._ice_est = self.ice_aware.get_ice_estimate()

    def _risk_proxy(self, state: NDArray[np.float64]) -> Tuple[float, float]:
        """风险代理 — 使用与 IceAware 相同的标准化公式。

        使用估计冰况 (来自观测器)，不使用 true ice。
        CVaR 分量从当前子控制器的诊断中获取 (如果可用)，
        否则用冰况参数估计。
        """
        pos_err = 0.0
        if self.target_position is not None:
            pos_err = float(np.linalg.norm(np.asarray(state[:2]) - np.asarray(self.target_position)))
        # 使用 ice_aware 子控制器的估计冰况 (经过观测器)
        ice_est = self.ice_aware.get_ice_estimate() if hasattr(self.ice_aware, 'get_ice_estimate') else self._raw_ice
        # CVaR 估计: 优先使用当前子控制器诊断, 否则用冰况代理
        cvar_est = 0.0
        current_ctrl = self._controllers.get(self._mode)
        if current_ctrl is not None:
            diag = current_ctrl.get_diagnostics()
            cvar_est = diag.get("risk_cvar", 0.0)
        if cvar_est == 0.0:
            c = ice_est["concentration"]
            h = ice_est["thickness"]
            cvar_est = float(np.clip(0.15 * c * (0.25 + h), 0.0, 1.0))
        total = compute_total_risk(
            pos_err, ice_est["concentration"], ice_est["thickness"], ice_est["drift_speed"], cvar_est
        )
        return total, pos_err

    def _select_mode(self, state: NDArray[np.float64], dt: float) -> DPMode:
        self._t += dt
        # 更新观测器 (如果尚未在 compute_control 中更新)
        if not getattr(self.ice_aware, '_observer_updated_this_step', False):
            self.ice_aware.update_ice_estimate()
        risk, pos_err = self._risk_proxy(state)

        # E9: 更新 HOCBF 迟滞激活状态机 (平滑风险 + 迟滞)
        cbf_radius = getattr(self.ice_aware.params, 'cbf_radius', 10.0)
        h_val = cbf_radius ** 2 - pos_err ** 2
        self._activation_sm.update(h_val=h_val, risk_raw=risk, dt=dt, current_time=self._t)
        risk_filtered = self._activation_sm.state.risk_filtered

        p = self.params
        if self._t - self._last_switch_t < p.dwell_time:
            return self._mode
        mode = self._mode
        if self._mode == DPMode.PRECISION:
            if risk >= p.ice_enter:
                mode = DPMode.ICE_AWARE
        elif self._mode == DPMode.ICE_AWARE:
            if risk >= p.extreme_risk_enter:
                mode = DPMode.ESCAPE
            elif risk >= p.high_risk_enter or pos_err >= p.pos_error_quasi_enter:
                mode = DPMode.QUASI_DP
            elif risk <= p.ice_exit:
                mode = DPMode.PRECISION
        elif self._mode == DPMode.QUASI_DP:
            if risk >= p.extreme_risk_enter:
                mode = DPMode.ESCAPE
            elif risk <= p.high_risk_exit and pos_err <= p.pos_error_quasi_exit:
                mode = DPMode.ICE_AWARE
        elif self._mode == DPMode.ESCAPE:
            # 修复: 允许 ESCAPE 直接回到 ICE_AWARE (当风险降到 ice_exit 以下时)
            # 避免必须经过 QUASI_DP 的额外 dwell_time 延迟
            if risk <= p.ice_exit:
                mode = DPMode.ICE_AWARE
            elif risk <= p.extreme_risk_exit:
                mode = DPMode.QUASI_DP
        if mode != self._mode:
            self._mode = mode
            self._last_switch_t = self._t
        return self._mode

    def compute_control(self, state: NDArray[np.float64], reference: Optional[Dict[str, Any]] = None, environment: Optional[Dict[str, Any]] = None, **kwargs) -> ControllerResult:
        dt = float(kwargs.get("dt", 0.1))
        # 观测器仅在 _select_mode 中更新一次, 避免重复更新导致
        # EMA 收敛速度加倍和噪声叠加
        mode = self._select_mode(state, dt)
        controller = self._controllers[mode]
        # 设置标志: 子控制器不要重复更新观测器 (已在 supervisor 中更新)
        self.ice_aware._observer_updated_this_step = True
        try:
            result = controller.compute_control(state, reference, environment, **kwargs)
        finally:
            self.ice_aware._observer_updated_this_step = False
        result.mode = mode.name.lower()
        diag = controller.get_diagnostics()

        # E9: 合并 HOCBF 迟滞激活状态机诊断
        activation_diag = self._activation_sm.get_diagnostics()

        diag.update({
            "supervisor_mode": int(mode),
            "supervisor_mode_name": mode.name,
            "solver_status": f"supervised:{diag.get('solver_status', 'unknown')}",
            # E9 activation diagnostics
            "activation_safety_mode": activation_diag["safety_mode"],
            "activation_hocbf_active": activation_diag["hocbf_active"],
            "activation_risk_filtered": activation_diag["risk_filtered"],
            "activation_mode_switch_count": activation_diag["mode_switch_count"],
            "activation_dwell_violations": activation_diag["dwell_time_violation_count"],
            "activation_risk_scale": activation_diag["risk_scale"],
            "activation_reason": activation_diag["activation_reason"],
        })
        self._last_diagnostics = diag
        return result

    def reset(self) -> None:
        for c in self._controllers.values():
            c.reset()
        self._mode = DPMode.PRECISION
        self._last_switch_t = -1e9
        self._t = 0.0
        self._ice_est = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
        self._last_diagnostics = {}
        # E9: 重置激活状态机
        self._activation_sm.reset()
