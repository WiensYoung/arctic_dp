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


class DPMode(IntEnum):
    PRECISION = 0
    ICE_AWARE = 1
    QUASI_DP = 2
    ESCAPE = 3


@dataclass
class PrecisionDPParams:
    kp_pos: float = 180.0
    kd_pos: float = 90.0
    ki_pos: float = 0.0
    kp_heading: float = 650.0
    kd_heading: float = 280.0
    ki_heading: float = 0.0
    max_force: float = 1500.0
    max_moment: float = 20000.0
    max_integral: float = 20.0
    solver_label: str = "precision_pd"


@dataclass
class IceAwareParams(PrecisionDPParams):
    ice_feedforward_gain: float = 0.55
    observer_alpha: float = 0.18
    observer_noise_std: float = 0.02
    cvar_alpha: float = 0.90
    cvar_samples: int = 64
    cvar_sigma_force: float = 150.0
    cbf_radius: float = 10.0
    cbf_gain: float = 120.0
    thrust_margin_soft_limit: float = 0.72
    risk_gain: float = 0.75
    solver_label: str = "ice_aware_risk_pd"
    # --- 冰力模型参数 (Lindqvist 1989 简化版) ---
    ice_crushing_strength_mpa: float = 2.0     # 冰单轴抗压强度 (MPa), 典型值 1-5
    ice_structure_factor: float = 0.45          # 结构系数 (船体形状因子)
    vessel_beam_m: float = 22.0                 # 船宽 (m), 雪龙2号约22m
    vessel_waterline_angle_deg: float = 30.0    # 水线角 (度)


@dataclass
class QuasiDPParams(IceAwareParams):
    watch_radius: float = 10.0
    relax_gain: float = 0.65
    pos_weight_scale_inside: float = 0.35
    max_force: float = 1300.0
    solver_label: str = "quasi_dp_safety"


@dataclass
class EscapeParams(IceAwareParams):
    safe_retreat_force: float = 800.0
    heading_to_ice_weight: float = 1.0
    max_force: float = 1000.0
    solver_label: str = "ice_vaning_escape"


@dataclass
class SupervisorParams:
    ice_enter: float = 0.28
    ice_exit: float = 0.20
    high_risk_enter: float = 0.58
    high_risk_exit: float = 0.42
    extreme_risk_enter: float = 0.82
    extreme_risk_exit: float = 0.65
    pos_error_quasi_enter: float = 8.5
    pos_error_quasi_exit: float = 6.0
    dwell_time: float = 8.0
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

    公式基于无量纲冰况参数的乘积形式，经过文献校准：
    - concentration ∈ [0, 1]
    - thickness 贡献通过 (0.3 + thickness) 归一化
    - drift_speed 贡献通过 (0.4 + drift_speed) 归一化
    """
    c = float(np.clip(concentration, 0.0, 1.0))
    h = max(0.0, float(thickness))
    v = max(0.0, float(drift_speed))
    return float(np.clip(c * (0.3 + h) * (0.4 + v), 0.0, 1.0))


def _ice_force_lindqvist_proxy(
    concentration: float,
    thickness: float,
    drift_speed: float,
    drift_direction_rad: float,
    psi: float,
    crushing_strength_mpa: float = 2.0,
    structure_factor: float = 0.45,
    vessel_beam_m: float = 22.0,
    vessel_length_m: float = 122.5,
    waterline_angle_deg: float = 30.0,
) -> NDArray[np.float64]:
    """基于 Lindqvist (1989) 简化的冰力代理模型。

    F = σ_c * structure_factor * h * B * f(v_r, α)
    其中 σ_c 为冰抗压强度, h 为冰厚, B 为船宽, f 为速度/角度相关因子。

    比原始多项式代理 (240*c*(0.2+h)*(0.25+v^2)) 更有物理依据。
    """
    c = float(np.clip(concentration, 0.0, 1.0))
    h = max(0.0, float(thickness))
    v = max(0.0, float(drift_speed))
    alpha = deg2rad(float(waterline_angle_deg))

    # Lindqvist 速度因子: 低速时以挤压为主，高速时以弯曲为主
    # 简化: f = 1.0 + 0.5 * v / (v + 0.5)  (饱和函数)
    speed_factor = 1.0 + 0.5 * v / (v + 0.5) if v > 0 else 1.0

    # 角度因子: 水线角越陡，冰力越大
    angle_factor = 1.0 + 0.3 * math.tan(min(alpha, math.pi / 3))

    # 基础冰力 (kN): σ_c(MPa) * h(m) * B(m) * 结构系数 * 各因子
    # 乘以 concentration 作为冰覆盖率修正
    base_force_kn = (
        crushing_strength_mpa * 1000.0  # kN/m^2
        * h * vessel_beam_m * structure_factor
        * speed_factor * angle_factor
        * c  # 冰覆盖率修正
    )

    # 转换为 NED 方向力
    force_ned = base_force_kn * np.array([np.cos(drift_direction_rad), np.sin(drift_direction_rad)])
    force_body = _rotation_body_from_ned(psi) @ force_ned

    # 力矩: 力臂约 0.18 * 船长
    lever = 0.18 * vessel_length_m
    moment = lever * force_body[1]

    return np.array([force_body[0], force_body[1], moment], dtype=np.float64)


class PrecisionDPController(BaseController):
    """高精度定位主模式：严格追踪目标点和艏向，不做准定位放松。"""

    def __init__(self, params: Optional[PrecisionDPParams] = None):
        super().__init__()
        self.params = params or PrecisionDPParams()
        self._solver_label = self.params.solver_label
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_psi: float = 0.0
        self._int_pos = np.zeros(2, dtype=np.float64)
        self._int_psi = 0.0

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

    def _raw_precision_tau(self, state: NDArray[np.float64], target_pos: Tuple[float, float], target_psi: float, dt: float) -> NDArray[np.float64]:
        p = self.params
        x, y, psi = float(state[0]), float(state[1]), float(state[2])
        u, v, r = float(state[3]), float(state[4]), float(state[5])
        pos_err_ned = np.array([x - target_pos[0], y - target_pos[1]], dtype=np.float64)
        vel_ned = np.array([np.cos(psi) * u - np.sin(psi) * v, np.sin(psi) * u + np.cos(psi) * v])
        e_psi = wrap_to_pi(psi - target_psi)
        self._int_pos = np.clip(self._int_pos + pos_err_ned * dt, -p.max_integral, p.max_integral)
        self._int_psi = float(np.clip(self._int_psi + e_psi * dt, -p.max_integral, p.max_integral))
        force_ned = -(p.kp_pos * pos_err_ned + p.kd_pos * vel_ned + p.ki_pos * self._int_pos)
        force_body = _rotation_body_from_ned(psi) @ force_ned
        moment = -(p.kp_heading * e_psi + p.kd_heading * r + p.ki_heading * self._int_psi)
        return _clip_tau(np.array([force_body[0], force_body[1], moment]), p.max_force, p.max_moment)

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
        risk = min(1.0, pos_err / 20.0)
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
            "risk_position": risk,
            "cvar_alpha": 0.0,
            "cvar_sample_count": 0,
            "cvar_quantile": 0.0,
            "cvar_tail_sample_count": 0,
            "risk_model_status": "not_used",
            "cbf_active": False,
            "cbf_status": "inactive",
            "cbf_slack": 0.0,
        }
        return ControllerResult(tau=tau, feasible=True, mode="precision", risk=risk, cost_estimate=pos_err * pos_err)

    def reset(self) -> None:
        self._int_pos[:] = 0.0
        self._int_psi = 0.0
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

    def set_cvar_seed(self, seed: int) -> None:
        """设置 CVaR 随机种子，应由 runner 根据场景 seed 设置。"""
        self._rng = np.random.default_rng(seed)

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
        a = float(np.clip(self.params.observer_alpha, 0.01, 1.0))
        for k, v in self._raw_ice.items():
            noise = 0.0
            if k in {"concentration", "thickness", "drift_speed"}:
                noise = float(self._rng.normal(0.0, self.params.observer_noise_std))
            self._ice_est[k] = (1.0 - a) * self._ice_est.get(k, v) + a * max(0.0, v + noise)
        self._ice_est["concentration"] = float(np.clip(self._ice_est["concentration"], 0.0, 1.0))
        return self._ice_est

    def _ice_force_proxy_body(self, psi: float, ice: Dict[str, float]) -> NDArray[np.float64]:
        """冰力代理 — 使用 Lindqvist (1989) 简化模型。

        替代原来的纯经验多项式代理 (240*c*(0.2+h)*(0.25+v^2))，
        新模型基于冰抗压强度、船宽、水线角等物理参数。
        """
        direction = deg2rad(float(ice.get("drift_direction", 0.0)))
        return _ice_force_lindqvist_proxy(
            concentration=ice.get("concentration", 0.0),
            thickness=ice.get("thickness", 0.0),
            drift_speed=ice.get("drift_speed", 0.0),
            drift_direction_rad=direction,
            psi=psi,
            crushing_strength_mpa=self.params.ice_crushing_strength_mpa,
            structure_factor=self.params.ice_structure_factor,
            vessel_beam_m=self.params.vessel_beam_m,
            waterline_angle_deg=self.params.vessel_waterline_angle_deg,
        )

    def _cvar_proxy(self, state: NDArray[np.float64], base_tau: NDArray[np.float64], ice: Dict[str, float]) -> Tuple[float, float, int]:
        if not self.use_cvar:
            return 0.0, 0.0, 0
        alpha = float(np.clip(self.params.cvar_alpha, 0.5, 0.99))
        n = int(max(8, self.params.cvar_samples))
        c = float(np.clip(ice.get("concentration", 0.0), 0.0, 1.0))
        h = max(0.0, float(ice.get("thickness", 0.0)))
        sigma = self.params.cvar_sigma_force * (0.25 + c) * (0.25 + h)
        draws = self._rng.normal(0.0, sigma, size=n)
        pos_err = np.linalg.norm(state[:2] - np.array(self._target_pos if self._target_pos else (0.0, 0.0)))
        losses = 0.015 * np.linalg.norm(base_tau[:2]) + 0.02 * np.abs(draws) + 0.5 * max(0.0, pos_err - self.params.cbf_radius)
        q = float(np.quantile(losses, alpha))
        tail = losses[losses >= q]
        cvar = float(np.mean(tail)) if len(tail) else q
        cvar_norm = float(np.clip(cvar / 50.0, 0.0, 1.0))
        return cvar_norm, q, int(len(tail))

    def _apply_cbf(self, state: NDArray[np.float64], tau: NDArray[np.float64]) -> Tuple[NDArray[np.float64], bool, float, str]:
        if not self.use_cbf or self._target_pos is None:
            return tau, False, 0.0, "inactive"
        radius = float(self.params.cbf_radius)
        pos = np.asarray(state[:2], dtype=np.float64)
        target = np.asarray(self._target_pos, dtype=np.float64)
        err = pos - target
        dist = float(np.linalg.norm(err))
        margin = radius - dist
        if dist < 0.75 * radius:
            return tau, False, margin, "inactive"
        outward_ned = err / max(dist, 1e-6)
        # safety correction points back to target, strongest outside boundary.
        gain = self.params.cbf_gain * (1.0 + max(0.0, dist - 0.75 * radius) / max(radius, 1e-6))
        corr_ned = -gain * outward_ned
        corr_body = _rotation_body_from_ned(float(state[2])) @ corr_ned
        tau2 = tau.copy()
        tau2[:2] += corr_body
        tau2 = _clip_tau(tau2, self.params.max_force, self.params.max_moment)
        return tau2, True, margin, "active" if margin >= 0 else "boundary_violation_corrected"

    def compute_control(self, state: NDArray[np.float64], reference: Optional[Dict[str, Any]] = None, environment: Optional[Dict[str, Any]] = None, **kwargs) -> ControllerResult:
        tic = time.perf_counter()
        dt = float(kwargs.get("dt", 0.1))
        if self._target_pos is None:
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="ice_aware", risk=0.0)
        tau = self._raw_precision_tau(state, self._target_pos, self._target_psi, dt)
        ice_est = self._update_ice_estimate()
        # Feed-forward cancellation of estimated ice load, clipped by actuator limits.
        tau_ff = -self.params.ice_feedforward_gain * self._ice_force_proxy_body(float(state[2]), ice_est)
        tau = _clip_tau(tau + tau_ff, self.params.max_force, self.params.max_moment)
        cvar, quantile, tail_count = self._cvar_proxy(state, tau, ice_est)
        tau, cbf_active, cbf_slack, cbf_status = self._apply_cbf(state, tau)
        pos_err = float(np.linalg.norm(np.asarray(state[:2]) - np.asarray(self._target_pos)))
        # 使用标准化冰风险公式 (与 supervisor 一致)
        ice_risk = _ice_risk_standardized(
            ice_est["concentration"], ice_est["thickness"], ice_est["drift_speed"]
        )
        risk = float(np.clip(0.35 * min(1.0, pos_err / 15.0) + 0.35 * ice_risk + 0.30 * cvar, 0.0, 1.0))
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
        }
        return ControllerResult(tau=tau, feasible=True, mode="ice_aware", risk=risk, cost_estimate=pos_err * pos_err + risk)

    def reset(self) -> None:
        super().reset()
        self._ice_est = {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0}
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
        ice_dir = deg2rad(float(ice.get("drift_direction", 0.0)))
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
            risk = float(np.clip(0.75 + 0.25 * min(1.0, pos_err / 20.0), 0.0, 1.0))
            self._last_diagnostics = {
                "solver_status": "analytical_escape",
                "solver_success": True,
                "solve_time_ms": (time.perf_counter() - tic) * 1000.0,
                "objective_value": pos_err * pos_err + yaw_err * yaw_err,
                "risk_total": risk,
                "risk_ice": 1.0,
                "risk_cvar": risk,
                "cvar_alpha": 0.0,
                "cvar_sample_count": 0,
                "cvar_quantile": 0.0,
                "cvar_tail_sample_count": 0,
                "risk_model_status": "escape",
                "cbf_active": True,
                "cbf_status": "escape_active",
                "cbf_slack": 0.0,
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
        self._t = 0.0

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        for c in self._controllers.values():
            c.set_target(x, y, psi_deg)
        self.target_position = (float(x), float(y))
        self.target_heading = deg2rad(float(psi_deg))

    def set_safe_region_radius(self, radius: float) -> None:
        """设置安全区域半径，同步到所有子控制器。"""
        for c in self._controllers.values():
            c.set_safe_region_radius(radius)

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

    def _risk_proxy(self, state: NDArray[np.float64]) -> Tuple[float, float]:
        """风险代理 — 使用与 IceAware 相同的标准化公式。

        CVaR 分量从当前子控制器的诊断中获取 (如果可用)，
        否则用冰况参数估计。
        """
        pos_err = 0.0
        if self.target_position is not None:
            pos_err = float(np.linalg.norm(np.asarray(state[:2]) - np.asarray(self.target_position)))
        ice_risk = _ice_risk_standardized(
            self._raw_ice["concentration"],
            self._raw_ice["thickness"],
            self._raw_ice["drift_speed"],
        )
        # CVaR 估计: 优先使用当前子控制器诊断, 否则用冰况代理
        cvar_est = 0.0
        current_ctrl = self._controllers.get(self._mode)
        if current_ctrl is not None:
            diag = current_ctrl.get_diagnostics()
            cvar_est = diag.get("risk_cvar", 0.0)
        if cvar_est == 0.0:
            # 用冰况参数估计 CVaR (与 IceAware._cvar_proxy 的 sigma 逻辑一致)
            c = self._raw_ice["concentration"]
            h = self._raw_ice["thickness"]
            cvar_est = float(np.clip(0.15 * c * (0.25 + h), 0.0, 1.0))
        total = float(np.clip(0.35 * min(1.0, pos_err / 15.0) + 0.35 * ice_risk + 0.30 * cvar_est, 0.0, 1.0))
        return total, pos_err

    def _select_mode(self, state: NDArray[np.float64], dt: float) -> DPMode:
        self._t += dt
        risk, pos_err = self._risk_proxy(state)
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
        mode = self._select_mode(state, dt)
        controller = self._controllers[mode]
        result = controller.compute_control(state, reference, environment, **kwargs)
        result.mode = mode.name.lower()
        diag = controller.get_diagnostics()
        diag.update({
            "supervisor_mode": int(mode),
            "supervisor_mode_name": mode.name,
            "solver_status": f"supervised:{diag.get('solver_status', 'unknown')}",
        })
        self._last_diagnostics = diag
        return result

    def reset(self) -> None:
        for c in self._controllers.values():
            c.reset()
        self._mode = DPMode.PRECISION
        self._last_switch_t = -1e9
        self._t = 0.0
        self._last_diagnostics = {}
