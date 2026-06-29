"""Sensor and observer models for Arctic DP estimation.

Models:
- PositionSensorModel: GNSS/BeiDou noise, bias, dropout
- HeadingSensorModel: gyro/compass noise, bias, drift
- IceConditionSensorModel: ice estimation error
- IceLoadObserver: first-order observer with convergence
- DisturbanceObserver: disturbance estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import numpy as np
from numpy.typing import NDArray


@dataclass
class SensorNoiseConfig:
    """Configuration for sensor noise."""
    gaussian_std: float = 0.0
    bias: float = 0.0
    random_walk_std: float = 0.0
    dropout_probability: float = 0.0
    time_delay_steps: int = 0
    low_pass_alpha: float = 1.0  # 1.0 = no filtering


class PositionSensorModel:
    """GNSS/BeiDou position sensor with noise, bias, dropout."""

    def __init__(self, config: Optional[SensorNoiseConfig] = None):
        self.config = config or SensorNoiseConfig(gaussian_std=0.5)
        self._bias = np.zeros(2)
        self._filtered = np.zeros(2)
        self._initialized = False
        from collections import deque
        self._delay_buffer: deque[NDArray] = deque()

    def measure(
        self,
        true_position: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Return measured position with noise/bias/dropout."""
        # Random walk bias
        if self.config.random_walk_std > 0:
            self._bias += rng.normal(0, self.config.random_walk_std, 2)

        # M2 fix: initialize filtered to true position on first call
        if not self._initialized:
            self._filtered = true_position.copy()
            self._initialized = True

        # Dropout
        if rng.random() < self.config.dropout_probability:
            return self._filtered  # Return last known

        # Measurement
        noise = rng.normal(0, self.config.gaussian_std, 2) if self.config.gaussian_std > 0 else np.zeros(2)
        measured = true_position + noise + self._bias + self.config.bias

        # Low-pass filter
        a = self.config.low_pass_alpha
        self._filtered = a * measured + (1 - a) * self._filtered

        # Time delay buffer
        if self.config.time_delay_steps > 0:
            self._delay_buffer.append(self._filtered.copy())
            if len(self._delay_buffer) > self.config.time_delay_steps:
                return self._delay_buffer.popleft()
            # Warmup: buffer未满时用第一个填充值填满, 保持最大延迟而非零延迟
            while len(self._delay_buffer) < self.config.time_delay_steps + 1:
                self._delay_buffer.appendleft(self._delay_buffer[0])
            return self._delay_buffer.popleft()

        return self._filtered


class HeadingSensorModel:
    """Gyro/compass heading sensor with noise, bias, drift."""

    def __init__(self, config: Optional[SensorNoiseConfig] = None):
        self.config = config or SensorNoiseConfig(gaussian_std=0.01)
        self._bias = 0.0
        self._filtered = 0.0
        self._initialized = False

    def measure(self, true_heading: float, rng: np.random.Generator) -> float:
        """Return measured heading with noise/bias/drift.

        Uses angle-aware EMA to correctly handle ±π wraparound.
        """
        from ..utils.math_utils import wrap_to_pi

        if self.config.random_walk_std > 0:
            self._bias += rng.normal(0, self.config.random_walk_std)

        noise = rng.normal(0, self.config.gaussian_std) if self.config.gaussian_std > 0 else 0.0
        measured = true_heading + noise + self._bias + self.config.bias
        measured = wrap_to_pi(measured)

        if not self._initialized:
            self._filtered = true_heading
            self._initialized = True

        # Angle-aware low-pass: use shortest angular difference
        a = self.config.low_pass_alpha
        delta = wrap_to_pi(measured - self._filtered)
        self._filtered = wrap_to_pi(self._filtered + a * delta)
        return self._filtered


@dataclass
class IceEstimate:
    """Estimated ice state with uncertainty.

    Note: drift_direction_deg stores the direction in DEGREES (0-360),
    consistent with IceState.drift_direction in ice_schedule.py.

    Compatibility properties (thickness, drift_speed, drift_direction) allow
    this class to be used interchangeably with IceState.
    """
    concentration: float = 0.0
    thickness_m: float = 0.0
    drift_speed_mps: float = 0.0
    drift_direction_deg: float = 0.0
    covariance: Optional[NDArray[np.float64]] = None
    source: str = "estimated"

    @property
    def thickness(self) -> float:
        """兼容 IceState.thickness"""
        return self.thickness_m

    @property
    def drift_speed(self) -> float:
        """兼容 IceState.drift_speed"""
        return self.drift_speed_mps

    @property
    def drift_direction(self) -> float:
        """兼容 IceState.drift_direction"""
        return self.drift_direction_deg

    def to_dict(self) -> dict:
        return {
            "concentration": self.concentration,
            "thickness": self.thickness_m,
            "drift_speed": self.drift_speed_mps,
            "drift_direction": self.drift_direction_deg,
            "source": self.source,
        }


class IceConditionSensorModel:
    """Ice condition estimation with uncertainty.

    Models estimation error for:
    - Ice concentration
    - Ice thickness
    - Ice drift speed
    - Ice drift direction
    """

    def __init__(
        self,
        concentration_std: float = 0.05,
        thickness_std: float = 0.1,
        drift_speed_std: float = 0.05,
        drift_direction_std_deg: float = 5.0,
        observer_alpha: float = 0.18,
        # 向后兼容: 旧字段名 drift_direction_std 解释为 degrees
        drift_direction_std: Optional[float] = None,
    ):
        self.concentration_std = concentration_std
        self.thickness_std = thickness_std
        self.drift_speed_std = drift_speed_std
        # 方向噪声单位: 度 (degree)。新字段 drift_direction_std_deg 优先。
        if drift_direction_std is not None:
            import warnings
            warnings.warn(
                "drift_direction_std is deprecated, use drift_direction_std_deg (same unit: degrees)",
                DeprecationWarning, stacklevel=2,
            )
            self.drift_direction_std_deg = float(drift_direction_std)
        else:
            self.drift_direction_std_deg = float(drift_direction_std_deg)
        self.observer_alpha = observer_alpha
        self._estimate = IceEstimate()

    def update(
        self,
        true_concentration: float,
        true_thickness: float,
        true_drift_speed: float,
        true_drift_direction: float,
        rng: np.random.Generator,
        dt: float = 0.1,
    ) -> IceEstimate:
        """Update ice estimate with noisy observation.

        Drift direction uses angle-aware EMA to correctly handle ±180° wraparound.
        """
        from ..utils.math_utils import angle_ema_deg

        # H7 fix: scale alpha by dt to make observer bandwidth dt-invariant
        # At dt_ref=0.1s and alpha=0.18, time constant ≈ 0.56s
        # For other dt: alpha_eff = 1 - (1-alpha)^(dt/dt_ref)
        dt_ref = 0.1
        a = 1.0 - (1.0 - self.observer_alpha) ** (dt / dt_ref) if dt > 0 else self.observer_alpha

        # Noisy observations
        c_obs = float(np.clip(true_concentration + rng.normal(0, self.concentration_std), 0, 1))
        h_obs = max(0, true_thickness + rng.normal(0, self.thickness_std))
        v_obs = max(0, true_drift_speed + rng.normal(0, self.drift_speed_std))
        d_obs = true_drift_direction + rng.normal(0, self.drift_direction_std_deg)

        # Exponential moving average (linear for scalar quantities)
        self._estimate.concentration = (1 - a) * self._estimate.concentration + a * c_obs
        self._estimate.thickness_m = (1 - a) * self._estimate.thickness_m + a * h_obs
        self._estimate.drift_speed_mps = (1 - a) * self._estimate.drift_speed_mps + a * v_obs
        # Angle-aware EMA for direction (handles ±180° wraparound)
        self._estimate.drift_direction_deg = angle_ema_deg(
            self._estimate.drift_direction_deg, d_obs, a,
        )
        self._estimate.source = "estimated"

        # Covariance (diagonal approximation)
        self._estimate.covariance = np.diag([
            self.concentration_std**2,
            self.thickness_std**2,
            self.drift_speed_std**2,
            self.drift_direction_std_deg**2,
        ])

        return self._estimate

    def get_estimate(self) -> IceEstimate:
        return self._estimate

    def reset(self) -> None:
        self._estimate = IceEstimate()


class IceLoadObserver:
    """降阶 Kalman 冰力观测器 — 同时估计冰力和冰力变化率。

    状态: x = [dF, dF_dot] (每 DOF), 其中 dF 是冰力, dF_dot 是冰力变化率
    观测: z = M·a - τ_control + D·v = dF (动力学残差)

    相比一阶低通滤波器的优势:
    - 更好的噪声抑制 (Kalman 增益自动调节)
    - 更快的跟踪 (利用冰力变化率预测)
    - 自适应带宽 (过程噪声 Q / 观测噪声 R 调节)

    Reference: Fossen (2011), "Handbook of Marine Craft Hydrodynamics and Motion
    Control", Section 10.3 — Disturbance Observer Design.
    """

    def __init__(self, alpha: float = 0.15, n_states: int = 3):
        """Initialize observer.

        Args:
            alpha: Filter gain (0, 1]. Higher = faster convergence but noisier.
                   Typical range: 0.05-0.3 for DP applications.
            n_states: Number of DOF (default 3: surge, sway, yaw).
        """
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._estimate = np.zeros(n_states)
        self._prev_velocity = np.zeros(n_states)
        self._initialized = False
        # Kalman 状态: [dF, dF_dot] per DOF
        self._x = np.zeros(2 * n_states)  # [dF_surge, dF_sway, dF_yaw, dF_dot_surge, ...]
        self._P = np.eye(2 * n_states) * 1e4  # 初始协方差
        # 过程噪声: 冰力变化率的方差
        self._Q_ddot = 500.0 ** 2  # 冰力变化率方差 (N/s)² — 较大值允许快速跟踪
        # 观测噪声: 动力学残差的方差
        self._R = 200.0 ** 2  # 观测噪声方差 (N)² — 较大值抑制噪声

    def update(
        self,
        tau_control: NDArray[np.float64],
        velocity: NDArray[np.float64],
        mass: float,
        Izz: float,
        Xu: float, Yv: float, Nr: float,
        Xu_abs: float, Yv_abs: float, Nr_abs: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """降阶 Kalman 观测器更新 — 估计冰力 [Fx, Fy, Mz]。

        状态模型 (每 DOF 独立):
            dF(k+1) = dF(k) + dt * dF_dot(k)
            dF_dot(k+1) = dF_dot(k) + w,  w ~ N(0, Q_ddot)
        观测模型:
            z(k) = dF(k) + v,  v ~ N(0, R)
        其中 z = M·a - τ_control + D·v (动力学残差)

        Args:
            tau_control: Applied control force [Fx, Fy, Mz] (N, N, N·m)
            velocity: Current body velocity [u, v, r] (m/s, m/s, rad/s)
            mass: Vessel mass (kg)
            Izz: Yaw moment of inertia (kg·m²)
            Xu, Yv, Nr: Linear damping coefficients
            Xu_abs, Yv_abs, Nr_abs: Quadratic damping coefficients
            dt: Time step (s)

        Returns:
            Updated ice force estimate [Fx, Fy, Mz]
        """
        velocity = np.asarray(velocity, dtype=np.float64).reshape(3,)

        if not self._initialized:
            self._prev_velocity = velocity.copy()
            self._initialized = True
            return self._estimate.copy()

        # 动力学残差: z = M·a - τ_control + D·v
        if dt > 1e-6:
            accel = (velocity - self._prev_velocity) / dt
        else:
            accel = np.zeros(3)

        M_accel = np.array([mass * accel[0], mass * accel[1], Izz * accel[2]])
        damping = np.array([
            Xu * velocity[0] + Xu_abs * abs(velocity[0]) * velocity[0],
            Yv * velocity[1] + Yv_abs * abs(velocity[1]) * velocity[1],
            Nr * velocity[2] + Nr_abs * abs(velocity[2]) * velocity[2],
        ])
        tau_control = np.asarray(tau_control, dtype=np.float64).reshape(3,)
        z = M_accel - tau_control + damping  # 观测量

        # Kalman 滤波 — 每 DOF 独立 (2 状态: dF, dF_dot)
        dt2 = dt * dt
        for i in range(3):
            # 状态转移矩阵 F = [[1, dt], [0, 1]]
            # 过程噪声 Q = [[Q_ddot*dt^3/3, Q_ddot*dt^2/2], [Q_ddot*dt^2/2, Q_ddot*dt]]
            q = self._Q_ddot
            Q11 = q * dt2 * dt / 3.0
            Q12 = q * dt2 / 2.0
            Q22 = q * dt

            # 预测
            x1_pred = self._x[i] + dt * self._x[i + 3]  # dF_pred
            x2_pred = self._x[i + 3]                      # dF_dot_pred

            # 预测协方差 P_pred = F @ P @ F.T + Q
            P = self._P
            P11 = P[i, i] + 2 * dt * P[i, i + 3] + dt2 * P[i + 3, i + 3] + Q11
            P12 = P[i, i + 3] + dt * P[i + 3, i + 3] + Q12
            P22 = P[i + 3, i + 3] + Q22

            # Kalman 增益 K = P_pred @ H.T / (H @ P_pred @ H.T + R)
            # H = [1, 0] (只观测 dF)
            S = P11 + self._R
            K1 = P11 / S
            K2 = P12 / S

            # 更新
            innov = z[i] - x1_pred  # 新息
            self._x[i] = x1_pred + K1 * innov          # dF 估计
            self._x[i + 3] = x2_pred + K2 * innov      # dF_dot 估计

            # 更新协方差 (Joseph 形式保证正定)
            self._P[i, i] = (1.0 - K1) * P11
            self._P[i, i + 3] = (1.0 - K1) * P12
            self._P[i + 3, i] = P12 - K2 * P11   # (I-KH)*P 的 [1,0] 项: -K2*P11 + P12
            self._P[i + 3, i + 3] = P22 - K2 * P12

        # 输出: 冰力估计 = 状态前 3 个分量
        self._estimate = self._x[:3].copy()
        self._prev_velocity = velocity.copy()

        return self._estimate.copy()

    def reset(self) -> None:
        """重置观测器状态。"""
        self._estimate = np.zeros(3)
        self._prev_velocity = np.zeros(3)
        self._initialized = False
        self._x = np.zeros(6)
        self._P = np.eye(6) * 1e4

    def get_estimate(self) -> NDArray[np.float64]:
        """Return current ice force estimate [Fx, Fy, Mz]."""
        return self._estimate.copy()


# 保留 DisturbanceObserver 作为 IceLoadObserver 的别名 (向后兼容)
DisturbanceObserver = IceLoadObserver
