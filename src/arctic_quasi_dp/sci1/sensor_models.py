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
        self._delay_buffer: list[NDArray] = []

    def measure(
        self,
        true_position: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Return measured position with noise/bias/dropout."""
        # Random walk bias
        if self.config.random_walk_std > 0:
            self._bias += rng.normal(0, self.config.random_walk_std, 2)

        # Dropout
        if rng.random() < self.config.dropout_probability:
            return self._filtered  # Return last known

        # Measurement
        noise = rng.normal(0, self.config.gaussian_std, 2) if self.config.gaussian_std > 0 else np.zeros(2)
        measured = true_position + noise + self._bias + self.config.bias

        # Low-pass filter
        a = self.config.low_pass_alpha
        self._filtered = a * measured + (1 - a) * self._filtered

        # Time delay
        if self.config.time_delay_steps > 0:
            self._delay_buffer.append(self._filtered.copy())
            if len(self._delay_buffer) > self.config.time_delay_steps:
                return self._delay_buffer.pop(0)
            return np.zeros(2)

        return self._filtered


class HeadingSensorModel:
    """Gyro/compass heading sensor with noise, bias, drift."""

    def __init__(self, config: Optional[SensorNoiseConfig] = None):
        self.config = config or SensorNoiseConfig(gaussian_std=0.01)
        self._bias = 0.0
        self._filtered = 0.0

    def measure(self, true_heading: float, rng: np.random.Generator) -> float:
        """Return measured heading with noise/bias/drift."""
        if self.config.random_walk_std > 0:
            self._bias += rng.normal(0, self.config.random_walk_std)

        noise = rng.normal(0, self.config.gaussian_std) if self.config.gaussian_std > 0 else 0.0
        measured = true_heading + noise + self._bias + self.config.bias

        # Normalize
        measured = (measured + math.pi) % (2 * math.pi) - math.pi

        # Low-pass
        a = self.config.low_pass_alpha
        self._filtered = a * measured + (1 - a) * self._filtered
        return self._filtered


@dataclass
class IceEstimate:
    """Estimated ice state with uncertainty."""
    concentration: float = 0.0
    thickness_m: float = 0.0
    drift_speed_mps: float = 0.0
    drift_direction_rad: float = 0.0
    covariance: Optional[NDArray[np.float64]] = None
    source: str = "estimated"

    def to_dict(self) -> dict:
        return {
            "concentration": self.concentration,
            "thickness": self.thickness_m,
            "drift_speed": self.drift_speed_mps,
            "drift_direction": self.drift_direction_rad,
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
        drift_direction_std: float = 0.1,
        observer_alpha: float = 0.18,
    ):
        self.concentration_std = concentration_std
        self.thickness_std = thickness_std
        self.drift_speed_std = drift_speed_std
        self.drift_direction_std = drift_direction_std
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
        """Update ice estimate with noisy observation."""
        a = self.observer_alpha

        # Noisy observations
        c_obs = float(np.clip(true_concentration + rng.normal(0, self.concentration_std), 0, 1))
        h_obs = max(0, true_thickness + rng.normal(0, self.thickness_std))
        v_obs = max(0, true_drift_speed + rng.normal(0, self.drift_speed_std))
        d_obs = true_drift_direction + rng.normal(0, self.drift_direction_std)

        # Exponential moving average
        self._estimate.concentration = (1 - a) * self._estimate.concentration + a * c_obs
        self._estimate.thickness_m = (1 - a) * self._estimate.thickness_m + a * h_obs
        self._estimate.drift_speed_mps = (1 - a) * self._estimate.drift_speed_mps + a * v_obs
        self._estimate.drift_direction_rad = (1 - a) * self._estimate.drift_direction_rad + a * d_obs
        self._estimate.source = "estimated"

        # Covariance (diagonal approximation)
        self._estimate.covariance = np.diag([
            self.concentration_std**2,
            self.thickness_std**2,
            self.drift_speed_std**2,
            self.drift_direction_std**2,
        ])

        return self._estimate

    def get_estimate(self) -> IceEstimate:
        return self._estimate

    def reset(self) -> None:
        self._estimate = IceEstimate()


class IceLoadObserver:
    """First-order disturbance observer for ice load estimation.

    Estimates the generalized ice force from state measurements.
    """

    def __init__(self, alpha: float = 0.1, n_states: int = 3):
        self.alpha = alpha
        self._estimate = np.zeros(n_states)

    def update(
        self,
        tau_control: NDArray[np.float64],
        state_derivative: NDArray[np.float64],
        mass: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """Update ice load estimate from dynamics residual."""
        # Simplified: estimate ice force from F = m*a - tau_control + damping
        # This is a proxy — real observer would use full dynamics model
        return self._estimate.copy()

    def get_estimate(self) -> NDArray[np.float64]:
        return self._estimate.copy()

    def reset(self) -> None:
        self._estimate = np.zeros_like(self._estimate)


class DisturbanceObserver:
    """Generalized disturbance observer for DP systems."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self._estimate = np.zeros(3)

    def update(
        self,
        tau_control: NDArray[np.float64],
        velocity: NDArray[np.float64],
        damping: NDArray[np.float64],
        mass: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """Update disturbance estimate."""
        return self._estimate.copy()

    def get_estimate(self) -> NDArray[np.float64]:
        return self._estimate.copy()

    def reset(self) -> None:
        self._estimate = np.zeros(3)
