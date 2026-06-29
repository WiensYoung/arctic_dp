"""Dimensionless scale analysis for proxy-to-full-scale comparability.

Computes dimensionless groups that characterize the relative magnitudes
of ice disturbance, actuator capacity, sensor noise, and control authority.
These ratios allow readers to assess whether proxy-scale benchmark results
are representative of full-scale behavior.

NOT a full-scale DP3 validation. This module provides quantitative
scale comparability metrics only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DimensionlessGroups:
    """Dimensionless ratios for scale comparability assessment."""

    ice_force_to_max_thrust: float
    """Ratio of representative ice force to maximum thrust capacity.
    Values > 1 mean ice can overpower actuators."""

    ice_moment_to_max_moment: float
    """Ratio of representative ice moment to maximum yaw moment capacity."""

    drift_speed_to_characteristic_speed: float
    """Ratio of ice drift speed to sqrt(g*L) characteristic speed.
    Indicates Froude number regime."""

    sensor_noise_to_safe_radius: float
    """Ratio of position sensor noise (1-sigma) to safe region radius.
    Values > 0.1 mean noise is significant relative to safety margin."""

    safe_radius_to_length: float
    """Ratio of safe region radius to vessel length.
    Indicates tightness of positioning requirement."""

    disturbance_time_to_control_time: float
    """Ratio of ice condition change timescale to control period.
    Values < 1 mean ice changes faster than control can respond."""

    power_cap_to_nominal_power: float
    """Ratio of total power cap to estimated nominal power demand."""

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of scale regime."""
        issues = []
        if self.ice_force_to_max_thrust > 0.8:
            issues.append("ice_force_near_actuator_limit")
        if self.sensor_noise_to_safe_radius > 0.1:
            issues.append("noise_significant_vs_safety_margin")
        if self.drift_speed_to_characteristic_speed > 0.5:
            issues.append("high_froude_number")
        if not issues:
            return "proxy_scale_comparable"
        return "scale_caution: " + ", ".join(issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ice_force_to_max_thrust": self.ice_force_to_max_thrust,
            "ice_moment_to_max_moment": self.ice_moment_to_max_moment,
            "drift_speed_to_characteristic_speed": self.drift_speed_to_characteristic_speed,
            "sensor_noise_to_safe_radius": self.sensor_noise_to_safe_radius,
            "safe_radius_to_length": self.safe_radius_to_length,
            "disturbance_time_to_control_time": self.disturbance_time_to_control_time,
            "power_cap_to_nominal_power": self.power_cap_to_nominal_power,
            "interpretation": self.interpretation,
        }


def _representative_ice_force(
    concentration: float,
    thickness: float,
    drift_speed: float,
    crushing_strength_mpa: float,
    structure_factor: float,
    vessel_beam_m: float,
    waterline_angle_deg: float,
) -> float:
    """Compute representative ice force magnitude (N) using Lindqvist proxy."""
    c = max(0.0, min(1.0, concentration))
    h = max(0.0, thickness)
    v = max(0.0, drift_speed)
    import math
    alpha = math.radians(waterline_angle_deg)
    speed_factor = 1.0 + 0.5 * v / (v + 0.5) if v > 0 else 1.0
    angle_factor = 1.0 + 0.3 * math.tan(min(alpha, math.pi / 3.0))
    return crushing_strength_mpa * 1e6 * h * vessel_beam_m * structure_factor * speed_factor * angle_factor * c


def compute_dimensionless_groups(
    vessel_params: Any,
    thruster_config: Any,
    scenario: Any,
    ice_state: Optional[Dict[str, float]] = None,
    control_dt: float = 0.1,
    sensor_noise_std_m: float = 0.5,
) -> DimensionlessGroups:
    """Compute dimensionless scale comparability groups.

    Args:
        vessel_params: VesselParams or VesselConfigBundle
        thruster_config: ThrusterConfig (or None for no-thruster scenarios)
        scenario: SCI1Scenario
        ice_state: dict with concentration, thickness, drift_speed (or None to use scenario defaults)
        control_dt: control period (s)
        sensor_noise_std_m: position sensor noise std (m)

    Returns:
        DimensionlessGroups with all ratios computed
    """
    import math

    # Extract vessel parameters
    if hasattr(vessel_params, 'vessel_params'):
        vp = vessel_params.vessel_params
    else:
        vp = vessel_params

    mass = getattr(vp, 'mass', 500000.0)
    length = getattr(vp, 'length', 122.5)
    beam = getattr(vp, 'beam', 22.0)
    crushing_mpa = getattr(vp, 'ice_crushing_strength_mpa', 0.0003)
    structure_factor = getattr(vp, 'ice_structure_factor', 0.45)
    waterline_deg = getattr(vp, 'waterline_angle_deg', 30.0)

    # Ice conditions (from scenario or provided)
    if ice_state is None:
        ice_state = {
            "concentration": getattr(scenario, 'ice_concentration', 0.5),
            "thickness": getattr(scenario, 'ice_thickness', 0.8),
            "drift_speed": getattr(scenario, 'ice_drift_speed', 0.3),
        }

    # Representative ice force
    ice_force = _representative_ice_force(
        ice_state["concentration"], ice_state["thickness"], ice_state["drift_speed"],
        crushing_mpa, structure_factor, beam, waterline_deg,
    )

    # Thruster capacity
    max_thrust = 0.0
    max_moment = 0.0
    if thruster_config is not None:
        for t in thruster_config.thrusters:
            if not t.faulty:
                max_thrust += t.max_thrust * t.degraded
                # Moment contribution: F * lever_arm
                lever = math.sqrt(t.x ** 2 + t.y ** 2)
                max_moment += t.max_thrust * t.degraded * lever
    else:
        max_thrust = 3000.0  # default proxy
        max_moment = 100000.0

    max_thrust = max(max_thrust, 1.0)
    max_moment = max(max_moment, 1.0)

    # Characteristic speed: sqrt(g * L)
    g = 9.81
    char_speed = math.sqrt(g * length)

    # Safe region radius
    safe_radius = getattr(scenario, 'safe_region_radius', 10.0)

    # Power cap
    power_cap_kw = 0.0
    if thruster_config is not None:
        power_cap_kw = getattr(thruster_config, 'max_total_power_kw', 0.0)
    # Estimate nominal power: sum of max_thrust^1.5 / sqrt(max_thrust) for each thruster
    nominal_power_kw = 0.0
    if thruster_config is not None:
        for t in thruster_config.thrusters:
            if not t.faulty:
                # Simplified: P ~ F^1.5 / sqrt(F_max)
                nominal_power_kw += (t.max_thrust * 0.5) ** 1.5 / max(math.sqrt(t.max_thrust), 1.0) * 0.001
    power_ratio = power_cap_kw / max(nominal_power_kw, 1.0) if power_cap_kw > 0 else float("inf")

    # Disturbance timescale: ice_drift_speed / length (rough)
    # Control time: control_dt
    # For step changes, use scenario duration / 10 as rough timescale
    duration = getattr(scenario, 'duration', 300.0)
    disturbance_time = duration / 10.0  # rough: ice changes over ~10% of duration
    disturbance_ratio = disturbance_time / max(control_dt, 0.01)

    return DimensionlessGroups(
        ice_force_to_max_thrust=ice_force / max_thrust,
        ice_moment_to_max_moment=(ice_force * length * 0.18) / max_moment,
        drift_speed_to_characteristic_speed=ice_state["drift_speed"] / max(char_speed, 0.01),
        sensor_noise_to_safe_radius=sensor_noise_std_m / max(safe_radius, 0.1),
        safe_radius_to_length=safe_radius / max(length, 1.0),
        disturbance_time_to_control_time=disturbance_ratio,
        power_cap_to_nominal_power=power_ratio,
    )
