"""共享的 Lindqvist 1989 简化冰力模型。

本模块是冰力计算的唯一真实来源 (single source of truth)，
消除 sim_loop.py、controllers.py、ice_models.py 之间的代码重复。

所有冰力计算应调用本模块的函数，而非各自内联实现。

Reference:
    Lindqvist, G. (1989). "A straightforward method for calculation of ice
    resistance of ships." POAC Conference.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from numpy.typing import NDArray


# 冰力模型常量
_LEVER_FRACTION = 0.18  # 力矩臂 = fraction * vessel_length (纵向)
_LATERAL_LEVER_FRACTION = 0.05  # 横向力臂 = fraction * vessel_beam (斜向冰载荷偏心)
_SPEED_V_REF = 0.5      # 速度因子参考速度 (m/s)
_ANGLE_ALPHA_MAX = math.pi / 3.0  # 水线角上限 (60°)
_ANGLE_TAN_FACTOR = 0.3  # 水线角因子系数


def lindqvist_speed_factor(drift_speed: float) -> float:
    """Lindqvist 速度因子: 1 + 0.5*v/(v+0.5)。

    Args:
        drift_speed: 冰漂移速度 (m/s), 非负

    Returns:
        速度因子 (>= 1.0)
    """
    v = max(0.0, float(drift_speed))
    if v > 0:
        return 1.0 + 0.5 * v / (v + _SPEED_V_REF)
    return 1.0


def lindqvist_angle_factor(waterline_angle_rad: float) -> float:
    """Lindqvist 水线角因子: 1 + 0.3*tan(min(alpha, 60°))。

    Args:
        waterline_angle_rad: 水线角 (rad)

    Returns:
        角度因子 (>= 1.0)
    """
    alpha = min(float(waterline_angle_rad), _ANGLE_ALPHA_MAX)
    return 1.0 + _ANGLE_TAN_FACTOR * math.tan(alpha)


def compute_ice_force_ned(
    concentration: float,
    thickness: float,
    drift_speed: float,
    drift_direction_deg: float,
    crushing_strength_mpa: float,
    vessel_beam_m: float,
    structure_factor: float,
    waterline_angle_rad: float,
) -> NDArray[np.float64]:
    """计算 NED 坐标系下的冰力 [Fx_ned, Fy_ned]。

    Args:
        concentration: 冰密集度 [0, 1]
        thickness: 冰厚度 (m)
        drift_speed: 冰漂移速度 (m/s)
        drift_direction_deg: 漂移方向 (度, 0=北, 顺时针)
        crushing_strength_mpa: 冰破碎强度 (MPa)
        vessel_beam_m: 船宽 (m)
        structure_factor: Lindqvist 结构因子
        waterline_angle_rad: 水线角 (rad)

    Returns:
        NED冰力向量 [Fx, Fy] (N)
    """
    c = float(np.clip(concentration, 0.0, 1.0))
    h = max(0.0, float(thickness))
    v = max(0.0, float(drift_speed))

    speed_factor = lindqvist_speed_factor(v)
    angle_factor = lindqvist_angle_factor(waterline_angle_rad)

    base_force = (
        crushing_strength_mpa * 1e6  # MPa -> Pa
        * h * vessel_beam_m * structure_factor
        * speed_factor * angle_factor * c
    )

    dir_rad = math.radians(drift_direction_deg)
    fx_ned = base_force * math.cos(dir_rad)
    fy_ned = base_force * math.sin(dir_rad)

    return np.array([fx_ned, fy_ned], dtype=np.float64)


def compute_ice_force_body(
    concentration: float,
    thickness: float,
    drift_speed: float,
    drift_direction_deg: float,
    vessel_psi: float,
    crushing_strength_mpa: float,
    vessel_beam_m: float,
    vessel_length_m: float,
    structure_factor: float,
    waterline_angle_rad: float,
) -> NDArray[np.float64]:
    """计算船体坐标系下的冰力 [Fx_body, Fy_body, Mz_body]。

    Args:
        concentration: 冰密集度 [0, 1]
        thickness: 冰厚度 (m)
        drift_speed: 冰漂移速度 (m/s)
        drift_direction_deg: 漂移方向 (度, 0=北, 顺时针)
        vessel_psi: 船舶航向 (rad, NED)
        crushing_strength_mpa: 冰破碎强度 (MPa)
        vessel_beam_m: 船宽 (m)
        vessel_length_m: 船长 (m)
        structure_factor: Lindqvist 结构因子
        waterline_angle_rad: 水线角 (rad)

    Returns:
        船体冰力向量 [Fx, Fy, Mz] (N, N, N*m)
    """
    force_ned = compute_ice_force_ned(
        concentration, thickness, drift_speed, drift_direction_deg,
        crushing_strength_mpa, vessel_beam_m, structure_factor, waterline_angle_rad,
    )

    # NED -> body 旋转
    cpsi = math.cos(vessel_psi)
    spsi = math.sin(vessel_psi)
    fx_body = cpsi * force_ned[0] + spsi * force_ned[1]
    fy_body = -spsi * force_ned[0] + cpsi * force_ned[1]

    # 力矩: Mz = x_cp * Fy - y_cp * Fx
    # 纵向力臂 ~0.18*L (船首到重心), 横向力臂 ~0.05*beam (斜向冰载荷偏心)
    lever_x = _LEVER_FRACTION * vessel_length_m
    lever_y = _LATERAL_LEVER_FRACTION * vessel_beam_m
    mz_body = lever_x * fy_body - lever_y * fx_body

    return np.array([fx_body, fy_body, mz_body], dtype=np.float64)


def compute_ice_force_body_from_dict(
    ice: Dict[str, float],
    vessel_psi: float,
    crushing_strength_mpa: float,
    vessel_beam_m: float,
    vessel_length_m: float,
    structure_factor: float,
    waterline_angle_rad: float,
) -> NDArray[np.float64]:
    """从字典格式的冰况计算船体冰力 (向后兼容)。

    Args:
        ice: 冰况字典, 包含 concentration, thickness, drift_speed, drift_direction
        其余参数同 compute_ice_force_body

    Returns:
        船体冰力向量 [Fx, Fy, Mz] (N, N, N*m)
    """
    return compute_ice_force_body(
        concentration=float(ice.get("concentration", 0.0)),
        thickness=float(ice.get("thickness", 0.0)),
        drift_speed=float(ice.get("drift_speed", 0.0)),
        drift_direction_deg=float(ice.get("drift_direction", 0.0)),
        vessel_psi=vessel_psi,
        crushing_strength_mpa=crushing_strength_mpa,
        vessel_beam_m=vessel_beam_m,
        vessel_length_m=vessel_length_m,
        structure_factor=structure_factor,
        waterline_angle_rad=waterline_angle_rad,
    )
