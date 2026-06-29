"""数学工具函数。"""

from __future__ import annotations

import math


def deg2rad(deg: float) -> float:
    """角度转弧度。"""
    return deg * math.pi / 180.0


def wrap_to_pi(angle: float) -> float:
    """将角度归一化到 [-pi, pi]。"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def wrap_angle_deg(angle_deg: float) -> float:
    """将角度归一化到 [-180, 180)。"""
    return (angle_deg + 180.0) % 360.0 - 180.0


def shortest_angle_diff_deg(target_deg: float, source_deg: float) -> float:
    """返回 target - source 的最短角度差, 结果在 [-180, 180)。"""
    return wrap_angle_deg(target_deg - source_deg)


def angle_ema_deg(prev_deg: float, new_deg: float, alpha: float) -> float:
    """角度感知的指数移动平均 (EMA)。

    使用最短弧度差计算, 正确处理 ±180° 环绕边界。
    例如: prev=179°, new=-179°, alpha=0.5 → ~180° (而非 0°)。
    """
    delta = shortest_angle_diff_deg(new_deg, prev_deg)
    return wrap_angle_deg(prev_deg + alpha * delta)
