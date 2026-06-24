"""数学工具函数。"""

from __future__ import annotations

import math


def deg2rad(deg: float) -> float:
    """角度转弧度。"""
    return deg * math.pi / 180.0


def wrap_to_pi(angle: float) -> float:
    """将角度归一化到 [-pi, pi]。"""
    return (angle + math.pi) % (2 * math.pi) - math.pi
