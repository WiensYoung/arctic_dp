"""Base controller interface.

所有 DP 控制器的基类，定义了标准接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class ControllerResult:
    """控制器输出结果。"""
    tau: NDArray[np.float64]            # [Fx, Fy, Mz] 船体坐标系
    feasible: bool = True               # 求解是否可行
    mode: str = "unknown"               # 当前模式名
    risk: float = 0.0                   # 风险指标 [0, 1]
    cost_estimate: float = 0.0          # 代价函数估计值


class BaseController:
    """DP 控制器基类。

    子类需实现 compute_control() 方法。
    """

    def __init__(self):
        self._solver_label: str = "base"
        self._last_diagnostics: Dict[str, Any] = {}
        self.target_position: Optional[tuple] = None
        self.target_heading: float = 0.0

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        """设置目标位置和艏向。"""
        pass

    def set_ice_conditions(
        self,
        ice_concentration: float,
        ice_thickness: float,
        ice_drift_speed: float,
        ice_drift_direction: float = 0.0,
    ) -> None:
        """设置冰况参数。"""
        pass

    def set_safe_region_radius(self, radius: float) -> None:
        """设置安全区域半径。默认空实现。"""
        pass

    def compute_control(
        self,
        state: NDArray[np.float64],
        reference: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ControllerResult:
        """计算控制输出。

        Args:
            state: [x, y, psi, u, v, r] (NED位置 + 艏向 + 船体速度 + 偏航率)
            reference: 可选参考轨迹
            environment: 可选环境信息
            **kwargs: dt 等额外参数

        Returns:
            ControllerResult
        """
        raise NotImplementedError

    def get_diagnostics(self) -> Dict[str, Any]:
        """返回最近一次 compute_control 的诊断信息。"""
        return dict(self._last_diagnostics)

    def reset(self) -> None:
        """重置控制器内部状态。"""
        self._last_diagnostics = {}
