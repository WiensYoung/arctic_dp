"""滑模控制器 (SMC) 基线 — 简化实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .base import BaseController, ControllerResult
from ..utils.math_utils import deg2rad, wrap_to_pi


@dataclass
class SMCParams:
    lambda_s: float = 0.5
    eta: float = 100.0
    phi: float = 0.1
    max_force: float = 1500.0
    max_moment: float = 20000.0


class SMCController(BaseController):
    """滑模控制器。"""

    def __init__(self, params: Optional[SMCParams] = None):
        super().__init__()
        self.params = params or SMCParams()
        self._solver_label = "smc"

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        self._target_pos = (float(x), float(y))
        self._target_psi = deg2rad(float(psi_deg))
        self.target_position = self._target_pos
        self.target_heading = self._target_psi

    def compute_control(
        self,
        state: NDArray[np.float64],
        reference: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ControllerResult:
        if not hasattr(self, "_target_pos") or self._target_pos is None:
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="smc", risk=0.0)

        x, y, psi = float(state[0]), float(state[1]), float(state[2])
        u, v, r = float(state[3]), float(state[4]), float(state[5])
        p = self.params

        pos_err = np.array([x - self._target_pos[0], y - self._target_pos[1]])
        e_psi = wrap_to_pi(psi - self._target_psi)

        cpsi, spsi = np.cos(psi), np.sin(psi)
        # body-to-NED velocity transform
        vel_ned = np.array([cpsi * u - spsi * v, spsi * u + cpsi * v])

        # 滑模面
        sx = vel_ned[0] + p.lambda_s * pos_err[0]
        sy = vel_ned[1] + p.lambda_s * pos_err[1]
        sr = r + p.lambda_s * e_psi

        # 饱和函数
        def sat(s):
            return np.clip(s / p.phi, -1.0, 1.0)

        fx_ned = -p.eta * sat(sx) - p.lambda_s * vel_ned[0]
        fy_ned = -p.eta * sat(sy) - p.lambda_s * vel_ned[1]
        mz = -p.eta * sat(sr) - p.lambda_s * r

        # NED-to-body force transform
        force_body = np.array([cpsi * fx_ned + spsi * fy_ned,
                               -spsi * fx_ned + cpsi * fy_ned])
        tau = np.array([force_body[0], force_body[1], mz])
        tau[0] = np.clip(tau[0], -p.max_force, p.max_force)
        tau[1] = np.clip(tau[1], -p.max_force, p.max_force)
        tau[2] = np.clip(tau[2], -p.max_moment, p.max_moment)

        pos_dist = float(np.linalg.norm(pos_err))
        self._last_diagnostics = {"solver_status": "smc", "solver_success": True}
        return ControllerResult(tau=tau, feasible=True, mode="smc", risk=min(1.0, pos_dist / 20.0))

    def reset(self) -> None:
        super().reset()
