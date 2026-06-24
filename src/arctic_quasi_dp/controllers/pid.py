"""PID 基线控制器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .base import BaseController, ControllerResult
from ..utils.math_utils import deg2rad, wrap_to_pi


@dataclass
class PIDParams:
    Kp_pos: float = 150.0
    Kd_pos: float = 90.0
    Ki_pos: float = 0.2
    Kp_heading: float = 600.0
    Kd_heading: float = 260.0
    Ki_heading: float = 0.5
    max_force: float = 1500.0
    max_moment: float = 20000.0


class PIDController(BaseController):
    """经典 PID 控制器。"""

    def __init__(self, params: Optional[PIDParams] = None):
        super().__init__()
        self.params = params or PIDParams()
        self._solver_label = "pid"
        self._int_pos = np.zeros(2, dtype=np.float64)
        self._int_psi = 0.0

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        self._target_pos = (float(x), float(y))
        self._target_psi = deg2rad(float(psi_deg))
        self.target_position = self._target_pos
        self.target_heading = self._target_psi
        self._int_pos[:] = 0.0
        self._int_psi = 0.0

    def compute_control(
        self,
        state: NDArray[np.float64],
        reference: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ControllerResult:
        dt = float(kwargs.get("dt", 0.1))
        if not hasattr(self, "_target_pos") or self._target_pos is None:
            return ControllerResult(tau=np.zeros(3), feasible=True, mode="pid", risk=0.0)

        x, y, psi = float(state[0]), float(state[1]), float(state[2])
        u, v, r = float(state[3]), float(state[4]), float(state[5])
        p = self.params

        pos_err_ned = np.array([x - self._target_pos[0], y - self._target_pos[1]])
        e_psi = wrap_to_pi(psi - self._target_psi)

        cpsi, spsi = np.cos(psi), np.sin(psi)
        # R: NED-to-body, R^T: body-to-NED
        vel_ned = np.array([cpsi * u - spsi * v, spsi * u + cpsi * v])

        self._int_pos = np.clip(self._int_pos + pos_err_ned * dt, -20.0, 20.0)
        self._int_psi = float(np.clip(self._int_psi + e_psi * dt, -20.0, 20.0))

        force_ned = -(p.Kp_pos * pos_err_ned + p.Kd_pos * vel_ned + p.Ki_pos * self._int_pos)
        # NED-to-body: force_body = R @ force_ned
        force_body = np.array([cpsi * force_ned[0] + spsi * force_ned[1],
                               -spsi * force_ned[0] + cpsi * force_ned[1]])

        moment = -(p.Kp_heading * e_psi + p.Kd_heading * r + p.Ki_heading * self._int_psi)

        tau = np.array([force_body[0], force_body[1], moment])
        tau[0] = np.clip(tau[0], -p.max_force, p.max_force)
        tau[1] = np.clip(tau[1], -p.max_force, p.max_force)
        tau[2] = np.clip(tau[2], -p.max_moment, p.max_moment)

        pos_err = float(np.linalg.norm(pos_err_ned))
        self._last_diagnostics = {"solver_status": "pid", "solver_success": True}
        return ControllerResult(tau=tau, feasible=True, mode="pid", risk=min(1.0, pos_err / 20.0))

    def reset(self) -> None:
        super().reset()
        self._int_pos[:] = 0.0
        self._int_psi = 0.0
