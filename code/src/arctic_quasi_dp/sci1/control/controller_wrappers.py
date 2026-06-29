"""Controller wrappers for safety filter integration.

Wraps any BaseController with an optional SoftHOCBFSafetyFilter.
The wrapper delegates nominal control to the inner controller,
then applies the safety filter to produce tau_safe.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ...controllers.base import BaseController, ControllerResult
from .safety_filter import (
    DisabledSafetyFilter,
    SafetyFilterResult,
    SoftHOCBFSafetyFilter,
)
from .risk_estimator import ProxyCVaRRiskEstimator, RiskEstimate


class SafetyFilteredController(BaseController):
    """Nominal controller + optional safety filter.

    Wraps any BaseController. If safety_filter is DisabledSafetyFilter,
    output is identical to nominal controller.
    """

    def __init__(
        self,
        nominal_controller: BaseController,
        safety_filter=None,
        risk_estimator: Optional[ProxyCVaRRiskEstimator] = None,
        risk_gain: float = 0.0,
    ):
        super().__init__()
        self.nominal = nominal_controller
        self.safety_filter = safety_filter or DisabledSafetyFilter()
        self.risk_estimator = risk_estimator
        self._risk_gain = risk_gain
        self._target_pos: Optional[Tuple[float, float]] = None
        self._target_psi: float = 0.0
        self._last_filter_result: Optional[SafetyFilterResult] = None
        self._last_risk: Optional[RiskEstimate] = None

    def set_target(self, x: float, y: float, psi_deg: float) -> None:
        self._target_pos = (float(x), float(y))
        self._target_psi = np.radians(float(psi_deg))
        self.nominal.set_target(x, y, psi_deg)
        self.target_position = self._target_pos
        self.target_heading = self._target_psi

    def set_ice_conditions(self, ice_concentration: float, ice_thickness: float,
                          ice_drift_speed: float, ice_drift_direction: float = 0.0) -> None:
        if hasattr(self.nominal, 'set_ice_conditions'):
            self.nominal.set_ice_conditions(
                ice_concentration, ice_thickness, ice_drift_speed, ice_drift_direction,
            )

    def set_safe_region_radius(self, radius: float) -> None:
        if hasattr(self.nominal, 'set_safe_region_radius'):
            self.nominal.set_safe_region_radius(radius)
        # Update HOCBF params if safety filter has them
        if hasattr(self.safety_filter, 'hocbf_params'):
            self.safety_filter.hocbf_params.safe_radius_m = float(radius)

    def set_cvar_seed(self, seed: int) -> None:
        if hasattr(self.nominal, 'set_cvar_seed'):
            self.nominal.set_cvar_seed(seed)
        if self.risk_estimator is not None:
            # 使用不同偏移避免 nominal 和 safety filter 的 RNG 产生相关样本
            self.risk_estimator.set_seed(seed + 1000000)

    def set_actuator_mode(self, mode: str, power_scale_factor: float = 1.0) -> None:
        """Forward discrete actuator mode to the safety filter if supported."""
        if hasattr(self.safety_filter, 'set_actuator_mode'):
            self.safety_filter.set_actuator_mode(mode, power_scale_factor=power_scale_factor)

    def set_vessel_params(self, vessel_params) -> None:
        """Forward vessel params to nominal controller."""
        if hasattr(self.nominal, 'set_vessel_params'):
            self.nominal.set_vessel_params(vessel_params)

    def compute_control(
        self,
        state: NDArray[np.float64],
        reference: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ControllerResult:
        dt = float(kwargs.get("dt", 0.1))

        # 1. Nominal control
        nominal_result = self.nominal.compute_control(state, reference, environment, **kwargs)
        tau_des = np.asarray(nominal_result.tau, dtype=np.float64).reshape(3,)

        # 2. Risk estimate
        risk_level = 0.0
        if self.risk_estimator is not None and self._target_pos is not None:
            ice = getattr(self.nominal, '_raw_ice',
                         {"concentration": 0.0, "thickness": 0.0, "drift_speed": 0.0, "drift_direction": 0.0})
            self._last_risk = self.risk_estimator.estimate(
                state, self._target_pos, ice, tau_des,
            )
            risk_level = self._last_risk.risk_level

        # 3. Safety filter
        if self._target_pos is not None:
            filter_result = self.safety_filter.filter(
                state=state,
                tau_des=tau_des,
                target_pos=self._target_pos,
                target_psi=self._target_psi,
                risk_level=risk_level,
                dt=dt,
            )
        else:
            filter_result = SafetyFilterResult(
                tau_des=tau_des.copy(), tau_safe=tau_des.copy(),
                status="no_target",
            )

        self._last_filter_result = filter_result

        # 4. Build result
        tau_safe = filter_result.tau_safe
        # Merge diagnostics
        diag = {}
        if hasattr(self.nominal, 'get_diagnostics'):
            diag = dict(self.nominal.get_diagnostics())
        diag.update(filter_result.to_dict())

        self._last_diagnostics = diag

        return ControllerResult(
            tau=tau_safe,
            feasible=nominal_result.feasible and filter_result.qp_success,
            mode=f"{nominal_result.mode}+sf",
            risk=nominal_result.risk,
            cost_estimate=nominal_result.cost_estimate,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        diag = {}
        if hasattr(self.nominal, 'get_diagnostics'):
            diag = dict(self.nominal.get_diagnostics())
        if self._last_filter_result is not None:
            diag.update(self._last_filter_result.to_dict())
        return diag

    def reset(self) -> None:
        if hasattr(self.nominal, 'reset'):
            self.nominal.reset()
        self._last_filter_result = None
        self._last_risk = None
        self._last_diagnostics = {}


def make_filtered_controller(
    nominal_controller: BaseController,
    filter_type: str = "disabled",
    hocbf_radius: float = 15.0,
    alpha1: float = 1.0,
    alpha2: float = 1.5,
    slack_weight: float = 10000.0,
    risk_gain: float = 0.0,
    max_force: float = 3000.0,
    max_moment: float = 100000.0,
    constraint_mode: str = "box",
    require_osqp: bool = False,
    disturbance_accel_bound_mps2: float = 0.0,
) -> SafetyFilteredController:
    """Factory for safety-filtered controllers.

    Args:
        nominal_controller: the base controller to wrap
        filter_type: "disabled", "fixed_soft_hocbf", "cvar_soft_hocbf", "no_safety_filter"
        hocbf_radius: HOCBF safe radius (m)
        alpha1, alpha2: HOCBF gains
        slack_weight: QP slack penalty
        risk_gain: CVaR risk modulation gain (0 = fixed, >0 = adaptive)
        max_force, max_moment: generalized-force limits
        constraint_mode: "box" or "polygon" proxy feasible set
        require_osqp: if true, report failure when OSQP is unavailable
        disturbance_accel_bound_mps2: optional bounded-disturbance robust-HOCBF margin

    Returns:
        SafetyFilteredController
    """
    from .hocbf import HOCBFParams

    if filter_type in ("disabled", "no_safety_filter"):
        return SafetyFilteredController(
            nominal_controller=nominal_controller,
            safety_filter=DisabledSafetyFilter(),
            risk_estimator=None,
        )

    hocbf_params = HOCBFParams(
        safe_radius_m=hocbf_radius,
        alpha1_base=alpha1,
        alpha2_base=alpha2,
        disturbance_accel_bound_mps2=disturbance_accel_bound_mps2,
    )

    safety_filter = SoftHOCBFSafetyFilter(
        hocbf_params=hocbf_params,
        max_force_x=max_force,
        max_force_y=max_force,
        max_moment_n=max_moment,
        slack_weight=slack_weight,
        risk_gain=risk_gain,
        constraint_mode=constraint_mode,
        require_osqp=require_osqp,
    )

    risk_estimator = ProxyCVaRRiskEstimator(cbf_radius=hocbf_radius)

    actual_risk_gain = risk_gain if filter_type == "cvar_soft_hocbf" else 0.0
    safety_filter.risk_gain = actual_risk_gain

    return SafetyFilteredController(
        nominal_controller=nominal_controller,
        safety_filter=safety_filter,
        risk_estimator=risk_estimator,
        risk_gain=actual_risk_gain,
    )
