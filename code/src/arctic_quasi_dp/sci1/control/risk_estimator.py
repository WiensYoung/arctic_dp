"""CVaR / tail-risk proxy estimator.

This is a proxy-scale risk estimator using the same formulation
as the existing controllers.py _cvar_proxy. It does NOT implement
formal stochastic CVaR-constrained control.

For documentation, use: "tail-risk proxy / CVaR-style risk estimator"
NOT: "formal CVaR-constrained stochastic control"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ..cvar_sampling import compute_cvar_vectorized, get_cvar_samples_for_profile


@dataclass
class RiskEstimate:
    """Risk estimate output."""
    risk_level: float = 0.0     # [0, 1] overall risk
    cvar_proxy: float = 0.0     # [0, 1] CVaR-style tail risk
    ice_risk: float = 0.0       # [0, 1] ice condition risk
    position_risk: float = 0.0  # [0, 1] position error risk


class ProxyCVaRRiskEstimator:
    """Proxy CVaR / tail-risk estimator.

    Uses the same risk formulation as IceAwarePrecisionDPController.
    Not a formal stochastic CVaR estimator.

    E8: Supports profile-based sample counts:
    - smoke: 32 samples
    - paper_small: 256 samples
    - paper_full: 512 samples
    """

    def __init__(
        self,
        cvar_alpha: float = 0.90,
        cvar_samples: int = 64,
        cvar_sigma_force: float = 150.0,
        cbf_radius: float = 10.0,
        profile: str = "smoke",
    ):
        self.cvar_alpha = cvar_alpha
        # E8: use profile-based sample count if cvar_samples not explicitly set
        self.cvar_samples = cvar_samples if cvar_samples != 64 else get_cvar_samples_for_profile(profile)
        self.cvar_sigma_force = cvar_sigma_force
        self.cbf_radius = cbf_radius
        self._rng = np.random.default_rng(2026)
        self._last_cvar_diagnostics: Dict = {}

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def estimate(
        self,
        state: NDArray[np.float64],
        target_pos: tuple,
        ice: Dict[str, float],
        tau: NDArray[np.float64],
    ) -> RiskEstimate:
        """Estimate risk from current state.

        Args:
            state: [x, y, psi, u, v, r]
            target_pos: (x, y) target
            ice: ice condition dict with concentration, thickness, drift_speed
            tau: current control input

        Returns:
            RiskEstimate
        """
        state = np.asarray(state, dtype=np.float64).reshape(6,)
        tau = np.asarray(tau, dtype=np.float64).reshape(3,)

        c = float(np.clip(ice.get("concentration", 0.0), 0.0, 1.0))
        h = max(0.0, float(ice.get("thickness", 0.0)))
        v = max(0.0, float(ice.get("drift_speed", 0.0)))

        # Position risk
        pos_err = float(np.linalg.norm(state[:2] - np.array(target_pos)))
        position_risk = float(np.clip(min(1.0, pos_err / 15.0), 0.0, 1.0))

        # Ice risk (shared formula)
        from ..controllers import _ice_risk_standardized
        ice_risk = _ice_risk_standardized(c, h, v)

        # E8: CVaR proxy with profile-based sampling and diagnostics
        alpha = float(np.clip(self.cvar_alpha, 0.5, 0.99))
        n = int(max(8, self.cvar_samples))
        sigma = self.cvar_sigma_force * c * (0.3 + h) * (0.4 + v)
        draws = self._rng.normal(0.0, sigma, size=n)

        max_f = 3000.0  # proxy-scale max force
        # Per-sample control saturation with actuator noise (consistent with controllers.py)
        actuator_noise = self._rng.normal(0.0, 0.02 * max_f, size=n)
        control_sat = np.clip((np.linalg.norm(tau[:2]) + np.abs(actuator_noise)) / max_f, 0.0, 1.5)
        ice_disturbance = np.abs(draws) / max_f
        radius = max(self.cbf_radius, 1.0)
        violation = max(0.0, pos_err - radius) / radius

        losses = 0.4 * control_sat + 0.3 * ice_disturbance + 0.3 * violation

        # E8: use vectorized CVaR computation with diagnostics
        cvar_result = compute_cvar_vectorized(losses, alpha)
        cvar_proxy = float(np.clip(cvar_result.cvar, 0.0, 1.0))

        # Store CVaR diagnostics for reporting
        self._last_cvar_diagnostics = {
            "cvar_alpha": alpha,
            "cvar_n_samples": cvar_result.n_samples,
            "cvar_n_tail": cvar_result.n_tail,
            "cvar_standard_error": cvar_result.cvar_standard_error,
            "cvar_relative_se": cvar_result.cvar_relative_se,
            "cvar_converged": cvar_result.cvar_converged,
            "cvar_var": cvar_result.var,
            "cvar_mean": cvar_result.mean,
        }

        # Overall risk (shared three-factor formula)
        from ..controllers import compute_total_risk
        risk_level = compute_total_risk(pos_err, c, h, v, cvar_proxy)

        return RiskEstimate(
            risk_level=risk_level,
            cvar_proxy=cvar_proxy,
            ice_risk=ice_risk,
            position_risk=position_risk,
        )
