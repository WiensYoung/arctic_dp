"""HOCBF activation hysteresis and dwell-time state machine.

Implements:
1. Hysteresis-based HOCBF activation (dual thresholds)
2. Minimum dwell-time enforcement between mode transitions
3. Smooth risk scheduling (first-order filtered risk)
4. Mode transition diagnostics for paper reporting

Reference:
- Hespanha & Morse (1999) "Stability of switched systems with dwell-time"
- Ames et al. (2019) "Control Barrier Function based QPs"

This module addresses the chattering problem in safety filter activation
and provides the theoretical foundation for practical safety guarantees
under switching dynamics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional

import numpy as np


class SafetyMode(IntEnum):
    """Safety supervisor modes with ordered severity."""
    NORMAL = 0
    CAUTION = 1
    SAFETY_FILTER_ACTIVE = 2
    EMERGENCY_BACKUP = 3


@dataclass
class HOCBFActivationConfig:
    """Configuration for hysteresis-based HOCBF activation.

    Uses dual thresholds to prevent chattering:
    - Activate when h <= h_activate OR risk >= risk_activate
    - Deactivate only when h >= h_deactivate AND risk <= risk_deactivate
    - Enforce minimum dwell time between transitions
    """
    # HOCBF hysteresis thresholds
    h_activate: float = 4.0       # activate when h(x) <= this
    h_deactivate: float = 9.0     # deactivate only when h(x) >= this

    # Risk hysteresis thresholds
    risk_activate: float = 0.65   # activate when risk >= this
    risk_deactivate: float = 0.45 # deactivate only when risk <= this

    # Dwell time
    min_dwell_time_s: float = 5.0  # minimum time in each mode

    # Risk smoothing
    risk_time_constant_s: float = 1.0  # first-order filter time constant

    # Mode transition thresholds
    caution_risk_threshold: float = 0.35
    emergency_risk_threshold: float = 0.85


@dataclass
class ActivationState:
    """Current state of the HOCBF activation state machine."""
    mode: SafetyMode = SafetyMode.NORMAL
    hocbf_active: bool = False
    risk_filtered: float = 0.0
    time_in_mode: float = 0.0
    mode_switch_count: int = 0
    dwell_time_violation_count: int = 0
    last_switch_time: float = 0.0
    activation_reason: str = "none"

    # Diagnostic history (bounded deques avoid O(n) pop(0))
    mode_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    risk_raw_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    risk_filtered_history: deque = field(default_factory=lambda: deque(maxlen=1000))


class HOCBFActivationStateMachine:
    """Hysteresis-based HOCBF activation with dwell-time enforcement.

    Implements a 4-mode safety supervisor:
    - NORMAL: standard operation, no safety filter
    - CAUTION: elevated monitoring, risk scheduling begins
    - SAFETY_FILTER_ACTIVE: HOCBF QP is active
    - EMERGENCY_BACKUP: maximum safety intervention

    The state machine enforces:
    1. Dual-threshold hysteresis to prevent chattering
    2. Minimum dwell time between transitions
    3. Smooth risk scheduling via first-order filtering
    4. Ordered mode transitions (can only increase by 1, decrease by 1)
    """

    def __init__(self, config: Optional[HOCBFActivationConfig] = None):
        self.config = config or HOCBFActivationConfig()
        self.state = ActivationState()
        self._prev_mode = SafetyMode.NORMAL

    def update(
        self,
        h_val: float,
        risk_raw: float,
        dt: float,
        current_time: float = 0.0,
    ) -> ActivationState:
        """Update activation state based on current h(x) and risk.

        Args:
            h_val: HOCBF safety function h(x) = R^2 - ||p-p_ref||^2
            risk_raw: Raw risk estimate (0-1)
            dt: Time step (s)
            current_time: Current simulation time (s)

        Returns:
            Updated activation state
        """
        cfg = self.config
        state = self.state

        # 1. Smooth risk scheduling (first-order filter)
        if cfg.risk_time_constant_s > 0:
            beta = dt / (cfg.risk_time_constant_s + dt)
        else:
            beta = 1.0
        state.risk_filtered = (1.0 - beta) * state.risk_filtered + beta * risk_raw

        # 2. Determine desired mode based on hysteresis
        desired_mode = self._compute_desired_mode(h_val, state.risk_filtered)

        # 3. Enforce dwell time
        state.time_in_mode += dt
        if desired_mode != state.mode:
            if state.time_in_mode < cfg.min_dwell_time_s:
                # Dwell time not satisfied — stay in current mode
                state.dwell_time_violation_count += 1
                desired_mode = state.mode
            else:
                # Enforce ordered transitions: can only change by 1 step
                current_val = int(state.mode)
                desired_val = int(desired_mode)
                if abs(desired_val - current_val) > 1:
                    # Move one step toward desired
                    if desired_val > current_val:
                        desired_mode = SafetyMode(current_val + 1)
                    else:
                        desired_mode = SafetyMode(current_val - 1)

        # 4. Apply mode transition
        if desired_mode != state.mode:
            self._prev_mode = state.mode
            state.mode = desired_mode
            state.time_in_mode = 0.0
            state.mode_switch_count += 1
            state.last_switch_time = current_time

        # 5. Determine HOCBF activation
        state.hocbf_active = state.mode >= SafetyMode.SAFETY_FILTER_ACTIVE

        # 6. Record activation reason
        state.activation_reason = self._describe_activation_reason(
            h_val, state.risk_filtered, state.mode
        )

        # 7. Record history (deque maxlen handles eviction automatically)
        state.risk_raw_history.append(risk_raw)
        state.risk_filtered_history.append(state.risk_filtered)
        state.mode_history.append(int(state.mode))

        return state

    def _compute_desired_mode(self, h_val: float, risk_filtered: float) -> SafetyMode:
        """Compute desired mode using hysteresis logic.

        Hysteresis prevents chattering:
        - Activate when h <= h_activate OR risk >= risk_activate
        - Deactivate only when h >= h_deactivate AND risk <= risk_deactivate
        """
        cfg = self.config
        current = self.state.mode

        if current == SafetyMode.NORMAL:
            # Escalate to CAUTION if risk is elevated
            if risk_filtered >= cfg.caution_risk_threshold:
                return SafetyMode.CAUTION
            return SafetyMode.NORMAL

        elif current == SafetyMode.CAUTION:
            # Escalate to SAFETY_FILTER_ACTIVE if hysteresis triggers
            if h_val <= cfg.h_activate or risk_filtered >= cfg.risk_activate:
                return SafetyMode.SAFETY_FILTER_ACTIVE
            # De-escalate to NORMAL if risk is low
            if risk_filtered < cfg.caution_risk_threshold * 0.8:
                return SafetyMode.NORMAL
            return SafetyMode.CAUTION

        elif current == SafetyMode.SAFETY_FILTER_ACTIVE:
            # Escalate to EMERGENCY if risk is very high
            if risk_filtered >= cfg.emergency_risk_threshold:
                return SafetyMode.EMERGENCY_BACKUP
            # De-escalate to CAUTION only with hysteresis
            if h_val >= cfg.h_deactivate and risk_filtered <= cfg.risk_deactivate:
                return SafetyMode.CAUTION
            return SafetyMode.SAFETY_FILTER_ACTIVE

        elif current == SafetyMode.EMERGENCY_BACKUP:
            # De-escalate to SAFETY_FILTER_ACTIVE with hysteresis
            if risk_filtered < cfg.emergency_risk_threshold * 0.9:
                return SafetyMode.SAFETY_FILTER_ACTIVE
            return SafetyMode.EMERGENCY_BACKUP

        return current

    def _describe_activation_reason(
        self, h_val: float, risk_filtered: float, mode: SafetyMode
    ) -> str:
        """Describe why the current mode was activated."""
        if mode == SafetyMode.NORMAL:
            return "normal_operation"
        elif mode == SafetyMode.CAUTION:
            return f"caution_risk={risk_filtered:.2f}"
        elif mode == SafetyMode.SAFETY_FILTER_ACTIVE:
            if h_val <= self.config.h_activate:
                return f"hocbf_h_trigger_h={h_val:.1f}"
            elif risk_filtered >= self.config.risk_activate:
                return f"hocbf_risk_trigger_r={risk_filtered:.2f}"
            return "hocbf_active"
        elif mode == SafetyMode.EMERGENCY_BACKUP:
            return f"emergency_risk={risk_filtered:.2f}"
        return "unknown"

    def get_risk_scale(self) -> float:
        """Get risk-dependent scaling factor for HOCBF gains.

        Returns a multiplier >= 1.0 that increases HOCBF aggressiveness
        as risk increases. Used to modulate alpha1, alpha2 in the QP.
        """
        risk = self.state.risk_filtered
        # Smooth scaling: 1.0 at risk=0, up to 2.0 at risk=1
        return 1.0 + 1.0 * float(np.clip(risk, 0.0, 1.0))

    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic metrics for trace logging."""
        state = self.state
        return {
            "safety_mode": float(int(state.mode)),
            "safety_mode_name": state.mode.name,
            "hocbf_active": 1.0 if state.hocbf_active else 0.0,
            "risk_filtered": float(state.risk_filtered),
            "time_in_mode_s": float(state.time_in_mode),
            "mode_switch_count": float(state.mode_switch_count),
            "dwell_time_violation_count": float(state.dwell_time_violation_count),
            "risk_scale": float(self.get_risk_scale()),
            "activation_reason": state.activation_reason,
        }

    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary metrics for paper tables."""
        state = self.state
        mode_hist = np.array(state.mode_history) if state.mode_history else np.array([0])
        risk_raw_hist = np.array(state.risk_raw_history) if state.risk_raw_history else np.array([0])
        risk_filt_hist = np.array(state.risk_filtered_history) if state.risk_filtered_history else np.array([0])

        # Chattering index: number of mode transitions per unit time
        n_steps = max(1, len(mode_hist))
        transitions = np.sum(np.diff(mode_hist) != 0) if len(mode_hist) > 1 else 0

        return {
            "mode_switch_count": float(state.mode_switch_count),
            "min_mode_dwell_time_s": float(state.time_in_mode),
            "dwell_time_violation_count": float(state.dwell_time_violation_count),
            "hocbf_active_rate": float(np.mean(mode_hist >= int(SafetyMode.SAFETY_FILTER_ACTIVE))),
            "hocbf_activation_count": float(transitions),
            "chattering_index": float(transitions) / max(1.0, n_steps * 0.1),
            "risk_raw_mean": float(np.mean(risk_raw_hist)),
            "risk_filtered_mean": float(np.mean(risk_filt_hist)),
        }

    def reset(self) -> None:
        """Reset state machine to initial state."""
        self.state = ActivationState()
        self._prev_mode = SafetyMode.NORMAL
