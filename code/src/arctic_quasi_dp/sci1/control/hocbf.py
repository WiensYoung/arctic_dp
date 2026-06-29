"""HOCBF (Higher-Order Control Barrier Function) computation.

This is a proxy-scale relative-degree-2 implementation for the
simplified 3-DOF benchmark. It should not be interpreted as a
full-scale ship safety proof.

Safety function:
    h(x) = R_safe^2 - ||p - p_ref||^2

HOCBF (relative-degree-2):
    psi_hocbf = h_ddot + (alpha1 + alpha2) * h_dot + alpha1 * alpha2 * h

Linear constraint on tau:
    a_hocbf @ tau >= b_hocbf
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from ...utils.math_utils import wrap_to_pi


@dataclass
class HOCBFParams:
    """HOCBF parameters.

    ``disturbance_accel_bound_mps2`` is an optional bounded-disturbance
    robustness margin used by the theory-oriented controller variants.  It is
    interpreted as an upper bound on the unmodelled translational acceleration
    projected into the NED plane.  When positive, the HOCBF right-hand side is
    tightened by ``2 * ||p-p_ref|| * disturbance_accel_bound_mps2``.  This is a
    conservative relative-degree-2 bound for
    ``h_ddot = -2||v||^2 - 2 e^T a`` and should be reported as a proxy-scale
    bounded-disturbance certificate rather than a full-scale proof.
    """
    safe_radius_m: float = 15.0
    alpha1_base: float = 1.0
    alpha2_base: float = 1.5
    vessel_mass_kg: float = 500000.0
    activation_margin: float = 0.75  # activate HOCBF when dist > margin * R
    disturbance_accel_bound_mps2: float = 0.0
    max_certificate_slack: float = 1e-7
    min_certificate_margin: float = -1e-7


def compute_hocbf_constraint(
    state: NDArray[np.float64],
    tau_des: NDArray[np.float64],
    target_pos: Tuple[float, float],
    target_psi: float,
    params: HOCBFParams,
    alpha1: float,
    alpha2: float,
    dt: float,
) -> Dict[str, float]:
    """Compute HOCBF linear constraint coefficients.

    The constraint is: a_hocbf @ tau >= b_hocbf
    which is equivalent to: psi_hocbf >= 0 when slack = 0.

    Args:
        state: [x, y, psi, u, v, r]
        tau_des: desired control [Fx, Fy, Mz]
        target_pos: (x, y) target
        target_psi: target heading (rad)
        params: HOCBF parameters
        alpha1, alpha2: HOCBF gains (may be risk-modulated)
        dt: timestep

    Returns:
        dict with a_hocbf (3,), b_hocbf (scalar), h_val, psi_hocbf
    """
    state = np.asarray(state, dtype=np.float64).reshape(6,)
    tau_des = np.asarray(tau_des, dtype=np.float64).reshape(3,)

    # Extract state
    px, py = float(state[0]), float(state[1])
    psi = float(state[2])
    u, v, r = float(state[3]), float(state[4]), float(state[5])

    # Position error in NED (防御性: 如果 target_pos 为 None, 约束自动满足)
    if target_pos is None:
        return {"a_hocbf_x": 0.0, "a_hocbf_y": 0.0, "a_hocbf_mz": 0.0,
                "b_hocbf": -1e6, "h_val": 1e6, "slack_min": 0.0}
    e_x = px - target_pos[0]
    e_y = py - target_pos[1]
    dist_sq = e_x ** 2 + e_y ** 2

    # Velocity in NED
    cpsi, spsi = np.cos(psi), np.sin(psi)
    vx_ned = cpsi * u - spsi * v
    vy_ned = spsi * u + cpsi * v

    # h(x) = R^2 - ||p - p_ref||^2
    R = params.safe_radius_m
    h_val = R ** 2 - dist_sq

    # h_dot = -2 * e^T * v_ned
    h_dot = -2.0 * (e_x * vx_ned + e_y * vy_ned)

    # For the QP constraint, we need the linear dependence on tau:
    # h_ddot depends on accel, which depends on tau.
    # h_ddot = -2 * ||v_ned||^2 - 2 * e^T * a_ned
    # a_ned = R_body2ned @ (tau / mass + disturbance/mass)
    # So: h_ddot = -2*||v||^2 - 2*e^T*R_body2ned@(tau/mass) + disturbance_terms
    # The tau-dependent part: -2 * e^T * R_body2ned / mass

    # Linear coefficient on tau (in body frame)
    mass = params.vessel_mass_kg
    # e^T * R_body2ned = [e_x*cpsi + e_y*spsi, -e_x*spsi + e_y*cpsi, 0]
    # (only Fx, Fy affect translational accel)
    a_tau_x = -2.0 * (e_x * cpsi + e_y * spsi) / mass
    a_tau_y = -2.0 * (-e_x * spsi + e_y * cpsi) / mass
    # H4 fix: Mz has relative degree > 2 to h through translational dynamics.
    # Mz affects psi_dot (yaw rate), which affects v_ned through rotation,
    # which then affects h through position. The instantaneous sensitivity
    # d(h_ddot)/d(Mz) is zero in continuous time. Setting to zero is the
    # correct conservative choice for the QP constraint.
    a_tau_n = 0.0

    a_hocbf = np.array([a_tau_x, a_tau_y, a_tau_n], dtype=np.float64)

    # Compute the tau-independent part of h_ddot
    # h_ddot_free = -2 * ||v_ned||^2 - 2 * e^T * a_disturbance_ned
    # For the free response, we use tau_des as the nominal acceleration
    v_ned_sq = vx_ned ** 2 + vy_ned ** 2
    accel_body_free = tau_des / mass  # simplified: ignore damping for free response
    accel_ned_free = np.array([
        cpsi * accel_body_free[0] - spsi * accel_body_free[1],
        spsi * accel_body_free[0] + cpsi * accel_body_free[1],
    ])
    h_ddot_free = -2.0 * v_ned_sq - 2.0 * (e_x * accel_ned_free[0] + e_y * accel_ned_free[1])

    # HOCBF: psi = h_ddot + (a1+a2)*h_dot + a1*a2*h >= 0
    psi_hocbf = h_ddot_free + (alpha1 + alpha2) * h_dot + alpha1 * alpha2 * h_val

    # The QP constraint: a_hocbf @ tau + delta >= b_hocbf
    # where b_hocbf = -psi_hocbf_free + a_hocbf @ tau_des (rearranging)
    # Actually: we want psi_hocbf(tau) >= 0
    # psi_hocbf(tau) = psi_hocbf_free + a_hocbf @ (tau - tau_des)
    # So: a_hocbf @ tau >= a_hocbf @ tau_des - psi_hocbf_free
    b_hocbf_nominal = float(a_hocbf @ tau_des - psi_hocbf)

    # Bounded-disturbance robustification.  If the unmodelled translational
    # acceleration satisfies ||a_d|| <= d_bar, the worst-case contribution to
    # h_ddot is lower-bounded by -2 ||e|| d_bar.  Tightening b by that amount
    # gives a conservative robust-HOCBF inequality under the simplified model.
    disturbance_accel_bound = max(0.0, float(getattr(params, "disturbance_accel_bound_mps2", 0.0)))
    robust_disturbance_margin = 2.0 * float(np.sqrt(dist_sq)) * disturbance_accel_bound
    b_hocbf = float(b_hocbf_nominal + robust_disturbance_margin)

    return {
        "a_hocbf": a_hocbf,
        "b_hocbf": b_hocbf,
        "b_hocbf_nominal": b_hocbf_nominal,
        "h_val": float(h_val),
        "h_dot": float(h_dot),
        "psi_hocbf": float(psi_hocbf),
        "robust_disturbance_margin": float(robust_disturbance_margin),
        "disturbance_accel_bound_mps2": float(disturbance_accel_bound),
    }
