# Method Theory Sketch: Soft-HOCBF-QP for Arctic Quasi-DP

## 1. Simplified Vessel Dynamics

The benchmark uses a 3-DOF (surge, sway, yaw) rigid-body model:

```
M * nu_dot + D(nu) * nu = tau_control + tau_ice + tau_wind
```

where:
- `M = diag(m, m, Izz)` is the inertia matrix (added mass included in linear damping)
- `D(nu) = diag(Xu + Xu_abs*|u|, Yv + Yv_abs*|v|, Nr + Nr_abs*|r|)` is linear + quadratic damping
- `tau_control` is the thruster-generated force/moment
- `tau_ice` is the ice interaction force (Lindqvist 1989 proxy)
- `tau_wind` is the wind forcing (quadratic drag model)

**Limitation**: This is a proxy-scale model. The ice crushing strength is set to 0.0003 MPa
(~1 kN forces) rather than the ISO 19906 range of 1-5 MPa. Results demonstrate algorithmic
behavior, not full-scale Arctic DP performance.

## 2. Safety Set Definition

The position safety set is a disk of radius R centered at the target:

```
S = { (x, y) : h(x, y) >= 0 }
h(x, y) = R^2 - ||p - p_ref||^2
```

where `p = (x, y)` is the vessel position and `p_ref` is the target.

## 3. Relative-Degree-2 Soft-HOCBF Condition

For the DP system, `h(x)` has relative degree 2 with respect to the control input `tau`:

```
h_dot = -2 * (p - p_ref)^T * v_ned
h_ddot = -2 * ||v_ned||^2 - 2 * (p - p_ref)^T * a_ned
```

where `v_ned` is the NED-frame velocity and `a_ned ≈ R_body2ned @ (tau / M)` is the
NED-frame acceleration (simplified, ignoring Coriolis and damping in the HOCBF derivative).

The CBF condition of degree 2 is:

```
h_ddot + (alpha1 + alpha2) * h_dot + alpha1 * alpha2 * h >= 0
```

This is affine in `tau`, allowing it to be cast as a linear constraint in a QP.

## 4. QP Formulation with Slack

The safety filter solves at each timestep:

```
minimize    0.5 * ||tau - tau_des||^2 + 0.5 * w_delta * delta^2
subject to  a_hocbf^T * tau + delta >= b_hocbf    (HOCBF constraint)
            delta >= 0                              (slack non-negativity)
            tau in feasible set                     (actuator limits)
```

where:
- `tau_des` is the nominal controller output
- `delta` is the slack variable allowing soft constraint satisfaction
- `w_delta` penalizes slack (default 1e4)
- `a_hocbf`, `b_hocbf` are the linearized CBF constraint coefficients

The QP is solved by OSQP (primary) or SciPy SLSQP (fallback).

## 5. Soft Feasibility Proposition

**Proposition (informal)**: Under the proxy-scale dynamics and the soft-HOCBF formulation:

1. If `delta = 0`, the CBF constraint is satisfied and the safety set is forward invariant
   (under the model assumptions).
2. If `delta > 0`, the constraint is violated by `delta / (w_delta)` amount, and the
   vessel may leave the safety set. The slack provides a graceful degradation mechanism.
3. The penalty `w_delta` controls the trade-off between tracking performance and safety:
   higher `w_delta` enforces safety more aggressively.

**Limitation**: This is NOT a formal forward invariance guarantee because:
- The dynamics model is simplified (no added mass, no Coriolis, linear damping)
- The HOCBF derivative ignores damping terms in `a_ned`
- The ice disturbance is bounded only by the proxy model, not by a proven bound
- The relative-degree-2 condition assumes the control input directly affects acceleration

## 6. Interpretation of Slack

- `slack = 0`: HOCBF constraint satisfied, safety maintained (within model assumptions)
- `slack > 0`: Minimum violation needed to maintain feasibility; indicates the nominal
  controller would violate safety without correction
- `slack_active_rate > 0`: Fraction of timesteps where safety filter had to correct
  the nominal control

The trace separates the geometric set value from the QP inequality value:

- `safety_set_h` tracks `h(x)` directly: positive means inside the watch circle,
  negative means outside.
- `hocbf_constraint_margin` / `safety_filter_hocbf_margin` tracks the actual
  QP inequality margin `a_hocbf @ tau_safe + slack - b_hocbf`. This is the
  field used for HOCBF constraint-satisfaction statistics.

## 7. Dwell-Time Switching / Practical ISS Sketch

The mode supervisor uses hysteresis-based switching with minimum dwell time:

```
PRECISION -> ICE_AWARE:     risk >= ice_enter (0.28)
ICE_AWARE -> QUASI_DP:      risk >= high_risk_enter (0.58) or pos_err >= 8.5m
ICE_AWARE -> ESCAPE:        risk >= extreme_risk_enter (0.82)
ESCAPE -> ICE_AWARE:        risk < ice_exit (0.20)
```

Minimum dwell time: 2.0 seconds (prevents chattering).

**Practical ISS argument (informal)**: Each mode is designed to be input-to-state stable
with respect to ice disturbance as input. The hysteresis prevents Zeno behavior, and the
dwell time ensures each mode has sufficient time to converge. The risk metric decreases
when the vessel is well-positioned and ice conditions are mild, enabling mode transitions
back to higher-precision modes.

**Limitation**: No formal proof of ISS for the switched system. The mode transitions are
heuristic, and there is no guarantee that the switched system preserves stability across
all possible switching sequences.

## 8. Explicit Limitations

1. **Proxy-scale only**: Ice forces are ~1 kN, not ~100 kN. Results demonstrate algorithmic
   behavior, not full-scale Arctic DP performance.
2. **No formal safety proof**: The HOCBF condition is derived under simplified dynamics.
   Forward invariance is not formally guaranteed.
3. **CVaR is a proxy**: The tail-risk estimate uses Monte Carlo sampling of ice force
   perturbations, not a true stochastic CVaR-constrained optimization.
4. **Observer is EMA-based**: The ice condition observer is a first-order low-pass filter
   with noise injection, not a proper disturbance observer (EKF, SMO, etc.).
5. **No wave disturbance**: The model does not include wave-induced forces.
6. **Single vessel**: All experiments use a 500-ton proxy vessel (or XueLong2-like
   experimental config). Multi-vessel validation is not included.
7. **Data replay scope**: `H1_mock_copernicus_fixture` uses a packaged mock
   NetCDF fixture for offline artifact checks. `H1_real_copernicus_era5_replay`
   requires a real subset file and records the actual path/checksum in
   `actual_data_usage.json`; it is still a replay input validation, not a
   full-scale DP validation.


## 9. Theory-Oriented Diagnostics

Additional bounded-disturbance, soft-certificate, and dwell-time diagnostics are documented in `METHOD_THEORETICAL_ASSUMPTIONS.md`. These diagnostics are designed for paper auditing and do not imply full-scale formal safety validation.
