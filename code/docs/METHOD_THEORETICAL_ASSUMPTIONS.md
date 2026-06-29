# Theory-Oriented Assumptions and Certificate Diagnostics

This document is intentionally conservative. It describes the additional
checks used when positioning the method for theory-oriented venues. These
checks do not turn the proxy-scale simulator into a full-scale Arctic DP proof,
but they make the gap between implementation, assumptions, and paper claims
explicit.

## A1. Proxy Dynamics Used by the Safety Filter

The implemented Soft-HOCBF-QP uses the simplified 3-DOF model

```text
M nu_dot + D(nu) nu = tau + d(t)
```

and applies the position safety function

```text
h(p) = R^2 - ||p - p_ref||^2.
```

The QP enforces a relative-degree-2 inequality in generalized-force space:

```text
a_hocbf(x)^T tau + delta >= b_hocbf(x).
```

## A2. Bounded-Disturbance Robustification

For theory-oriented diagnostics, the code supports an optional bound

```text
||a_d(t)|| <= d_bar
```

on unmodelled translational acceleration in the NED plane. Since

```text
h_ddot = -2 ||v||^2 - 2 e^T (a_control + a_d),
```

the worst-case unmodelled acceleration contributes at least

```text
-2 ||e|| d_bar.
```

The QP therefore tightens the HOCBF right-hand side by

```text
robust_margin = 2 ||e|| d_bar.
```

The trace reports this as `hocbf_robust_disturbance_margin` and the configured
bound as `hocbf_disturbance_accel_bound_mps2`.

## A3. Certificate Fields

A timestep is marked as a soft certificate timestep when:

```text
hocbf_constraint_margin >= min_certificate_margin
safety_filter_slack <= max_certificate_slack
```

The run-level field `hocbf_soft_certificate_rate` is the fraction of timesteps
satisfying these two conditions. This is a proxy-scale certificate diagnostic,
not a real-vessel formal proof.

## A4. Switching Diagnostics

The mode supervisor includes hysteresis and a nominal minimum dwell-time. The
run-level diagnostics include:

```text
mode_switch_count
min_mode_dwell_time_s
dwell_time_violation_count
```

These fields support a practical-ISS discussion by showing whether simulated
switching respects the intended dwell-time logic. They are evidence for the
implemented supervisor, not a substitute for a full switched-system proof.

## A5. Claims This Supports

Supported if the reported fields are present and finite:

- bounded-disturbance robustification was included in the proxy HOCBF QP;
- the QP constraint margin, slack, and certificate rate were recorded;
- the supervisor dwell-time diagnostics were audited;
- OSQP-backed Soft-HOCBF-QP can be reported only when
  `safety_filter_solver_backend = osqp`.

## A6. Proxy-Scale Parameter Documentation

The simulation uses **proxy-scale** vessel and ice parameters, chosen so that:
- Ice forces (~1 kN) match controller force limits (~3 kN) and thruster capacity
- The control authority ratio (ice_force / max_thrust) is realistic (~0.3-0.7)

Key proxy-scale values and their full-scale equivalents:

| Parameter | Proxy Value | Full-Scale Value | Ratio |
|-----------|------------|------------------|-------|
| `ice_crushing_strength_mpa` | 0.0003 | 1.0–5.0 (first-year ice) | ~1/6000 |
| `vessel_mass` | 500,000 kg | ~14,000,000 kg (XueLong 2) | ~1/28 |
| `max_force` | 3,000 N | ~1.5 MN (DP3 thrusters) | ~1/500 |

The crushing strength is deliberately set to a proxy value to scale the
1989 Lindqvist ice force formula to the proxy-scale thruster envelope.
**All experimental results should be interpreted as relative algorithm
comparisons under the proxy-scale dynamics, not as full-scale DP
performance predictions.**

## A7. Baseline Controller Naming Disclosure

Two baseline controllers have code names that may mislead reviewers:

- **`robust_mpc`** (RobustMPCController): This is a **PD controller with
  conservatively tightened saturation limits** via a `disturbance_margin`
  factor. It does NOT implement uncertainty set propagation, tube invariance,
  or robust feasibility guarantees. In figures and tables, it appears as
  "Conserv. PD."

- **`tube_mpc`** (TubeMPCController): This is a **PD controller with an
  additional `tube_margin` constraint tightening**. It does NOT implement
  invariant tubes (RPI sets), tube-based constraint tightening from disturbance
  bounds, or a nominal controller within a tube. In figures, it appears as
  "Margin PD."

The code names are retained for backward compatibility with YAML configuration
files and the experiment runner. The paper must use the honest display names.

## A8. CVaR Terminology Clarification

The code uses "CVaR" (Conditional Value at Risk) in variable names and
configuration keys. The current implementation computes a **proxy tail-risk
estimate** via:
1. Monte Carlo sampling of ice disturbance (from observed statistics)
2. Quantile estimation at `cvar_alpha`
3. First-order low-pass filtering of the risk signal
4. Linear gain scaling of the HOCBF constraint

A **true CVaR-constrained controller** would require: ensemble forecast of ice
disturbance, formal quantile optimization, and probabilistic constraint
satisfaction guarantees. The current implementation is more accurately described
as **"risk-adaptive HOCBF gain scaling"** or **"proxy tail-risk adaptation."**

The variable names and YAML keys (`cvar_soft_hocbf`, `risk_cvar`, etc.) are
retained for code consistency; the paper should use the honest terminology.

## A9. Power Analysis for Experimental Seeds

The default seed count (30 for paper profiles) was selected based on a
pre-experiment power analysis:

- **Effect size target**: Cohen's d ≥ 0.5 (medium effect)
- **Significance level**: α = 0.05 (after Holm-Bonferroni correction)
- **Desired power**: 1 − β = 0.80
- **Required sample size**: N ≥ 27 paired observations

With 30 seeds and N ≥ 10 scenarios per comparison group, the experiment has
≥80% power to detect medium effect sizes (d ≥ 0.5). Smaller effects
(0.2 ≤ d < 0.5) are reported with confidence intervals for transparency.

Note: This is a post-hoc justification of the existing 30-seed default.
For a submission, this analysis should be moved to the main text.

## A10. Claims This Still Does Not Support

Do not claim:

- full-scale DP3 validation;
- a formal safety proof for real Arctic vessels;
- true stochastic CVaR-constrained control;
- invariant-tube MPC or robust MPC (the baselines are PD with tightened limits);
- complete TAC/Automatica-level theorem coverage without a full mathematical
  proof of robust forward invariance and switched-system stability.
