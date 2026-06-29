# Soft-HOCBF-QP method scope and limitations

This document is part of the artifact and is intentionally conservative. It
summarizes what the current code implements and what it does **not** prove.

## Implemented method path

The method controllers (`fixed_soft_hocbf`, `cvar_soft_hocbf`) wrap an existing
nominal controller and apply a small Soft-HOCBF-QP in generalized-force space:

```text
nominal controller -> tau_des -> Soft-HOCBF-QP -> tau_safe -> thruster allocator
```

The QP variable is:

```text
z = [tau_x, tau_y, tau_n, delta]
```

The objective is:

```text
0.5 * ||tau - tau_des||_W^2 + 0.5 * p_delta * delta^2
```

The implemented safety set is the proxy-scale watch-circle set:

```text
h(x) = R_safe^2 - ||p - p_ref||^2
```

The current implementation uses a relative-degree-2 proxy HOCBF constraint of
the form:

```text
a_hocbf(x)^T tau + delta >= b_hocbf(x, alpha1, alpha2)
delta >= 0
```

`delta` is a soft relaxation variable. It is reported in every trace as
`safety_filter_slack` and must be interpreted as a safety-pressure indicator,
not as proof that the physical vessel remained formally invariant under all
unmodelled disturbances.

## CVaR / tail-risk modulation

`cvar_soft_hocbf` uses a CVaR-style tail-risk proxy to scale HOCBF gains:

```text
risk_scale = 1 + risk_gain * clip(risk_level, 0, 1)
alpha1 = alpha1_base * risk_scale
alpha2 = alpha2_base * risk_scale
```

This is a tail-risk-adaptive safety filter. It is **not** a formal stochastic
CVaR-constrained controller unless a distributional forecast ensemble and a
mathematically defined CVaR risk constraint are added.

## Actuator feasible set

The current actuator-aware mode uses a proxy conservative inner polygon. Radial
vertices are shrunk and checked with the project `ThrusterAllocator`; the
resulting polygon is converted to fixed-shape linear constraints for the QP.
This improves consistency between the safety filter and downstream allocation,
but it remains a proxy-scale approximation. It must not be described as a
full-scale thruster-allocation certificate.

## Claims supported by the current code

The current code can support claims such as:

- a proxy-scale Soft-HOCBF-QP safety-filter method layer is implemented;
- the method reports solver status, slack, correction norm, risk scaling, and
  actuator-feasible-set type in trace files;
- a packaged mock data-driven replay file exercises the H-group replay path;
- a real-replay configuration can use an externally downloaded/bundled real
  Copernicus subset when present and records `fallback_used=false`, path, and
  checksum in `actual_data_usage.json`;
- ADRC, robust-MPC-style, and tube-MPC-style proxy baselines are runnable.

## Claims not supported by the current code

Do **not** claim:

- full-scale DP3 validation;
- formal safety proof for real Arctic ship operations;
- true stochastic CVaR-constrained control;
- complete invariant-tube MPC with a proven robust positive invariant set;
- real Copernicus validation from the packaged mock fixture.

## Recommended future work before strong top-journal claims

1. Run paper results with OSQP installed and verify `safety_filter_solver_backend = osqp`.
2. Replace the packaged mock NetCDF with a real downloaded Copernicus/ERA5 subset
   and lock its checksum.
3. Derive a disturbance-bounded HOCBF using vessel mass/inertia/damping from the
   selected vessel configuration.
4. Add a true forecast ensemble and formal CVaR definition.
5. Implement a certified Tube-MPC/Robust-MPC baseline if the paper claims a
   strong control-theory comparison.
