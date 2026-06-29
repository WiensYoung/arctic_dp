# Theory-Top-Journal-Oriented Enhancement Summary

Date: 2026-06-28

This revision builds on the P0 submission-readiness package and adds theory-oriented diagnostics. It does not claim Automatica/TAC readiness by itself; it makes the implementation more auditable for such a target.

## Implemented

1. Bounded-disturbance Soft-HOCBF tightening
   - `HOCBFParams.disturbance_accel_bound_mps2` added.
   - HOCBF RHS is tightened by `2 * ||p-p_ref|| * d_bar`.
   - Trace reports `hocbf_robust_disturbance_margin` and `hocbf_disturbance_accel_bound_mps2`.

2. Soft certificate diagnostics
   - Safety filter now reports:
     - `hocbf_nominal_constraint_margin`
     - `hocbf_robust_disturbance_margin`
     - `hocbf_disturbance_accel_bound_mps2`
     - `hocbf_a_norm`
     - `hocbf_b`
     - `hocbf_b_nominal`
     - `hocbf_soft_certificate`
   - Summary reports:
     - `hocbf_soft_certificate_rate`
     - `hocbf_robust_disturbance_margin_p95`
     - `hocbf_disturbance_accel_bound_mps2`
     - `hocbf_nominal_constraint_margin_min`

3. Switching/dwell-time diagnostics
   - Summary reports:
     - `mode_switch_count`
     - `min_mode_dwell_time_s`
     - `dwell_time_violation_count`

4. OSQP convergence robustness
   - OSQP `max_iter` increased from 200 to 2000 to avoid first-step false failures under tighter polygon/HOCBF constraints.
   - Focused C4 scale-comparison test now reports 100% QP success for method controllers.

5. Theory documentation
   - Added `docs/METHOD_THEORETICAL_ASSUMPTIONS.md`.
   - Updated `docs/METHOD_THEORY_SKETCH.md` with a pointer to theory-oriented diagnostics.

6. Tests
   - Added `tests/sci1/test_theory_top_journal_diagnostics.py`.

## Validation

- `python -m compileall -q src scripts tests`: OK
- `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`: 380 passed, 4 skipped, 50 warnings
- `sci1_method_smoke.yaml`: OK, OSQP backend observed for safety-filter controllers
- `sci1_scale_comparison.yaml`: OK, no solver backend `error`, min QP success rate 1.0 for safety-filter runs
- `sci1_artifact_check.yaml`: OK
- `sci1_real_replay_h1.yaml`: OK, `fallback_used=false`, `mock_fixture_used=false`

## Still Not Supported

Do not claim:

- full-scale DP3 validation;
- a complete Automatica/TAC-level proof package;
- formal safety proof for real Arctic vessel operations;
- true stochastic CVaR-constrained control;
- complete invariant Tube-MPC implementation.

This revision supports a stronger paper appendix/technical report by adding bounded-disturbance margins, QP certificate diagnostics, and dwell-time evidence, but a full Automatica/TAC submission still requires a rigorous theorem/proof chain and stronger robust-control baselines.
