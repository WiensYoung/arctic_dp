# P0 Submission-Readiness Fix Summary

Date: 2026-06-28

This patch focuses on hard submission-readiness issues found in the full-code audit.

## Completed fixes

1. **Formal dependencies**
   - Added `xarray`, `netCDF4`, and `h5netcdf` to formal project dependencies.
   - Kept `osqp` as a formal dependency for method/paper runs.

2. **Cached OSQP path**
   - Reworked `SoftHOCBFSafetyFilter` to keep an OSQP solver instance across timesteps when the sparse pattern is unchanged.
   - Added trace fields for solver setup/update counts and factorization reuse evidence.

3. **HOCBF diagnostics semantics**
   - Split geometric safety-set value from the QP inequality margin:
     - `safety_set_h` = `R^2 - ||p-p_ref||^2`
     - `hocbf_constraint_margin` / `safety_filter_hocbf_margin` = `a_hocbf @ tau_safe + slack - b_hocbf`
   - Added aggregate diagnostics in `summary/hocbf_diagnostics.csv`.

4. **Data replay semantics**
   - Added explicit `H1_mock_copernicus_fixture` and `H1_real_copernicus_era5_replay` scenario separation.
   - Added real replay config that uses the real subset path and fails fast when data is missing.
   - Added `actual_data_usage.json` and `summary/actual_data_usage.csv` to record actual file path, SHA256, source type, and fallback status.

5. **Scale comparison evidence**
   - Made `sci1_scale_comparison.yaml` lightweight and runnable.
   - Runner now writes `summary/scale_analysis.csv` for every run.

6. **Method statistics**
   - Added explicit method comparisons for `no_safety_filter`, `fixed_soft_hocbf`, `cvar_soft_hocbf`, and `ice_aware`.
   - Missing-controller comparisons now fail fast instead of silently creating all-NaN tables.

7. **LESO-ADRC frame consistency**
   - Fixed LESO initialization and update so x/y observer states and control inputs are both in NED frame.

8. **Regression tests**
   - Added `tests/sci1/test_p0_submission_fixes.py`.

## Validation performed

- `python -m compileall -q src scripts tests` — OK
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` — `376 passed, 4 skipped`
- `sci1_smoke.yaml` — OK
- `sci1_artifact_check.yaml` — OK
- `sci1_method_smoke.yaml` — OK, OSQP backend observed
- `sci1_scale_comparison.yaml` — OK, `scale_analysis.csv` generated
- `sci1_real_replay_h1.yaml` — OK, real subset recorded with `fallback_used=false`

## Remaining limitations

- This is still a proxy-scale benchmark and method artifact, not full-scale DP3 validation.
- The HOCBF proof remains a theory sketch under simplified dynamics.
- Tube/robust MPC baselines remain lightweight/proxy unless upgraded to certified robust MPC or invariant Tube-MPC.
- Real replay uses the provided subset in the archive; broader real historical validation still requires additional external data.
