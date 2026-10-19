# CPU Parallel Experiment Runtime Upgrade Summary

Date: 2026-06-27
Target server: 32-core AMD EPYC 9654 Linux node

## Implemented

1. Task-level process parallelism
   - Added `--jobs` and `--parallel-backend {serial,process}` to the SCI1 runner.
   - Parallel unit: `scenario_id x controller x seed`.
   - Recommended for a 32-core server: `--jobs 24` to `--jobs 28`.

2. Oversubscription protection
   - Runner sets BLAS/OpenMP environment defaults to one thread per worker:
     - `OMP_NUM_THREADS=1`
     - `MKL_NUM_THREADS=1`
     - `OPENBLAS_NUM_THREADS=1`
     - `NUMEXPR_NUM_THREADS=1`
     - `VECLIB_MAXIMUM_THREADS=1`
   - Users should still export these variables before launching large runs.

3. Resume support
   - Added `--resume`.
   - Each task writes an atomic payload under `raw/tasks/<task_hash>.json` and `raw/tasks/<task_hash>.done`.
   - Re-running with `--resume` skips completed tasks whose task hash matches the current config hash.

4. Trace I/O reduction
   - Added `--no-save-traces` alias.
   - Added `--save-traces-on-failure`.
   - Added `--trace-downsample N`.
   - Paper parallel config disables full traces by default and saves traces only for failure/infeasible cases.

5. Statistics split
   - Added `--skip-statistics` to defer expensive Wilcoxon/bootstrap comparison tables.
   - Added `--statistics-only` to recompute summary/statistics from existing `raw/per_seed_metrics.csv`.

6. New CPU-parallel configs
   - `configs/sci1/sci1_method_fast.yaml`
   - `configs/sci1/sci1_method_paper_parallel.yaml`
   - `configs/sci1/sci1_representative_traces.yaml`

7. Regression tests
   - Added `tests/sci1/test_parallel_runner_cli.py`.
   - Covers CLI merge behavior, task hash stability, trace-save policy, and parallel config semantics.

## Batch scheduling follow-up (2026-10-19)

The 24–28 worker recommendation above applies to the total task-worker budget,
not a separate allocation for every concurrently launched experiment.
`run_all.py --jobs N` now runs experiments sequentially within each phase and
passes N to the active experiment's existing task-level process pool. Thus one
batch invocation has at most N experiment task workers active at once. This does
not coordinate separate batch invocations or count runner/helper processes and
native-library threads. Explicit BLAS/OpenMP environment settings remain honored.

Phase selection, resume flags, per-experiment logs, smoke gating, and required
versus optional failure handling are preserved. Launch errors are recorded as
failures; interrupted waits stop the current experiment (including its process
group on POSIX) before propagating the exception. Ctrl-C exits with status 130.
See `tests/sci1/test_run_all.py` for lightweight scheduling regressions using
fake processes and tiny Python children, without launching simulations.
No server-load measurements or performance improvement estimates are claimed.

## Recommended 32-core run command

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHONPATH=src python -m arctic_quasi_dp.sci1.runner \
  --config configs/sci1/sci1_method_paper_parallel.yaml \
  --out /scratch/arctic_dp/method_paper_parallel \
  --jobs 28 \
  --parallel-backend process \
  --resume \
  --skip-statistics

PYTHONPATH=src python -m arctic_quasi_dp.sci1.runner \
  --config configs/sci1/sci1_method_paper_parallel.yaml \
  --out /scratch/arctic_dp/method_paper_parallel \
  --statistics-only
```

## Validation performed in this environment

- `python -m compileall -q src scripts tests`: passed.
- `tests/sci1/test_parallel_runner_cli.py`: 4 passed.
- `sci1_method_fast.yaml` with `--jobs 2 --parallel-backend process --resume`: completed.
- Resume rerun of `sci1_method_fast.yaml`: completed without rerunning pending tasks.
- Full pytest printed `384 passed, 4 skipped, 50 warnings`, but the wrapper process did not return cleanly in this sandbox after the success summary. Treat the printed result as useful evidence, but re-run full pytest on the deployment server.

## Supported claim

The runner now supports single-node, task-level process parallel execution suitable for 32-core CPU servers.

## Not claimed

- No GPU acceleration was added.
- No distributed multi-node scheduler such as Ray/Dask was added.
- No Numba/Cython/Rust simulator kernel was added.
