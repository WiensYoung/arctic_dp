from pathlib import Path

import yaml

from arctic_quasi_dp.sci1.runner import (
    ExperimentTask,
    _trace_should_be_saved,
    merge_config,
    parse_args,
)


def test_parallel_cli_options_merge_into_config(tmp_path):
    args = parse_args([
        "--config", "configs/sci1/sci1_method_smoke.yaml",
        "--out", str(tmp_path),
        "--jobs", "28",
        "--parallel-backend", "process",
        "--resume",
        "--skip-statistics",
        "--no-save-traces",
        "--trace-downsample", "5",
        "--save-traces-on-failure",
    ])
    cfg = merge_config({"profile": "method_smoke", "seeds": 1}, args)
    assert cfg["runtime"]["jobs"] == 28
    assert cfg["runtime"]["parallel_backend"] == "process"
    assert cfg["runtime"]["resume"] is True
    assert cfg["runtime"]["skip_statistics"] is True
    assert cfg["output"]["save_traces"] is False
    assert cfg["output"]["trace_downsample"] == 5
    assert cfg["output"]["save_traces_on_failure"] is True


def test_experiment_task_hash_is_stable_and_sensitive():
    base = ExperimentTask(
        scenario={"scenario_id": "B2_ice_concentration_jump"},
        controller="cvar_soft_hocbf",
        seed=0,
        profile="method_smoke",
        config_hash="abc",
    )
    same = ExperimentTask(dict(base.scenario), base.controller, base.seed, base.profile, base.config_hash)
    changed = ExperimentTask(dict(base.scenario), base.controller, 1, base.profile, base.config_hash)
    assert base.task_hash == same.task_hash
    assert base.task_hash != changed.task_hash


def test_trace_save_policy_for_failure_only():
    assert _trace_should_be_saved(True, False, {}) is True
    assert _trace_should_be_saved(False, False, {"failure": 1}) is False
    assert _trace_should_be_saved(False, True, {"failure": 1}) is True
    assert _trace_should_be_saved(False, True, {"safety_filter_infeasible_rate": 0.1}) is True
    assert _trace_should_be_saved(False, True, {"failure": 0, "safety_filter_infeasible_rate": 0.0}) is False


def test_parallel_configs_are_summary_or_trace_profiles():
    paper_cfg = yaml.safe_load(Path("configs/sci1/sci1_method_paper_parallel.yaml").read_text())
    assert paper_cfg["runtime"]["jobs"] == 28
    assert paper_cfg["runtime"]["parallel_backend"] == "process"
    assert paper_cfg["output"]["save_traces"] is False
    assert paper_cfg["output"]["save_traces_on_failure"] is True

    trace_cfg = yaml.safe_load(Path("configs/sci1/sci1_representative_traces.yaml").read_text())
    assert trace_cfg["output"]["save_traces"] is True
    assert trace_cfg["output"]["trace_downsample"] == 1
