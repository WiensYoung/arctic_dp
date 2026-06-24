"""SCI1 集成测试。

验证:
- 场景构建完整性
- 指标计算正确性
- 统计比较功能
- 列名容错
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from arctic_quasi_dp.sci1.scenarios import build_sci1_scenarios, SCI1Scenario
from arctic_quasi_dp.sci1.metrics import (
    summarize_run,
    aggregate_summary,
    run_all_comparisons,
    statistical_comparison,
    _cohens_d,
    _wilcoxon_rank_sum_p,
    _find_column,
    _detect_saturation_columns,
)


# ---------- 场景测试 ----------

class TestScenarios:
    def test_smoke_profile_count(self):
        scenarios = build_sci1_scenarios("smoke")
        assert len(scenarios) >= 9  # A1,A2,B1,B2,B3,B4,C1,D1,E1,F1

    def test_paper_profile_count(self):
        scenarios = build_sci1_scenarios("paper")
        assert len(scenarios) >= 9

    def test_all_scenarios_have_ids(self):
        for s in build_sci1_scenarios("smoke"):
            assert s.scenario_id
            assert s.group

    def test_scenario_groups_covered(self):
        scenarios = build_sci1_scenarios("smoke")
        groups = {s.group for s in scenarios}
        assert "A_precision" in groups
        assert "B_ice_enhancement" in groups
        assert "C_fault_tolerance" in groups
        assert "D_safety_degradation" in groups
        assert "E_realtime" in groups
        assert "F_miz" in groups

    def test_timevarying_scenario(self):
        scenarios = build_sci1_scenarios("smoke")
        b4 = [s for s in scenarios if s.scenario_id == "B4_nonstationary_ice_ramp"]
        assert len(b4) == 1
        s = b4[0]
        assert s.ice_time_varying
        ice_start = s.ice_conditions_at(0.0)
        ice_end = s.ice_conditions_at(s.duration)
        assert ice_start["concentration"] < ice_end["concentration"]
        assert ice_start["drift_speed"] < ice_end["drift_speed"]

    def test_scenario_to_dict(self):
        s = build_sci1_scenarios("smoke")[0]
        d = s.to_dict()
        assert "scenario_id" in d
        assert "duration" in d

    def test_scenario_to_config_kwargs(self):
        s = build_sci1_scenarios("smoke")[0]
        kw = s.to_config_kwargs()
        assert "duration" in kw
        assert "dt" in kw


# ---------- 指标测试 ----------

class TestMetrics:
    def _make_test_df(self, n=100):
        """创建测试用 DataFrame。"""
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "position_error": rng.normal(0.5, 0.1, n),
            "heading_error": rng.normal(0.05, 0.01, n),
            "violation": np.zeros(n),
            "solve_time_ms": rng.uniform(0.1, 1.0, n),
            "risk_cvar": rng.uniform(0.0, 0.3, n),
            "energy": np.cumsum(rng.uniform(0.1, 0.5, n)),
        })

    def test_summarize_run_basic(self):
        df = self._make_test_df()
        result = summarize_run(df, "test", "ctrl", 0, 0.1)
        assert "rms_position_error_m" in result
        assert "failure" in result
        assert result["n_steps"] == 100
        assert np.isfinite(result["rms_position_error_m"])

    def test_summarize_run_failure_threshold(self):
        # 位置误差 > 1.5 * safe_region_radius 应触发 failure
        df = pd.DataFrame({
            "position_error": [20.0] * 10,
            "violation": [0.0] * 10,
        })
        result = summarize_run(df, "test", "ctrl", 0, 0.1, safe_region_radius=10.0)
        assert result["failure"] == 1  # 20 > 15

    def test_summarize_run_no_failure(self):
        df = pd.DataFrame({
            "position_error": [0.5] * 10,
            "violation": [0.0] * 10,
        })
        result = summarize_run(df, "test", "ctrl", 0, 0.1, safe_region_radius=10.0)
        assert result["failure"] == 0

    def test_aggregate_summary(self):
        rows = [
            {"scenario_id": "S1", "controller": "A", "seed": 0, "rms_position_error_m": 1.0, "failure": 0},
            {"scenario_id": "S1", "controller": "A", "seed": 1, "rms_position_error_m": 1.2, "failure": 0},
            {"scenario_id": "S1", "controller": "B", "seed": 0, "rms_position_error_m": 0.5, "failure": 0},
        ]
        df = pd.DataFrame(rows)
        agg = aggregate_summary(df)
        assert len(agg) == 2
        a_row = agg[(agg["scenario_id"] == "S1") & (agg["controller"] == "A")]
        assert len(a_row) == 1
        assert abs(a_row["rms_position_error_m_mean"].iloc[0] - 1.1) < 0.01

    def test_column_name_fallback(self):
        df = pd.DataFrame({"pos_error": [1.0, 2.0], "head_error": [0.1, 0.2]})
        result = summarize_run(df, "test", "ctrl", 0, 0.1)
        # 应该找到 pos_error 列
        assert np.isfinite(result["rms_position_error_m"])


# ---------- 统计检验测试 ----------

class TestStatisticalTests:
    def test_cohens_d_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _cohens_d(a, a) == 0.0

    def test_cohens_d_different(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        d = _cohens_d(a, b)
        assert d < 0  # b 的均值更大

    def test_wilcoxon_same_distribution(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        p = _wilcoxon_rank_sum_p(a, b)
        assert p > 0.01  # 相同分布不应显著

    def test_wilcoxon_different_distribution(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(5, 1, 50)
        p = _wilcoxon_rank_sum_p(a, b)
        assert p < 0.001  # 不同分布应显著

    def test_statistical_comparison(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "controller": ["pid"] * 20 + ["full"] * 20,
            "rms_position_error_m": np.concatenate([rng.normal(2.0, 0.5, 20), rng.normal(1.0, 0.3, 20)]),
        })
        result = statistical_comparison(df, "rms_position_error_m", "pid", "full")
        assert result["significant"]
        assert result["improvement_pct"] > 0  # full 应该更好


# ---------- 列名检测 ----------

class TestColumnDetection:
    def test_find_column_exact(self):
        df = pd.DataFrame({"position_error": [1.0]})
        assert _find_column(df, ["position_error"]) == "position_error"

    def test_find_column_fallback(self):
        df = pd.DataFrame({"pos_error": [1.0]})
        assert _find_column(df, ["position_error", "pos_error"]) == "pos_error"

    def test_find_column_missing(self):
        df = pd.DataFrame({"other": [1.0]})
        assert _find_column(df, ["position_error", "pos_error"]) is None

    def test_detect_saturation_columns(self):
        df = pd.DataFrame({"thruster_1_actual": [0.5], "thruster_2_actual": [0.6]})
        cols = _detect_saturation_columns(df)
        assert len(cols) == 2
