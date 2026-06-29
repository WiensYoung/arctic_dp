"""Statistics relative improvement tests."""

import math
import numpy as np
import pytest


class TestRelativeImprovement:
    """相对改进统计测试。"""

    def test_lower_is_better_improvement(self):
        """lower-is-better 指标的改进应为正。"""
        import pandas as pd
        from arctic_quasi_dp.sci1.statistics import paired_comparison
        rows = []
        for seed in range(10):
            rows.append({"scenario_id": "B1", "controller": "pid", "seed": seed,
                         "rms_position_error_m": 2.0})
            rows.append({"scenario_id": "B1", "controller": "full", "seed": seed,
                         "rms_position_error_m": 1.0})
        df = pd.DataFrame(rows)
        result = paired_comparison(df, "rms_position_error_m", "pid", "full", lower_is_better=True)
        # pid=2.0, full=1.0, improvement = (2-1)/2*100 = 50%
        assert result["relative_improvement_pct"] > 0
        assert result["relative_improvement_pct"] == pytest.approx(50.0, abs=1.0)

    def test_ci_fields_exist(self):
        """relative_improvement_ci_lo/hi 应存在。"""
        import pandas as pd
        from arctic_quasi_dp.sci1.statistics import paired_comparison
        rows = []
        for seed in range(10):
            rows.append({"scenario_id": "B1", "controller": "pid", "seed": seed,
                         "rms_position_error_m": 2.0 + seed * 0.1})
            rows.append({"scenario_id": "B1", "controller": "full", "seed": seed,
                         "rms_position_error_m": 1.0 + seed * 0.05})
        df = pd.DataFrame(rows)
        result = paired_comparison(df, "rms_position_error_m", "pid", "full", lower_is_better=True)
        assert "relative_improvement_ci_lo" in result
        assert "relative_improvement_ci_hi" in result
        assert not math.isnan(result["relative_improvement_ci_lo"])

    def test_baseline_zero_no_crash(self):
        """baseline 为 0 时不应崩溃。"""
        import pandas as pd
        from arctic_quasi_dp.sci1.statistics import paired_comparison
        rows = []
        for seed in range(5):
            rows.append({"scenario_id": "B1", "controller": "pid", "seed": seed,
                         "rms_position_error_m": 0.0})
            rows.append({"scenario_id": "B1", "controller": "full", "seed": seed,
                         "rms_position_error_m": 1.0})
        df = pd.DataFrame(rows)
        result = paired_comparison(df, "rms_position_error_m", "pid", "full", lower_is_better=True)
        # Should not crash, improvement should be finite
        assert math.isfinite(result["relative_improvement_pct"])

    def test_method_field_present(self):
        """method 字段应存在。"""
        import pandas as pd
        from arctic_quasi_dp.sci1.statistics import paired_comparison
        rows = []
        for seed in range(5):
            rows.append({"scenario_id": "B1", "controller": "pid", "seed": seed,
                         "rms_position_error_m": 2.0})
            rows.append({"scenario_id": "B1", "controller": "full", "seed": seed,
                         "rms_position_error_m": 1.0})
        df = pd.DataFrame(rows)
        result = paired_comparison(df, "rms_position_error_m", "pid", "full")
        assert result["method"] == "paired_wilcoxon_signed_rank"
        assert result["paired_by"] == "scenario_id + seed"
