"""Actuator chattering metrics tests."""

import math
import numpy as np
import pytest


class TestChatteringMetrics:
    """执行器 chattering 指标测试。"""

    def test_constant_thrust_zero_variation(self):
        """常值推力时 total variation 应为 0。"""
        import pandas as pd
        from arctic_quasi_dp.sci1.metrics import summarize_run
        df = pd.DataFrame({
            "time": [0.1, 0.2, 0.3],
            "position_error": [1.0, 1.0, 1.0],
            "heading_error": [0.1, 0.1, 0.1],
            "violation": [0.0, 0.0, 0.0],
            "energy": [0.0, 1.0, 2.0],
            "thruster_0_actual_thrust": [100.0, 100.0, 100.0],
        })
        result = summarize_run(df, "test", "pid", 0, 0.1)
        assert result["thrust_total_variation"] == 0.0

    def test_azimuth_wraparound_correct(self):
        """角度从 179° 到 -179° 时 variation 应约 2° 而非 358°。"""
        from arctic_quasi_dp.utils.math_utils import shortest_angle_diff_deg
        d = abs(shortest_angle_diff_deg(-179.0, 179.0))
        assert abs(d - 2.0) < 1e-6

    def test_missing_angle_fields_nan(self):
        """缺少角度字段时 chattering 指标应为 NaN。"""
        import pandas as pd
        from arctic_quasi_dp.sci1.metrics import summarize_run
        df = pd.DataFrame({
            "time": [0.1, 0.2],
            "position_error": [1.0, 1.0],
            "heading_error": [0.1, 0.1],
            "violation": [0.0, 0.0],
            "energy": [0.0, 1.0],
        })
        result = summarize_run(df, "test", "pid", 0, 0.1)
        # No angle columns → azimuth metric should be unavailable rather than a silent zero.
        assert math.isnan(result["azimuth_total_variation"])
        assert result["azimuth_metric_available"] == 0.0

    def test_power_cap_active_rate(self):
        """power_cap_active_rate 应正确计算。"""
        import pandas as pd
        from arctic_quasi_dp.sci1.metrics import summarize_run
        df = pd.DataFrame({
            "time": [0.1, 0.2, 0.3, 0.4],
            "position_error": [1.0, 1.0, 1.0, 1.0],
            "heading_error": [0.1, 0.1, 0.1, 0.1],
            "violation": [0.0, 0.0, 0.0, 0.0],
            "energy": [0.0, 1.0, 2.0, 3.0],
            "power_cap_active": [0.0, 1.0, 1.0, 0.0],
        })
        result = summarize_run(df, "test", "pid", 0, 0.1)
        assert abs(result["power_cap_active_rate"] - 0.5) < 1e-6
