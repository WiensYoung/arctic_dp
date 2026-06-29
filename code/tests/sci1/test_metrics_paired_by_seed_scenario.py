"""Test that statistical comparison is paired by scenario_id + seed."""

import numpy as np
import pandas as pd
import pytest

from arctic_quasi_dp.sci1.metrics import statistical_comparison


class TestPairedComparison:
    """Stats must be paired by scenario_id + seed."""

    def test_paired_samples_used(self):
        """Comparison should use paired (scenario_id, seed) matching."""
        data = []
        for scenario in ["S1", "S2"]:
            for seed in range(5):
                data.append({"scenario_id": scenario, "controller": "pid", "seed": seed,
                             "rms_position_error_m": 10.0 + np.random.randn()})
                data.append({"scenario_id": scenario, "controller": "full", "seed": seed,
                             "rms_position_error_m": 5.0 + np.random.randn()})
        run_df = pd.DataFrame(data)
        result = statistical_comparison(run_df, "rms_position_error_m", "pid", "full")
        assert result["paired_samples"] == 10  # 2 scenarios * 5 seeds

    def test_effect_size_direction_in_output(self):
        """Output should include effect_size_direction field."""
        data = []
        for seed in range(5):
            data.append({"scenario_id": "S1", "controller": "pid", "seed": seed,
                         "rms_position_error_m": 10.0 + seed * 0.5})
            data.append({"scenario_id": "S1", "controller": "full", "seed": seed,
                         "rms_position_error_m": 5.0 + seed * 0.3})
        run_df = pd.DataFrame(data)
        result = statistical_comparison(run_df, "rms_position_error_m", "pid", "full")
        assert "effect_size_direction" in result
        assert result["effect_size_direction"] == "positive_means_proposed_better"

    def test_improvement_pct_correct_sign(self):
        """Improvement should be positive when proposed is better."""
        data = []
        for seed in range(10):
            data.append({"scenario_id": "S1", "controller": "pid", "seed": seed,
                         "rms_position_error_m": 10.0})
            data.append({"scenario_id": "S1", "controller": "full", "seed": seed,
                         "rms_position_error_m": 5.0})
        run_df = pd.DataFrame(data)
        result = statistical_comparison(run_df, "rms_position_error_m", "pid", "full")
        assert result["improvement_pct"] > 0, "Improvement should be positive"
