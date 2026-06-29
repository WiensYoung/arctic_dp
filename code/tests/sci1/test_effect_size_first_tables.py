"""Effect-size-first statistical table tests."""

import math
import numpy as np
import pandas as pd
import pytest


class TestEffectSizeFirstTables:
    """Verify statistical comparisons CSV has required fields."""

    def test_statistical_comparisons_has_required_columns(self):
        """statistical_comparisons.csv must have effect-size-first columns."""
        from arctic_quasi_dp.sci1.statistics import paired_comparison
        from arctic_quasi_dp.sci1.controllers import compute_total_risk

        # Create synthetic paired data
        np.random.seed(42)
        base_vals = np.random.normal(5.0, 1.0, 20)
        prop_vals = base_vals - np.random.normal(0.5, 0.3, 20)  # proposed is better

        result = paired_comparison(
            pd.DataFrame({
                "controller": ["baseline"] * 20 + ["proposed"] * 20,
                "scenario_id": ["s1"] * 10 + ["s2"] * 10 + ["s1"] * 10 + ["s2"] * 10,
                "seed": list(range(10)) * 4,
                "pos_error": np.concatenate([base_vals, prop_vals]),
            }),
            metric="pos_error",
            baseline="baseline",
            proposed="proposed",
            lower_is_better=True,
        )

        required_keys = [
            "paired_samples", "baseline_mean", "proposed_mean", "diff_mean",
            "diff_ci_lo", "diff_ci_hi", "relative_improvement_pct",
            "cohens_dz", "p_value", "method",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_effect_size_direction_correct(self):
        """Positive cohens_dz should mean proposed is better (for lower-is-better)."""
        from arctic_quasi_dp.sci1.statistics import paired_comparison

        np.random.seed(42)
        # Proposed has lower values (better for position error), with varying differences
        base_vals = np.array([10.0, 12.0, 11.0, 14.0, 13.0, 15.0, 11.5, 13.5])
        prop_vals = np.array([4.0, 7.0, 5.5, 9.0, 6.0, 10.0, 5.0, 8.0])

        n = len(base_vals)
        result = paired_comparison(
            pd.DataFrame({
                "controller": ["base"] * n + ["prop"] * n,
                "scenario_id": ["s1"] * n + ["s1"] * n,
                "seed": list(range(n)) * 2,
                "metric": np.concatenate([base_vals, prop_vals]),
            }),
            metric="metric",
            baseline="base",
            proposed="prop",
            lower_is_better=True,
        )

        assert result["cohens_dz"] > 0, "Positive d_z should mean proposed is better"

    def test_ci_not_nan_with_sufficient_data(self):
        """CI should not be NaN with >= 5 paired samples."""
        from arctic_quasi_dp.sci1.statistics import paired_comparison

        np.random.seed(42)
        base_vals = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        prop_vals = np.array([9.0, 10.0, 11.0, 12.0, 13.0])

        result = paired_comparison(
            pd.DataFrame({
                "controller": ["base"] * 5 + ["prop"] * 5,
                "scenario_id": ["s1"] * 5 + ["s1"] * 5,
                "seed": list(range(5)) * 2,
                "metric": np.concatenate([base_vals, prop_vals]),
            }),
            metric="metric",
            baseline="base",
            proposed="prop",
            lower_is_better=True,
        )

        assert not math.isnan(result["diff_ci_lo"]), "CI lo should not be NaN"
        assert not math.isnan(result["diff_ci_hi"]), "CI hi should not be NaN"
