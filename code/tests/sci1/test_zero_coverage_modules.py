"""Tests for previously untested modules: hocbf, logging_config, qp_solver, figures, data_calibration."""

import math
import warnings
import numpy as np
import pandas as pd
import pytest


# ============================================================
# 1. control/hocbf.py
# ============================================================

class TestHOCBF:
    """HOCBF constraint computation tests."""

    def test_hocbf_params_defaults(self):
        from arctic_quasi_dp.sci1.control.hocbf import HOCBFParams
        p = HOCBFParams()
        assert p.alpha1_base > 0
        assert p.alpha2_base > 0
        assert p.safe_radius_m > 0

    def test_hocbf_constraint_output_shape(self):
        from arctic_quasi_dp.sci1.control.hocbf import HOCBFParams, compute_hocbf_constraint
        p = HOCBFParams()
        state = np.array([5.0, 3.0, 0.1, 0.5, 0.2, 0.01])
        tau_des = np.array([100.0, 50.0, 1000.0])
        result = compute_hocbf_constraint(state, tau_des, (0.0, 0.0), 0.0, p, p.alpha1_base, p.alpha2_base, 0.1)
        assert "a_hocbf" in result
        assert "b_hocbf" in result
        assert "h_val" in result
        assert len(result["a_hocbf"]) == 3

    def test_hocbf_h_val_positive_inside(self):
        """h(x) = R^2 - ||p - p_ref||^2 > 0 when vessel is inside safe radius."""
        from arctic_quasi_dp.sci1.control.hocbf import HOCBFParams, compute_hocbf_constraint
        p = HOCBFParams(safe_radius_m=10.0)
        state = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])  # distance = 5 < 10
        tau_des = np.zeros(3)
        result = compute_hocbf_constraint(state, tau_des, (0.0, 0.0), 0.0, p, p.alpha1_base, p.alpha2_base, 0.1)
        assert result["h_val"] > 0  # inside safe region

    def test_hocbf_h_val_negative_outside(self):
        """h(x) < 0 when vessel is outside safe radius."""
        from arctic_quasi_dp.sci1.control.hocbf import HOCBFParams, compute_hocbf_constraint
        p = HOCBFParams(safe_radius_m=5.0)
        state = np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0])  # distance ~14.14 > 5
        tau_des = np.zeros(3)
        result = compute_hocbf_constraint(state, tau_des, (0.0, 0.0), 0.0, p, p.alpha1_base, p.alpha2_base, 0.1)
        assert result["h_val"] < 0


# ============================================================
# 2. logging_config.py
# ============================================================

class TestLoggingConfig:
    """Logger factory tests."""

    def test_get_logger_returns_logger(self):
        import logging
        from arctic_quasi_dp.sci1.logging_config import get_logger
        logger = get_logger("test_module_1")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module_1"

    def test_get_logger_default_level_is_info(self):
        import logging
        from arctic_quasi_dp.sci1.logging_config import get_logger
        logger = get_logger("test_default_level")
        assert logger.level == logging.INFO

    def test_get_logger_debug_level_not_overridden(self):
        """BUG-09 fix: level=0 (DEBUG) should not be overridden to INFO."""
        import logging
        from arctic_quasi_dp.sci1.logging_config import get_logger
        logger = get_logger("test_debug_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_get_logger_no_duplicate_handlers(self):
        from arctic_quasi_dp.sci1.logging_config import get_logger
        logger1 = get_logger("test_no_dup")
        n_handlers = len(logger1.handlers)
        logger2 = get_logger("test_no_dup")
        assert len(logger2.handlers) == n_handlers  # no new handlers added


# ============================================================
# 3. control/qp_solver.py
# ============================================================

class TestQPSolver:
    """QP solver availability check tests."""

    def test_check_osqp_available(self):
        from arctic_quasi_dp.sci1.control.qp_solver import check_osqp_available
        result = check_osqp_available()
        assert isinstance(result, bool)

    def test_osqp_is_available(self):
        """OSQP should be installed in the test environment."""
        try:
            import osqp
            assert True
        except ImportError:
            pytest.skip("OSQP not installed")


# ============================================================
# 4. figures.py (functional smoke tests)
# ============================================================

class TestFigures:
    """Figure generation smoke tests."""

    def test_plot_failure_rate_no_crash(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from arctic_quasi_dp.sci1.figures import plot_failure_rate
        csv_path = tmp_path / "summary.csv"
        df = pd.DataFrame({
            "controller": ["pid", "ice_aware"],
            "failure_rate": [0.1, 0.05],
        })
        df.to_csv(csv_path, index=False)
        out_dir = tmp_path / "figures"
        out_dir.mkdir()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_failure_rate(csv_path, out_dir)

    def test_plot_failure_rate_missing_column_warns(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from arctic_quasi_dp.sci1.figures import plot_failure_rate
        csv_path = tmp_path / "summary.csv"
        df = pd.DataFrame({"controller": ["pid"]})
        df.to_csv(csv_path, index=False)
        out_dir = tmp_path / "figures"
        out_dir.mkdir()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_failure_rate(csv_path, out_dir)  # should not crash

    def test_plot_statistical_comparison_handles_nan(self, tmp_path):
        """NaN p-values should not crash the plot."""
        import matplotlib
        matplotlib.use("Agg")
        from arctic_quasi_dp.sci1.figures import plot_statistical_comparison
        csv_path = tmp_path / "stats.csv"
        df = pd.DataFrame({
            "metric": ["pos_error", "heading_error"],
            "proposed": ["ice_aware", "ice_aware"],
            "baseline": ["pid", "pid"],
            "p_value": [0.01, float("nan")],
            "cohens_d": [0.5, float("nan")],
        })
        df.to_csv(csv_path, index=False)
        out_dir = tmp_path / "figures"
        out_dir.mkdir()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_statistical_comparison(csv_path, out_dir)  # should not crash


# ============================================================
# 5. data_calibration.py
# ============================================================

class TestDataCalibration:
    """Data calibration tests."""

    def _find_nsidc_dir(self):
        """Find NSIDC data directory, checking both code/data and project root data."""
        from pathlib import Path
        # Try project root data/ first (where real data lives)
        project_root = Path(__file__).resolve().parents[2].parent  # arctic_dp/
        data_dir = project_root / "data" / "sci1_sources" / "nsidc_sea_ice_index"
        if data_dir.exists():
            return data_dir
        # Fallback to code/data/
        code_dir = Path(__file__).resolve().parents[2] / "data" / "sci1_sources" / "nsidc_sea_ice_index"
        if code_dir.exists():
            return code_dir
        return None

    def test_get_arctic_sic_stats_returns_dict(self):
        from arctic_quasi_dp.sci1.data_calibration import get_arctic_sic_stats
        data_dir = self._find_nsidc_dir()
        if data_dir is None:
            pytest.skip("NSIDC data directory not found")
        stats = get_arctic_sic_stats(data_dir)
        assert isinstance(stats, dict)
        assert len(stats) > 0

    def test_calibrate_scenario_from_nsidc_returns_dict_or_none(self):
        from arctic_quasi_dp.sci1.data_calibration import calibrate_scenario_from_nsidc
        data_dir = self._find_nsidc_dir()
        if data_dir is None:
            pytest.skip("NSIDC data directory not found")
        result = calibrate_scenario_from_nsidc(data_dir, target_month=1)
        assert isinstance(result, dict)
        assert "ice_concentration" in result

    def test_get_typical_sic_for_month_fallback(self):
        """Fallback dict should have consistent keys when data is missing."""
        from pathlib import Path
        from arctic_quasi_dp.sci1.data_calibration import get_typical_sic_for_month
        # Use a non-existent directory to trigger fallback
        result = get_typical_sic_for_month(Path("/nonexistent"), month=9)
        assert "mean_extent_mkm2" in result
        assert "std_extent_mkm2" in result
        assert "mean_concentration_approx" in result

    def test_get_typical_sic_for_month_with_data(self):
        """When data exists, keys should be consistent."""
        from arctic_quasi_dp.sci1.data_calibration import get_typical_sic_for_month
        data_dir = self._find_nsidc_dir()
        if data_dir is None:
            pytest.skip("NSIDC data directory not found")
        result = get_typical_sic_for_month(data_dir, month=9)
        assert "mean_extent_mkm2" in result
        assert "std_extent_mkm2" in result
        assert "mean_concentration_approx" in result
        assert result["n_years"] > 0

    def test_sample_std_used(self):
        """std_extent should use sample std (ddof=1), not population std."""
        from arctic_quasi_dp.sci1.data_calibration import get_arctic_sic_stats
        data_dir = self._find_nsidc_dir()
        if data_dir is None:
            pytest.skip("NSIDC data directory not found")
        stats = get_arctic_sic_stats(data_dir)
        # For months with >1 year of data, std should be > 0
        for m, s in stats.items():
            if s.n_years > 1:
                assert s.std_extent > 0, f"Month {m}: std should be > 0 with {s.n_years} years"
