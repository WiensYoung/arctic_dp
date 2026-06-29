"""HOCBF diagnostics completeness tests."""

import math
import numpy as np
import pytest


class TestHOCBFDiagnostics:
    """Verify HOCBF margin/slack statistics are computable."""

    def test_safety_filter_margin_in_result(self):
        """SafetyFilterResult should contain hocbf_margin."""
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        import numpy as np

        f = SoftHOCBFSafetyFilter()
        state = np.array([5.0, 3.0, 0.1, 0.5, 0.2, 0.01])
        tau_des = np.array([100.0, 50.0, 1000.0])
        result = f.filter(state, tau_des, (0.0, 0.0), 0.0)
        assert hasattr(result, 'hocbf_margin')
        assert math.isfinite(result.hocbf_margin)

    def test_safety_filter_slack_in_result(self):
        """SafetyFilterResult should contain slack."""
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        import numpy as np

        f = SoftHOCBFSafetyFilter()
        state = np.array([5.0, 3.0, 0.1, 0.5, 0.2, 0.01])
        tau_des = np.array([100.0, 50.0, 1000.0])
        result = f.filter(state, tau_des, (0.0, 0.0), 0.0)
        assert hasattr(result, 'slack')
        assert result.slack >= 0.0

    def test_metrics_extracts_hocbf_margin(self):
        """summarize_run should extract hocbf_margin_min."""
        import pandas as pd
        from arctic_quasi_dp.sci1.metrics import summarize_run

        # Create minimal trace with safety filter fields
        df = pd.DataFrame({
            "time": [0.1, 0.2, 0.3],
            "position_error": [1.0, 2.0, 3.0],
            "heading_error": [0.01, 0.02, 0.03],
            "tau_x": [100.0, 200.0, 300.0],
            "tau_y": [50.0, 100.0, 150.0],
            "solver_time_ms": [1.0, 2.0, 3.0],
            "solver_success": [1, 1, 1],
            "safety_filter_qp_success": [1, 1, 1],
            "safety_filter_solve_time_ms": [1.0, 2.0, 3.0],
            "safety_filter_slack": [0.0, 0.01, 0.02],
            "safety_filter_hocbf_margin": [0.5, 0.3, 0.1],
            "safety_filter_correction_norm": [0.0, 10.0, 20.0],
            "safety_filter_solver_backend": ["osqp", "osqp", "osqp"],
            "risk_total": [0.1, 0.2, 0.3],
            "risk_ice": [0.1, 0.1, 0.1],
            "risk_cvar": [0.05, 0.05, 0.05],
            "cbf_slack": [5.0, 3.0, 1.0],
            "violation": [0, 0, 0],
            "energy": [0.0, 1.0, 2.0],
        })
        result = summarize_run(df, scenario_id="test", controller="test", seed=0, dt=0.1, max_force=3000.0)
        assert "safety_filter_hocbf_margin_min" in result
        assert result["safety_filter_hocbf_margin_min"] == pytest.approx(0.1)
        assert "safety_filter_slack_active_rate" in result
        assert "safety_filter_solver_backend" in result
        assert result["safety_filter_solver_backend"] == "osqp"
