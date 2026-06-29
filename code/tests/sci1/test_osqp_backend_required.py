"""OSQP backend must be available for formal method paper experiments."""

import pytest


class TestOSQPBackendRequired:
    """Verify OSQP is available and used by safety filter."""

    def test_osqp_importable(self):
        """OSQP must be importable for formal experiments."""
        import osqp
        assert hasattr(osqp, 'OSQP')

    def test_safety_filter_uses_osqp(self):
        """SoftHOCBFSafetyFilter should use OSQP when available."""
        from arctic_quasi_dp.sci1.control.safety_filter import SoftHOCBFSafetyFilter
        import numpy as np

        f = SoftHOCBFSafetyFilter()
        assert f._osqp_available, "OSQP should be detected as available"

        # Run a simple QP
        state = np.array([5.0, 3.0, 0.1, 0.5, 0.2, 0.01])
        tau_des = np.array([100.0, 50.0, 1000.0])
        result = f.filter(state, tau_des, (0.0, 0.0), 0.0)
        assert result.solver_backend == "osqp", f"Expected osqp, got {result.solver_backend}"

    def test_method_smoke_backend_is_osqp(self, tmp_path):
        """Method smoke trace must show OSQP backend for safety-filtered controllers."""
        import pandas as pd
        import os
        trace_dir = os.environ.get("SCI1_METHOD_SMOKE_DIR", None)
        if trace_dir is None:
            pytest.skip("SCI1_METHOD_SMOKE_DIR not set; run method_smoke first")

        # Check at least one safety-filtered controller trace
        import glob
        traces = glob.glob(os.path.join(trace_dir, "*fixed_soft_hocbf*.csv"))
        if not traces:
            pytest.skip("No fixed_soft_hocbf trace found")

        df = pd.read_csv(traces[0])
        if "safety_filter_solver_backend" not in df.columns:
            pytest.fail("safety_filter_solver_backend column missing from trace")

        backends = df["safety_filter_solver_backend"].dropna().unique()
        assert "osqp" in backends, f"Expected osqp backend, got {backends}"
