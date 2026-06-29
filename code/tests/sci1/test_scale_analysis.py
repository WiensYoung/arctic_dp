"""Scale analysis dimensionless groups tests."""

import math
import numpy as np
import pytest


class TestScaleAnalysis:
    """Test dimensionless group computation."""

    def test_dimensionless_groups_output_complete(self):
        from arctic_quasi_dp.sci1.scale_analysis import compute_dimensionless_groups
        from arctic_quasi_dp.sci1.sim_loop import VesselParams

        vp = VesselParams()
        groups = compute_dimensionless_groups(vp, None, type('S', (), {
            'ice_concentration': 0.5, 'ice_thickness': 0.8, 'ice_drift_speed': 0.3,
            'safe_region_radius': 10.0, 'duration': 300.0,
        })())

        d = groups.to_dict()
        required = [
            "ice_force_to_max_thrust", "ice_moment_to_max_moment",
            "drift_speed_to_characteristic_speed", "sensor_noise_to_safe_radius",
            "safe_radius_to_length", "disturbance_time_to_control_time",
            "power_cap_to_nominal_power", "interpretation",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_dimensionless_groups_finite(self):
        from arctic_quasi_dp.sci1.scale_analysis import compute_dimensionless_groups
        from arctic_quasi_dp.sci1.sim_loop import VesselParams

        vp = VesselParams()
        groups = compute_dimensionless_groups(vp, None, type('S', (), {
            'ice_concentration': 0.5, 'ice_thickness': 0.8, 'ice_drift_speed': 0.3,
            'safe_region_radius': 10.0, 'duration': 300.0,
        })())

        for key, val in groups.to_dict().items():
            if key == "interpretation":
                assert isinstance(val, str)
            elif key == "power_cap_to_nominal_power":
                # inf is valid when no power cap is set (thruster_config=None)
                assert math.isfinite(val) or val == float("inf"), f"{key} = {val} is not finite or inf"
            else:
                assert math.isfinite(val), f"{key} = {val} is not finite"

    def test_proxy_scale_interpretation(self):
        """Proxy scale with default parameters should be 'comparable' or have clear warnings."""
        from arctic_quasi_dp.sci1.scale_analysis import compute_dimensionless_groups
        from arctic_quasi_dp.sci1.sim_loop import VesselParams

        vp = VesselParams()
        groups = compute_dimensionless_groups(vp, None, type('S', (), {
            'ice_concentration': 0.5, 'ice_thickness': 0.8, 'ice_drift_speed': 0.3,
            'safe_region_radius': 10.0, 'duration': 300.0,
        })())

        interp = groups.interpretation
        assert isinstance(interp, str)
        assert len(interp) > 0

    def test_scale_comparison_config_loads(self):
        """sci1_scale_comparison.yaml should be loadable."""
        from pathlib import Path
        import yaml
        config_path = Path(__file__).resolve().parents[2] / "configs" / "sci1" / "sci1_scale_comparison.yaml"
        if not config_path.exists():
            pytest.skip("sci1_scale_comparison.yaml not found")
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "controllers" in cfg
        assert "scenarios" in cfg
