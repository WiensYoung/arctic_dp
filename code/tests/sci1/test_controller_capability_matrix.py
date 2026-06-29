"""Test controller capability matrix correctness."""

import pytest
from arctic_quasi_dp.sci1.tables import generate_table2_controller_matrix
from arctic_quasi_dp.sci1.runner import _CONTROLLER_CAPABILITIES


class TestControllerCapabilityMatrix:
    """Controller capability matrix must be accurate."""

    def test_matrix_has_all_controllers(self):
        df = generate_table2_controller_matrix()
        assert len(df) >= 10, f"Expected >= 10 controllers, got {len(df)}"

    def test_full_has_all_capabilities(self):
        caps = _CONTROLLER_CAPABILITIES["full"]
        assert caps["observer"]
        assert caps["cvar"]
        assert caps["cbf"]
        assert caps["mode_supervisor"]
        assert caps["quasi_dp"]
        assert caps["escape"]

    def test_full_not_oracle(self):
        caps = _CONTROLLER_CAPABILITIES["full"]
        assert not caps["oracle"], "full must NOT use oracle ice"

    def test_oracle_full_is_oracle(self):
        caps = _CONTROLLER_CAPABILITIES["oracle_full"]
        assert caps["oracle"], "oracle_full must use oracle ice"

    def test_pid_no_advanced_features(self):
        caps = _CONTROLLER_CAPABILITIES["pid"]
        assert not caps["observer"]
        assert not caps["cvar"]
        assert not caps["cbf"]
        assert not caps["mode_supervisor"]

    def test_ablation_missing_one_feature(self):
        for abl in ["no_cbf", "no_cvar", "no_observer", "no_fallback"]:
            caps = _CONTROLLER_CAPABILITIES[abl]
            assert caps["mode_supervisor"], f"{abl} should have mode_supervisor"

    def test_no_cbf_disables_cbf(self):
        assert not _CONTROLLER_CAPABILITIES["no_cbf"]["cbf"]

    def test_no_cvar_disables_cvar(self):
        assert not _CONTROLLER_CAPABILITIES["no_cvar"]["cvar"]

    def test_no_observer_disables_observer(self):
        assert not _CONTROLLER_CAPABILITIES["no_observer"]["observer"]

    def test_matrix_to_csv(self):
        df = generate_table2_controller_matrix()
        assert "controller" in df.columns
        assert "uses_observer" in df.columns or "observer" in df.columns
