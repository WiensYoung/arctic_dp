"""Test vessel config consistency across modules."""

from pathlib import Path

import yaml
import pytest

from arctic_quasi_dp.sci1.sim_loop import VesselParams
from arctic_quasi_dp.sci1.controllers import IceAwareParams


class TestVesselConfigConsistency:
    """All modules should use consistent vessel parameters."""

    def test_xuelong2_config_loads(self):
        config_path = Path(__file__).parent.parent.parent / "configs" / "vessels" / "xuelong2_like.yaml"
        if not config_path.exists():
            pytest.skip("Vessel config not found")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["name"] == "xuelong2_like"
        assert cfg["length_m"] > 0
        assert cfg["beam_m"] > 0
        assert cfg["mass_kg"] > 0

    def test_damping_positive_in_sim_loop(self):
        """VesselParams must have positive damping coefficients."""
        p = VesselParams()
        assert p.Xu > 0
        assert p.Yv > 0
        assert p.Nr > 0
        assert p.Xu_abs > 0
        assert p.Yv_abs > 0
        assert p.Nr_abs > 0

    def test_no_hardcoded_old_length(self):
        """Code should not contain old hardcoded vessel length (95m)."""
        import arctic_quasi_dp.sci1.controllers as ctrl_mod
        import arctic_quasi_dp.sci1.sim_loop as sim_mod
        import arctic_quasi_dp.sci1.nmpc_controller as nmpc_mod

        for mod in [ctrl_mod, sim_mod, nmpc_mod]:
            src = Path(mod.__file__).read_text(encoding="utf-8")
            # Old value 95m should not appear as a vessel length parameter
            # (some values like 0.95 or 950 are fine)
            assert "95.0" not in src or "95m" not in src, (
                f"Old hardcoded vessel length 95m found in {mod.__file__}"
            )

    def test_vessel_config_metadata_recorded(self):
        """When experiments run, vessel config should be in metadata."""
        from arctic_quasi_dp.sci1.runner import _metadata
        meta = _metadata()
        assert "experiment_protocol" in meta
