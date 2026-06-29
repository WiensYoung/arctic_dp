"""Test unit conversion consistency across the project."""

import numpy as np
import pytest

from arctic_quasi_dp.sci1.units import mpa_to_pa, kn_to_n, mn_to_n, UNITS_MANIFEST


class TestUnitsConsistency:

    def test_mpa_to_pa(self):
        assert mpa_to_pa(1.0) == 1_000_000.0
        assert mpa_to_pa(2.0) == 2_000_000.0
        assert mpa_to_pa(0.0) == 0.0

    def test_kn_to_n(self):
        assert kn_to_n(1.0) == 1000.0
        assert kn_to_n(500.0) == 500_000.0

    def test_mn_to_n(self):
        assert mn_to_n(1.0) == 1_000_000.0

    def test_units_manifest_complete(self):
        required = {"force", "moment", "mass", "length", "time", "ice_strength"}
        assert required.issubset(set(UNITS_MANIFEST.keys()))

    def test_units_manifest_si(self):
        assert UNITS_MANIFEST["force"] == "N"
        assert UNITS_MANIFEST["moment"] == "N_m"
        assert UNITS_MANIFEST["mass"] == "kg"
        assert UNITS_MANIFEST["ice_strength"] == "Pa"

    def test_no_1000_conversion_in_ice_force(self):
        """Ice force must use * 1e6 for MPa->Pa, not * 1000."""
        from pathlib import Path
        import arctic_quasi_dp.sci1.controllers as ctrl_mod
        import arctic_quasi_dp.sci1.sim_loop as sim_mod
        import arctic_quasi_dp.sci1.nmpc_controller as nmpc_mod

        for mod in [ctrl_mod, sim_mod, nmpc_mod]:
            src = Path(mod.__file__).read_text(encoding="utf-8")
            # Should not have * 1000.0 in ice force context
            # (timing * 1000.0 is OK)
            lines = src.split("\n")
            for i, line in enumerate(lines):
                if "1000.0" in line and "ice" in line.lower():
                    # Only timing conversions should have * 1000.0
                    assert "perf_counter" in line or "time" in line.lower(), (
                        f"Found * 1000.0 in ice context at {mod.__file__}:{i+1}: {line.strip()}"
                    )
