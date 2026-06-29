"""Test data manifest records source provenance."""

import json
import tempfile
from pathlib import Path

import pytest

from arctic_quasi_dp.sci1.data_sources import write_manifest


class TestManifestProvenance:
    """Data manifest must record source types and provenance."""

    def test_manifest_creates_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            write_manifest(path)
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(data, dict)

    def test_manifest_has_literature_calibrations(self):
        """Manifest should include literature calibrations for provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            write_manifest(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            assert "literature_calibrations" in data

    def test_scenarios_mark_data_type(self):
        """Scenarios should include evidence_level field."""
        from arctic_quasi_dp.sci1.scenarios import build_sci1_scenarios
        scenarios = build_sci1_scenarios("smoke")
        for s in scenarios:
            assert hasattr(s, "evidence_level"), (
                f"Scenario {s.scenario_id} missing evidence_level"
            )
            valid_levels = {
                "literature_calibrated", "literature-calibrated synthetic",
                "synthetic", "observed_gridded", "reanalysis", "in_situ",
                "data_driven_copernicus",
            }
            assert s.evidence_level in valid_levels, (
                f"Unknown evidence_level: {s.evidence_level}"
            )
