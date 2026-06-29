"""Real replay manifest and fail-fast tests."""

from pathlib import Path
import pytest


class TestRealReplayManifest:
    """Verify mock fixture is labeled and real replay fails fast without data."""

    def test_mock_fixture_labeled_as_mock(self):
        """Mock fixture in manifest must be labeled as mock_fixture."""
        from arctic_quasi_dp.sci1.data_sources import PACKAGED_REPLAY_SOURCES
        for src in PACKAGED_REPLAY_SOURCES:
            assert "mock" in src.status.lower() or "fixture" in src.status.lower(), \
                f"Source {src.name} status={src.status} must contain 'mock' or 'fixture'"
            assert "not a real" in src.access_note.lower() or "mock" in src.access_note.lower(), \
                f"Source {src.name} access_note must state it is not real data"

    def test_mock_fixture_has_checksum(self):
        """Mock fixture must have a SHA256 checksum."""
        from arctic_quasi_dp.sci1.data_sources import PACKAGED_REPLAY_SOURCES
        for src in PACKAGED_REPLAY_SOURCES:
            assert src.checksum_sha256 is not None and len(src.checksum_sha256) > 0, \
                f"Source {src.name} must have checksum_sha256"

    def test_mock_fixture_checksum_matches_file(self):
        """Mock fixture SHA256 must match actual file on disk."""
        from arctic_quasi_dp.sci1.data_sources import PACKAGED_REPLAY_SOURCES, _file_sha256
        # data/ is at project root (parents[2] from tests/sci1/), not in code/
        project_root = Path(__file__).resolve().parents[2].parent
        for src in PACKAGED_REPLAY_SOURCES:
            if src.local_path:
                fpath = project_root / src.local_path
                if fpath.exists():
                    actual = _file_sha256(fpath)
                    assert actual == src.checksum_sha256, \
                        f"SHA256 mismatch for {src.name}: expected {src.checksum_sha256}, got {actual}"

    def test_real_replay_config_fail_fast(self):
        """sci1_real_replay_h1.yaml must exist and require real data."""
        config_path = Path(__file__).resolve().parents[2] / "configs" / "sci1" / "sci1_real_replay_h1.yaml"
        if not config_path.exists():
            pytest.skip("sci1_real_replay_h1.yaml not found")
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        runtime = cfg.get("runtime", {})
        assert runtime.get("fail_fast") is True or cfg.get("fail_fast_on_missing_data") is True, \
            "Real replay config must have runtime.fail_fast: true or fail_fast_on_missing_data: true"

    def test_real_data_sources_exist(self):
        """REAL_DATA_SOURCES must list real downloaded data."""
        from arctic_quasi_dp.sci1.data_sources import REAL_DATA_SOURCES
        assert len(REAL_DATA_SOURCES) >= 2, "Must have at least ERA5 wind + Copernicus ice"
        for src in REAL_DATA_SOURCES:
            assert src.status == "downloaded", f"Source {src.name} status={src.status}, expected 'downloaded'"
            assert "mock" not in src.name.lower() or "real" in src.name.lower(), \
                f"Real data source {src.name} should not be labeled as mock"

    def test_real_data_files_on_disk(self):
        """Real data files referenced in REAL_DATA_SOURCES must exist on disk."""
        from arctic_quasi_dp.sci1.data_sources import REAL_DATA_SOURCES
        project_root = Path(__file__).resolve().parents[2].parent
        for src in REAL_DATA_SOURCES:
            if src.local_path:
                fpath = project_root / src.local_path
                assert fpath.exists(), f"Real data file not found: {fpath}"

    def test_real_data_loadable(self):
        """Real data files must be loadable by data_bridge."""
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[2].parent

        # ERA5 wind
        era5_wind = project_root / "data/sci1_sources/era5/era5_arctic_wind_2020_jan1_7_real.nc"
        if era5_wind.exists():
            from arctic_quasi_dp.sci1.data_bridge import load_era5_wind_data
            result = load_era5_wind_data(era5_wind, lat_range=(85, 75))
            assert result is not None, "ERA5 wind data failed to load"
            assert result["u10"].size > 0, "ERA5 wind u10 is empty"

        # Copernicus ice
        cop_ice = project_root / "data/sci1_sources/copernicus/arctic_ice_2024_jan1_7_real.nc"
        if cop_ice.exists():
            from arctic_quasi_dp.sci1.data_bridge import load_copernicus_ice_data
            result = load_copernicus_ice_data(cop_ice, lat_range=(75, 85))
            assert result is not None, "Copernicus ice data failed to load"
            assert "siconc" in result, "Copernicus ice data missing siconc"
