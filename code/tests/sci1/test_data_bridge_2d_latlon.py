"""Test data_bridge handles 2D lat/lon grids and coordinate variations."""

import numpy as np
import pytest

from arctic_quasi_dp.sci1.data_bridge import _find_coord


class TestFindCoord:
    """Test coordinate finding logic."""

    def test_find_coord_in_coords(self):
        """Should find coordinates in ds.coords."""
        try:
            import xarray as xr
        except ImportError:
            pytest.skip("xarray not installed")

        ds = xr.Dataset(
            {"temp": (["latitude", "longitude"], np.zeros((3, 3)))},
            coords={"latitude": [70, 80, 90], "longitude": [-10, 0, 10]},
        )
        lat = _find_coord(ds, ["latitude", "lat", "y"])
        assert lat is not None
        assert len(lat) == 3

    def test_find_coord_in_variables(self):
        """Should find coordinates in ds.variables (not in coords)."""
        try:
            import xarray as xr
        except ImportError:
            pytest.skip("xarray not installed")

        ds = xr.Dataset({
            "temp": (["y", "x"], np.zeros((3, 3))),
            "lat": (["y", "x"], np.array([[70, 70, 70], [80, 80, 80], [90, 90, 90]])),
            "lon": (["y", "x"], np.array([[-10, 0, 10], [-10, 0, 10], [-10, 0, 10]])),
        })
        lat = _find_coord(ds, ["latitude", "lat", "y"])
        assert lat is not None

    def test_find_coord_none_when_missing(self):
        """Should return None when no matching coordinate found."""
        try:
            import xarray as xr
        except ImportError:
            pytest.skip("xarray not installed")

        ds = xr.Dataset({"temp": (["z"], [1, 2, 3])})
        lat = _find_coord(ds, ["latitude", "lat", "y"])
        assert lat is None


class Test2DLatLon:
    """Test that 2D lat/lon grids are handled."""

    def test_2d_grid_shape(self):
        """2D lat/lon arrays should have consistent shape."""
        try:
            import xarray as xr
        except ImportError:
            pytest.skip("xarray not installed")

        # 2D lat/lon (common in reanalysis products)
        lat2d = np.array([[70, 70, 70], [80, 80, 80], [90, 90, 90]])
        lon2d = np.array([[-10, 0, 10], [-10, 0, 10], [-10, 0, 10]])
        ds = xr.Dataset({
            "siconc": (["y", "x"], np.random.rand(3, 3)),
            "lat": (["y", "x"], lat2d),
            "lon": (["y", "x"], lon2d),
        })
        lat = _find_coord(ds, ["lat", "latitude"])
        assert lat is not None
        assert lat.ndim == 2


class TestDataManifestProvenance:
    """Test that data manifest records provenance correctly."""

    def test_data_manifest_has_provenance_fields(self):
        """Generated manifest should record data source type."""
        from arctic_quasi_dp.sci1.data_sources import write_manifest
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "test_manifest.json"
            write_manifest(manifest_path)
            import json
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert "sources" in manifest or "created_utc" in manifest
