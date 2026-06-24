"""数据桥接模块测试。"""

import numpy as np
import pytest
from pathlib import Path

from arctic_quasi_dp.sci1.data_bridge import (
    DataDrivenIceSchedule,
    DataDrivenWindSchedule,
    WindState,
    extract_ice_statistics,
    load_copernicus_ice_data,
    load_era5_wind_data,
)


# 检查是否有真实数据文件
COPERNICUS_FILE = Path("data/sci1_sources/copernicus/arctic_ice_drift_2020_jan.nc")
ERA5_WIND_FILE = Path("data/sci1_sources/era5/era5_arctic_wind_2020_jan_full.nc")
HAS_COPERNICUS = COPERNICUS_FILE.exists()
HAS_ERA5_WIND = ERA5_WIND_FILE.exists()


class TestWindState:
    def test_speed(self):
        w = WindState(u10=3.0, v10=4.0)
        assert abs(w.speed - 5.0) < 0.01

    def test_direction(self):
        w = WindState(u10=0.0, v10=1.0)
        assert abs(w.direction_deg - 90.0) < 0.1


class TestDataDrivenIceSchedule:
    @pytest.mark.skipif(not HAS_COPERNICUS, reason="Copernicus data not downloaded")
    def test_load_and_interpolate(self):
        schedule = DataDrivenIceSchedule(COPERNICUS_FILE, lat=80.0, lon=0.0, duration=300.0)
        s = schedule.at(0.0)
        assert 0.0 <= s.concentration <= 1.0
        assert s.thickness >= 0.0

    @pytest.mark.skipif(not HAS_COPERNICUS, reason="Copernicus data not downloaded")
    def test_time_interpolation(self):
        schedule = DataDrivenIceSchedule(COPERNICUS_FILE, lat=80.0, lon=0.0, duration=300.0)
        s0 = schedule.at(0.0)
        s150 = schedule.at(150.0)
        s300 = schedule.at(300.0)
        # 不同时间应有不同值 (除非数据恰好相同)
        assert np.isfinite(s0.concentration)
        assert np.isfinite(s300.concentration)

    def test_missing_file_returns_zero(self):
        schedule = DataDrivenIceSchedule(Path("nonexistent.nc"), lat=80.0, lon=0.0)
        s = schedule.at(0.0)
        assert s.concentration == 0.0


class TestExtractIceStatistics:
    @pytest.mark.skipif(not HAS_COPERNICUS, reason="Copernicus data not downloaded")
    def test_extract_stats(self):
        stats = extract_ice_statistics(COPERNICUS_FILE, lat_range=(75.0, 85.0))
        assert "siconc_mean" in stats
        assert "sithick_mean" in stats
        assert "drift_speed_mean" in stats
        assert 0.0 <= stats["siconc_mean"] <= 1.0
        assert stats["sithick_mean"] >= 0.0
        assert stats["drift_speed_mean"] >= 0.0


class TestLoadCopernicusData:
    @pytest.mark.skipif(not HAS_COPERNICUS, reason="Copernicus data not downloaded")
    def test_load_returns_dict(self):
        data = load_copernicus_ice_data(COPERNICUS_FILE)
        assert data is not None
        assert "siconc" in data
        assert "lat" in data

    def test_missing_file_returns_none(self):
        data = load_copernicus_ice_data(Path("nonexistent.nc"))
        assert data is None


class TestLoadERA5Wind:
    @pytest.mark.skipif(not HAS_ERA5_WIND, reason="ERA5 wind data not downloaded")
    def test_load_returns_dict(self):
        data = load_era5_wind_data(ERA5_WIND_FILE)
        assert data is not None
        assert "u10" in data or "v10" in data
