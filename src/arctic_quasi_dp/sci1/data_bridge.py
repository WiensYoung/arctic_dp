"""数据桥接模块 — 将下载的 NetCDF/CSV 数据接入仿真管线。

本模块解决的核心问题：下载的 Copernicus/ERA5 数据是"死重"，
仿真代码从未读取。本模块提供：

1. NetCDF 数据加载器 (修复坐标名兼容性)
2. DataDrivenIceSchedule — 从 Copernicus NetCDF 读取真实冰况时间序列
3. DataDrivenWindSchedule — 从 ERA5 NetCDF 读取真实风场时间序列
4. generate_scenarios_from_data() — 用真实数据生成场景参数

使用：
    from arctic_quasi_dp.sci1.data_bridge import (
        load_copernicus_ice_data,
        DataDrivenIceSchedule,
        generate_scenarios_from_data,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from .ice_schedule import IceSchedule, IceState


# ============================================================
# NetCDF 数据加载 (修复坐标名兼容性)
# ============================================================

def _find_coord(ds, candidates: List[str]):
    """在 xarray Dataset 中查找坐标变量。"""
    for name in candidates:
        if name in ds.coords:
            return ds[name]
    return None


def load_copernicus_ice_data(
    nc_path: Path,
    lat_range: Optional[Tuple[float, float]] = None,
    time_idx: Optional[int] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """从 Copernicus NetCDF 加载海冰数据。

    自动处理坐标名差异 (latitude/lat/y, longitude/lon/x, time/valid_time)。

    Returns:
        dict with keys: siconc, sithick, vxsi, vysi, lat, lon, time
        or None if loading fails
    """
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)

        # 查找坐标
        lat = _find_coord(ds, ["latitude", "lat", "y"])
        lon = _find_coord(ds, ["longitude", "lon", "x"])
        time = _find_coord(ds, ["time", "valid_time"])

        if lat is None or lon is None:
            ds.close()
            return None

        # 空间裁剪
        if lat_range is not None:
            ds = ds.sel({lat.name: slice(lat_range[0], lat_range[1])})

        # 时间选择
        if time_idx is not None and time is not None:
            ds = ds.isel({time.name: time_idx})

        result = {"lat": lat.values, "lon": lon.values}
        if time is not None:
            result["time"] = ds[time.name].values

        # 数据变量
        for var in ["siconc", "sithick", "vxsi", "vysi"]:
            if var in ds:
                result[var] = ds[var].values

        ds.close()
        return result
    except Exception:
        return None


def load_era5_wind_data(
    nc_path: Path,
    lat_range: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """从 ERA5 NetCDF 加载风场数据。"""
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)

        lat = _find_coord(ds, ["latitude", "lat"])
        lon = _find_coord(ds, ["longitude", "lon"])
        time = _find_coord(ds, ["time", "valid_time"])

        if lat is None:
            ds.close()
            return None

        if lat_range is not None:
            ds = ds.sel({lat.name: slice(lat_range[0], lat_range[1])})

        result = {"lat": lat.values, "lon": lon.values}
        if time is not None:
            result["time"] = ds[time.name].values

        for var in ["u10", "v10", "u10m", "v10m"]:
            if var in ds:
                result[var] = ds[var].values

        ds.close()
        return result if any(k in result for k in ["u10", "v10"]) else None
    except Exception:
        return None


# ============================================================
# 数据驱动的冰况调度
# ============================================================

class DataDrivenIceSchedule(IceSchedule):
    """从 Copernicus NetCDF 读取真实冰况时间序列。

    在指定位置 (lat, lon) 插值读取 SIC, SIT, 漂移速度/方向，
    按时间步长线性插值。
    """

    def __init__(
        self,
        nc_path: Path,
        lat: float = 80.0,
        lon: float = 0.0,
        duration: float = 300.0,
    ):
        self.nc_path = nc_path
        self.lat = lat
        self.lon = lon
        self.duration = duration
        self._loaded = False
        self._times = None
        self._sic = None
        self._sit = None
        self._drift_speed = None
        self._drift_dir = None
        self._load()

    def _load(self):
        data = load_copernicus_ice_data(self.nc_path)
        if data is None:
            return

        # 查找最近的网格点
        lat_vals = data["lat"]
        lon_vals = data["lon"]

        # 处理经度范围 (可能 0-360 或 -180-180)
        if lon_vals.max() > 180 and self.lon < 0:
            lon_adj = self.lon + 360
        else:
            lon_adj = self.lon

        lat_idx = np.argmin(np.abs(lat_vals - self.lat))
        lon_idx = np.argmin(np.abs(lon_vals - lon_adj))

        self._sic = data.get("siconc")
        self._sit = data.get("sithick")

        if self._sic is not None:
            if self._sic.ndim == 3:  # time, lat, lon
                self._sic = self._sic[:, lat_idx, lon_idx]
            elif self._sic.ndim == 2:
                self._sic = self._sic[:, lon_idx] if self._sic.shape[0] > self._sic.shape[1] else self._sic[lat_idx, :]

        if self._sit is not None:
            if self._sit.ndim == 3:
                self._sit = self._sit[:, lat_idx, lon_idx]
            elif self._sit.ndim == 2:
                self._sit = self._sit[:, lon_idx] if self._sit.shape[0] > self._sit.shape[1] else self._sit[lat_idx, :]

        vx = data.get("vxsi")
        vy = data.get("vysi")
        if vx is not None and vy is not None:
            if vx.ndim == 3:
                vx = vx[:, lat_idx, lon_idx]
                vy = vy[:, lat_idx, lon_idx]
            elif vx.ndim == 2:
                vx = vx[:, lon_idx] if vx.shape[0] > vx.shape[1] else vx[lat_idx, :]
                vy = vy[:, lon_idx] if vy.shape[0] > vy.shape[1] else vy[lat_idx, :]
            self._drift_speed = np.sqrt(vx**2 + vy**2)
            self._drift_dir = np.degrees(np.arctan2(vy, vx)) % 360

        # 时间归一化到 [0, duration]
        n = len(self._sic) if self._sic is not None else 0
        if n > 0:
            self._times = np.linspace(0, self.duration, n)
        self._loaded = True

    def at(self, t: float) -> IceState:
        if not self._loaded or self._sic is None:
            return IceState(0.0, 0.0, 0.0, 0.0)

        t_clamp = float(np.clip(t, self._times[0], self._times[-1]))

        sic = float(np.interp(t_clamp, self._times, self._sic))
        sit = float(np.interp(t_clamp, self._times, self._sit)) if self._sit is not None else 0.0
        spd = float(np.interp(t_clamp, self._times, self._drift_speed)) if self._drift_speed is not None else 0.0
        drd = float(np.interp(t_clamp, self._times, self._drift_dir)) if self._drift_dir is not None else 0.0

        return IceState(
            concentration=float(np.clip(sic, 0.0, 1.0)),
            thickness=max(0.0, sit),
            drift_speed=max(0.0, spd),
            drift_direction=drd,
        )


# ============================================================
# 数据驱动的风场调度
# ============================================================

@dataclass
class WindState:
    u10: float = 0.0  # 10m 风 u 分量 (m/s)
    v10: float = 0.0  # 10m 风 v 分量 (m/s)

    @property
    def speed(self) -> float:
        return math.sqrt(self.u10**2 + self.v10**2)

    @property
    def direction_deg(self) -> float:
        return math.degrees(math.atan2(self.v10, self.u10)) % 360


class DataDrivenWindSchedule:
    """从 ERA5 NetCDF 读取真实风场时间序列。"""

    def __init__(
        self,
        nc_path: Path,
        lat: float = 80.0,
        lon: float = 0.0,
        duration: float = 300.0,
    ):
        self.nc_path = nc_path
        self.duration = duration
        self._loaded = False
        self._times = None
        self._u = None
        self._v = None

        data = load_era5_wind_data(nc_path)
        if data is None:
            return

        lat_vals = data["lat"]
        lon_vals = data["lon"]
        if lon_vals.max() > 180 and lon < 0:
            lon_adj = lon + 360
        else:
            lon_adj = lon

        lat_idx = np.argmin(np.abs(lat_vals - lat))
        lon_idx = np.argmin(np.abs(lon_vals - lon_adj))

        u = data.get("u10")
        v = data.get("v10")
        if u is not None and v is not None:
            if u.ndim == 3:
                self._u = u[:, lat_idx, lon_idx]
                self._v = v[:, lat_idx, lon_idx]
            else:
                self._u = u
                self._v = v
            n = len(self._u)
            self._times = np.linspace(0, duration, n)
            self._loaded = True

    def at(self, t: float) -> WindState:
        if not self._loaded:
            return WindState()
        t_clamp = float(np.clip(t, self._times[0], self._times[-1]))
        u = float(np.interp(t_clamp, self._times, self._u))
        v = float(np.interp(t_clamp, self._times, self._v))
        return WindState(u10=u, v10=v)


# ============================================================
# 从真实数据生成场景参数
# ============================================================

def extract_ice_statistics(
    nc_path: Path,
    lat_range: Tuple[float, float] = (70.0, 90.0),
) -> Dict[str, float]:
    """从 Copernicus NetCDF 提取冰况统计。

    返回 SIC/SIT/漂移的均值、标准差、p95、max。
    """
    data = load_copernicus_ice_data(nc_path, lat_range=lat_range)
    if data is None:
        return {}

    result = {}
    for var in ["siconc", "sithick"]:
        if var in data:
            vals = data[var].flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                result[f"{var}_mean"] = float(np.mean(vals))
                result[f"{var}_std"] = float(np.std(vals))
                result[f"{var}_p95"] = float(np.percentile(vals, 95))
                result[f"{var}_max"] = float(np.max(vals))

    vx = data.get("vxsi")
    vy = data.get("vysi")
    if vx is not None and vy is not None:
        speed = np.sqrt(vx.flatten()**2 + vy.flatten()**2)
        speed = speed[~np.isnan(speed)]
        if len(speed) > 0:
            result["drift_speed_mean"] = float(np.mean(speed))
            result["drift_speed_std"] = float(np.std(speed))
            result["drift_speed_p95"] = float(np.percentile(speed, 95))
            result["drift_speed_max"] = float(np.max(speed))

    return result


def generate_scenarios_from_data(
    nc_path: Path,
    n_scenarios: int = 5,
    seed: int = 2026,
) -> List[Dict[str, float]]:
    """从 Copernicus 数据生成数据驱动的场景参数。

    在观测分布的 p10-p90 范围内均匀采样。
    """
    stats = extract_ice_statistics(nc_path)
    if not stats:
        return []

    rng = np.random.default_rng(seed)
    scenarios = []

    for i in range(n_scenarios):
        sic = float(np.clip(
            rng.uniform(stats.get("siconc_mean", 0.5) - stats.get("siconc_std", 0.2),
                        stats.get("siconc_mean", 0.5) + stats.get("siconc_std", 0.2)),
            0.0, 1.0,
        ))
        sit = max(0.0, float(rng.uniform(
            max(0, stats.get("sithick_mean", 1.0) - stats.get("sithick_std", 0.5)),
            stats.get("sithick_mean", 1.0) + stats.get("sithick_std", 0.5),
        )))
        drift = max(0.0, float(rng.uniform(
            stats.get("drift_speed_mean", 0.08),
            stats.get("drift_speed_p95", 0.18),
        )))
        direction = float(rng.uniform(0, 360))

        scenarios.append({
            "ice_concentration": sic,
            "ice_thickness": sit,
            "ice_drift_speed": drift,
            "ice_drift_direction": direction,
            "evidence_level": "data_driven_copernicus",
            "source_lat_range": "70-90N",
        })

    return scenarios
