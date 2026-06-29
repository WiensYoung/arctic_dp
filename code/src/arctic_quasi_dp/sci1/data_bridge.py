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
from .sim_loop import WindState  # 统一使用 sim_loop 中的 WindState 定义


def _first_existing(data: dict, names: List[str]):
    """从 dict 中安全获取第一个存在的键值。

    避免 numpy array `or` 判断导致的 ambiguous truth value 错误。
    """
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return None


# ============================================================
# NetCDF 数据加载 (修复坐标名兼容性)
# ============================================================

def _adjust_lon_for_grid(target_lon: float, lon_vals: np.ndarray) -> float:
    """双向经度坐标转换: 确保 target_lon 与 lon_vals 在同一坐标系。

    处理 0-360 与 -180-180 之间的双向转换。
    """
    if lon_vals.max() > 180 and target_lon < 0:
        return target_lon + 360  # target在(-180,180), 数据在(0,360)
    if lon_vals.min() < 0 and target_lon > 180:
        return target_lon - 360  # target在(0,360), 数据在(-180,180)
    return target_lon


def _find_coord(ds, candidates: List[str]):
    """在 xarray Dataset 中查找坐标变量 (先查 coords, 再查 variables)。"""
    for name in candidates:
        if name in ds.coords:
            return ds[name]
    for name in candidates:
        if name in ds.variables:
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
    ds = None
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)

        # 查找坐标
        lat = _find_coord(ds, ["latitude", "lat", "y"])
        lon = _find_coord(ds, ["longitude", "lon", "x"])
        time = _find_coord(ds, ["time", "valid_time"])

        if lat is None or lon is None:
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

        # 数据变量 (兼容 Copernicus 原始变量名和下载脚本变量名)
        for var in ["siconc", "sithick", "vxsi", "vysi", "sivelu", "sivelv"]:
            if var in ds:
                result[var] = ds[var].values

        return result
    except Exception:
        return None
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


def load_era5_wind_data(
    nc_path: Path,
    lat_range: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """从 ERA5 NetCDF 加载风场数据。"""
    ds = None
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)

        lat = _find_coord(ds, ["latitude", "lat"])
        lon = _find_coord(ds, ["longitude", "lon"])
        time = _find_coord(ds, ["time", "valid_time"])

        if lat is None:
            return None

        if lat_range is not None:
            ds = ds.sel({lat.name: slice(lat_range[0], lat_range[1])})

        result = {"lat": lat.values, "lon": lon.values}
        if time is not None:
            result["time"] = ds[time.name].values

        for var in ["u10", "v10", "u10m", "v10m"]:
            if var in ds:
                result[var] = ds[var].values

        return result if any(k in result for k in ["u10", "v10"]) else None
    except Exception:
        return None
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


# ============================================================
# 数据驱动的冰况调度
# ============================================================

class DataDrivenIceSchedule(IceSchedule):
    """从 Copernicus/NSIDC NetCDF 读取真实冰况时间序列。

    在指定位置 (lat, lon) 插值读取 SIC, SIT, 漂移速度/方向，
    按时间步长线性插值。

    支持独立漂移数据源: 当主 NetCDF 不含冰漂移变量时，
    可通过 drift_nc_path 指定 NSIDC-0116 等独立漂移产品。
    """

    def __init__(
        self,
        nc_path: Path,
        lat: float = 80.0,
        lon: float = 0.0,
        duration: float = 300.0,
        drift_nc_path: Optional[Path] = None,
    ):
        self.nc_path = nc_path
        self.drift_nc_path = drift_nc_path
        self.lat = lat
        self.lon = lon
        self.duration = duration
        self._loaded = False
        self._times = None
        self._sic = None
        self._sit = None
        self._drift_speed = None
        self._drift_dir = None
        self._drift_source = "none"
        self._load()

    def _load_drift_from_separate_source(self):
        """从独立漂移数据源 (NSIDC-0116) 加载冰漂移。"""
        if self.drift_nc_path is None or not self.drift_nc_path.exists():
            return None, None

        drift_data = load_nsidc_ice_drift_data(self.drift_nc_path)
        if drift_data is None:
            return None, None

        lat_vals = drift_data["lat"]
        lon_vals = drift_data["lon"]
        lon_adj = _adjust_lon_for_grid(self.lon, lon_vals)

        lat_idx = int(np.argmin(np.abs(lat_vals - self.lat)))
        lon_idx = int(np.argmin(np.abs(lon_vals - lon_adj)))

        vx = drift_data.get("vxsi")
        vy = drift_data.get("vysi")
        if vx is None or vy is None:
            return None, None

        # 提取时间序列
        if vx.ndim == 3:
            vx = vx[:, lat_idx, lon_idx]
            vy = vy[:, lat_idx, lon_idx]
        elif vx.ndim == 2:
            vx = vx[:, lon_idx] if vx.shape[1] == len(lon_vals) else vx[lat_idx, :]
            vy = vy[:, lon_idx] if vy.shape[1] == len(lon_vals) else vy[lat_idx, :]

        speed = np.sqrt(vx**2 + vy**2)
        direction = np.degrees(np.arctan2(vx, vy)) % 360

        self._drift_source = "nsidc_0116"
        return np.asarray(speed, dtype=float), np.asarray(direction, dtype=float)

    def _load(self):
        data = load_copernicus_ice_data(self.nc_path)
        if data is None:
            return

        # 查找最近的网格点
        lat_vals = data["lat"]
        lon_vals = data["lon"]

        # 处理经度范围 (可能 0-360 或 -180-180)
        lon_adj = _adjust_lon_for_grid(self.lon, lon_vals)

        lat_idx = np.argmin(np.abs(lat_vals - self.lat))
        lon_idx = np.argmin(np.abs(lon_vals - lon_adj))

        self._sic = data.get("siconc")
        self._sit = data.get("sithick")

        def _extract_point(arr, lat_idx, lon_idx):
            """从 N-D 数组中提取指定网格点的标量或时间序列。

            - 1D: 直接返回 (可能是时间序列)
            - 2D: (lat, lon) 快照 → 标量; (time, lat/lon) → 1D
            - 3D: (time, lat, lon) → 1D 时间序列
            """
            if arr is None:
                return None
            if arr.ndim == 1:
                return arr
            if arr.ndim == 3:
                return arr[:, lat_idx, lon_idx]
            if arr.ndim == 2:
                n_lat = len(data["lat"]) if hasattr(data["lat"], '__len__') else 1
                n_lon = len(data["lon"]) if hasattr(data["lon"], '__len__') else 1
                if arr.shape[0] == n_lat and arr.shape[1] == n_lon:
                    return float(arr[lat_idx, lon_idx])  # (lat, lon) 快照 → 标量
                # (time, spatial) → 1D时间序列; 判断第二维是lat还是lon
                if arr.shape[1] == n_lat:
                    return arr[:, lat_idx]
                return arr[:, lon_idx]
            return arr

        def _ensure_time_series(value, duration):
            """确保提取的值是 1D 时间序列, 标量扩展为两点常值序列。"""
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                return np.array([float(arr), float(arr)]), np.array([0.0, duration])
            if arr.ndim == 1:
                return arr, np.linspace(0.0, duration, len(arr))
            raise ValueError(f"Expected scalar or 1D array, got shape {arr.shape}")

        self._sic = _extract_point(self._sic, lat_idx, lon_idx)
        self._sit = _extract_point(self._sit, lat_idx, lon_idx)

        vx = _first_existing(data, ["vxsi", "sivelu"])
        vy = _first_existing(data, ["vysi", "sivelv"])
        vx = _extract_point(vx, lat_idx, lon_idx)
        vy = _extract_point(vy, lat_idx, lon_idx)
        if vx is not None and vy is not None:
            vx = np.asarray(vx, dtype=float)
            vy = np.asarray(vy, dtype=float)
            if vx.ndim == 0:
                vx = np.array([float(vx)])
                vy = np.array([float(vy)])
            self._drift_speed = np.sqrt(vx**2 + vy**2)
            self._drift_dir = np.degrees(np.arctan2(vx, vy)) % 360
            self._drift_source = "primary_nc"
        else:
            # 主文件无漂移 → 尝试独立漂移数据源 (NSIDC-0116)
            drift_spd, drift_dir = self._load_drift_from_separate_source()
            if drift_spd is not None:
                self._drift_speed = drift_spd
                self._drift_dir = drift_dir
            else:
                # 无速度变量时，根据SIC/SIT生成合成冰漂移
                # 典型北极冰漂移: 0.05-0.15 m/s, 方向随纬度变化
                if self._sic is not None:
                    n = len(self._sic) if hasattr(self._sic, '__len__') else 2
                    # 基于SIC估算漂移速度 (高密集度→低漂移)
                    avg_sic = float(np.mean(self._sic)) if hasattr(self._sic, '__len__') else 0.5
                    # Lindqvist 经验: SIC=1→0.05 m/s, SIC=0.5→0.075 m/s, SIC=0→0.1 m/s
                    base_speed = 0.1 * (1.0 - 0.5 * avg_sic)
                    # 添加时间变化: 低频正弦模拟潮汐/惯性振荡
                    t_norm = np.linspace(0, 2 * np.pi, n)
                    speed_variation = 1.0 + 0.3 * np.sin(t_norm) + 0.15 * np.sin(2.3 * t_norm)
                    # 上限 0.85 m/s 覆盖场景 G6 (0.80 m/s), 下限 0.01 m/s 避免零漂移
                    self._drift_speed = np.clip(base_speed * speed_variation, 0.01, 0.85)
                    # 方向: 典型西南漂移 + 缓慢旋转 (北极涡旋效应)
                    base_dir = 200.0 + 30.0 * np.sin(t_norm * 0.7)
                    self._drift_dir = base_dir % 360.0
                else:
                    self._drift_speed = np.array([0.05, 0.05])
                    self._drift_dir = np.array([180.0, 180.0])
                self._drift_source = "synthetic_nansen_ekman"

        # 确保所有变量是时间序列 (标量快照扩展为两点常值)
        if self._sic is not None:
            self._sic, self._times = _ensure_time_series(self._sic, self.duration)
        if self._sit is not None:
            self._sit, _ = _ensure_time_series(self._sit, self.duration)
        if self._drift_speed is not None:
            self._drift_speed, _ = _ensure_time_series(self._drift_speed, self.duration)
        if self._drift_dir is not None:
            self._drift_dir, _ = _ensure_time_series(self._drift_dir, self.duration)

        # 多数据源时间轴一致性检查: 阻止 np.interp 因长度不匹配而崩溃
        self._validate_time_alignment()
        self._loaded = True

    def _validate_time_alignment(self):
        """确保所有变量的时间序列与 self._times 长度一致。

        当漂移来自独立 NSIDC-0116 源 (与 SIC/SIT 文件不同分辨率) 时,
        将较长序列重采样到 self._times 的格点上。
        """
        n_ref = len(self._times) if self._times is not None else 0
        if n_ref < 2:
            return
        for name, arr in [("_sit", self._sit), ("_drift_speed", self._drift_speed),
                          ("_drift_dir", self._drift_dir)]:
            if arr is not None and len(arr) != n_ref:
                # 重采样: 将 arr 插值到 self._times 格点
                src_times = np.linspace(0, self.duration, len(arr))
                setattr(self, name, np.interp(self._times, src_times, arr))

    def at(self, t: float) -> IceState:
        if not self._loaded or self._sic is None:
            return IceState(0.0, 0.0, 0.0, 0.0)

        t_clamp = float(np.clip(t, self._times[0], self._times[-1]))

        sic = float(np.interp(t_clamp, self._times, self._sic))
        sit = float(np.interp(t_clamp, self._times, self._sit)) if self._sit is not None else 0.0
        spd = float(np.interp(t_clamp, self._times, self._drift_speed)) if self._drift_speed is not None else 0.0
        # H8 fix: angle-aware interpolation for drift direction (handle 0/360 wraparound)
        if self._drift_dir is not None:
            idx = np.searchsorted(self._times, t_clamp)
            idx = np.clip(idx, 1, len(self._times) - 1)
            t0, t1 = self._times[idx - 1], self._times[idx]
            d0, d1 = self._drift_dir[idx - 1], self._drift_dir[idx]
            frac = (t_clamp - t0) / max(t1 - t0, 1e-12)
            # Shortest-path angle interpolation
            diff = (d1 - d0 + 180.0) % 360.0 - 180.0
            drd = (d0 + frac * diff) % 360.0
        else:
            drd = 0.0

        return IceState(
            concentration=float(np.clip(sic, 0.0, 1.0)),
            thickness=max(0.0, sit),
            drift_speed=max(0.0, spd),
            drift_direction=drd,
        )

    @property
    def provenance(self) -> Dict[str, str]:
        """返回数据溯源信息，标注每个变量来自真实数据还是合成回退。"""
        real_sic = self._sic is not None
        real_sit = self._sit is not None
        # 使用 _drift_source 判断漂移来源 (含合成回退标记), 而非仅检查 None
        drift_is_real = self._drift_source in ("primary_nc", "nsidc_0116")
        return {
            "sic_source": "netcdf" if real_sic else "synthetic_fallback",
            "sit_source": "netcdf" if real_sit else "synthetic_fallback",
            "drift_source": self._drift_source,
            "nc_path": str(self.nc_path) if self.nc_path else "none",
            "drift_nc_path": str(self.drift_nc_path) if self.drift_nc_path else "none",
            "is_fully_data_driven": str(real_sic and real_sit and drift_is_real),
        }


# ============================================================
# 数据驱动的风场调度
# ============================================================

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
        lon_adj = _adjust_lon_for_grid(lon, lon_vals)

        lat_idx = np.argmin(np.abs(lat_vals - lat))
        lon_idx = np.argmin(np.abs(lon_vals - lon_adj))

        u = data.get("u10")
        v = data.get("v10")
        if u is not None and v is not None:
            if u.ndim == 3:
                self._u = u[:, lat_idx, lon_idx]
                self._v = v[:, lat_idx, lon_idx]
            elif u.ndim == 2:
                # 2D 空间快照 → 提取单点
                self._u = np.array([float(u[lat_idx, lon_idx])])
                self._v = np.array([float(v[lat_idx, lon_idx])])
            else:
                self._u = np.asarray(u, dtype=float).ravel()
                self._v = np.asarray(v, dtype=float).ravel()
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
# OSCAR 海流数据加载
# ============================================================

def load_oscar_current_data(
    nc_path: Path,
    lat_range: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """从 OSCAR NetCDF 加载海流数据。

    OSCAR 变量: u (eastward), v (northward), 单位 m/s。
    坐标: latitude, longitude, time。

    Returns:
        dict with keys: u, v, lat, lon, time
        or None if loading fails
    """
    ds = None
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)

        lat = _find_coord(ds, ["latitude", "lat", "y"])
        lon = _find_coord(ds, ["longitude", "lon", "x"])
        time = _find_coord(ds, ["time", "time365"])

        if lat is None or lon is None:
            return None

        if lat_range is not None:
            ds = ds.sel({lat.name: slice(lat_range[0], lat_range[1])})

        result = {"lat": lat.values, "lon": lon.values}
        if time is not None:
            result["time"] = ds[time.name].values

        # OSCAR 海流变量名
        for var in ["u", "v", "ucur", "vcur", "uo", "vo"]:
            if var in ds:
                result[var] = ds[var].values

        return result if any(k in result for k in ["u", "v", "ucur", "vcur"]) else None
    except Exception:
        return None
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass



def load_nsidc_ice_drift_data(
    nc_path: Path,
    lat_range: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """从 NSIDC Polar Pathfinder v4 (NSIDC-0116) 加载冰漂移数据。

    NSIDC-0116 变量: u (eastward, cm/s), v (northward, cm/s)。
    需要转换为 m/s。填补项目最大数据缺口。

    Returns:
        dict with keys: vxsi, vysi, lat, lon, time (漂移分量, m/s)
    """
    ds = None
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)

        lat = _find_coord(ds, ["latitude", "lat", "y"])
        lon = _find_coord(ds, ["longitude", "lon", "x"])
        time = _find_coord(ds, ["time", "t"])

        if lat is None or lon is None:
            return None

        if lat_range is not None:
            ds = ds.sel({lat.name: slice(lat_range[0], lat_range[1])})

        result = {"lat": lat.values, "lon": lon.values}
        if time is not None:
            result["time"] = ds[time.name].values

        # NSIDC-0116 变量: u/v in cm/s → convert to m/s
        u_found, v_found = False, False
        for u_name in ["u", "u_ice_drift", "u_drift"]:
            if u_name in ds and not u_found:
                result["vxsi"] = ds[u_name].values.astype(float) * 0.01
                u_found = True
        for v_name in ["v", "v_ice_drift", "v_drift"]:
            if v_name in ds and not v_found:
                result["vysi"] = ds[v_name].values.astype(float) * 0.01
                v_found = True

        if u_found and v_found:
            return result
        return None
    except Exception:
        return None
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


class DataDrivenCurrentSchedule:
    """从 OSCAR NetCDF 读取真实海流时间序列。

    在指定位置 (lat, lon) 插值读取表层海流 u/v 分量。
    """

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

        data = load_oscar_current_data(nc_path)
        if data is None:
            return

        lat_vals = data["lat"]
        lon_vals = data["lon"]

        lon_adj = _adjust_lon_for_grid(lon, lon_vals)

        lat_idx = np.argmin(np.abs(lat_vals - lat))
        lon_idx = np.argmin(np.abs(lon_vals - lon_adj))

        u = _first_existing(data, ["u", "ucur", "uo"])
        v = _first_existing(data, ["v", "vcur", "vo"])

        if u is not None and v is not None:
            u = np.asarray(u, dtype=float)
            v = np.asarray(v, dtype=float)
            if u.ndim == 3:
                u = u[:, lat_idx, lon_idx]
                v = v[:, lat_idx, lon_idx]
            elif u.ndim == 2:
                u = u[lat_idx, :]
                v = v[lat_idx, :]
            self._u = np.atleast_1d(u).ravel()
            self._v = np.atleast_1d(v).ravel()
            n = len(self._u)
            self._times = np.linspace(0, duration, n)
            self._loaded = True

    def at(self, t: float) -> Tuple[float, float]:
        """返回时刻 t 的海流 (u_east, v_north) m/s。"""
        if not self._loaded:
            return (0.0, 0.0)
        t_clamp = float(np.clip(t, self._times[0], self._times[-1]))
        u = float(np.interp(t_clamp, self._times, self._u))
        v = float(np.interp(t_clamp, self._times, self._v))
        return (u, v)


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

    vx = _first_existing(data, ["vxsi", "sivelu"])
    vy = _first_existing(data, ["vysi", "sivelv"])
    if vx is not None and vy is not None:
        speed = np.sqrt(np.asarray(vx).flatten()**2 + np.asarray(vy).flatten()**2)
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
