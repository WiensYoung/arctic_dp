#!/usr/bin/env python3
"""为 2020 mock Copernicus 数据生成合成冰速度变量 (vxsi/vysi)。

基于 SIC 估算冰漂移:
- 高密集度 → 低漂移 (consolidated ice)
- 低密集度 → 高漂移 (fragmented ice)
- 方向: 典型西南漂移 + 惯性振荡

生成文件: copernicus/arctic_ice_2020_jan1_7_with_velocity.nc
"""
import numpy as np
from pathlib import Path

try:
    import xarray as xr
except ImportError:
    print("需要 xarray: pip install xarray netCDF4")
    exit(1)

SRC = Path(__file__).parent / "copernicus" / "arctic_ice_2020_jan1_7.nc"
DST = Path(__file__).parent / "copernicus" / "arctic_ice_2020_jan1_7_with_velocity.nc"

print(f"读取: {SRC}")
ds = xr.open_dataset(SRC)

sic = ds["siconc"].values  # (time, lat, lon)
sit = ds["sithick"].values
n_time, n_lat, n_lon = sic.shape
print(f"维度: time={n_time}, lat={n_lat}, lon={n_lon}")
print(f"SIC 均值: {np.nanmean(sic):.3f}, SIT 均值: {np.nanmean(sit):.3f} m")

# 生成合成冰速度
# 基于 SIC: 高密集度 → 低漂移
rng = np.random.default_rng(2020)
base_speed = 0.1 * (1.0 - 0.5 * sic)  # SIC=1→0.05, SIC=0.5→0.075, SIC=0→0.1

# 时间变化: 惯性振荡 (约12h周期)
t_norm = np.linspace(0, 2 * np.pi, n_time)
speed_mod = np.zeros_like(sic)
for i in range(n_time):
    speed_mod[i] = 1.0 + 0.3 * np.sin(t_norm[i]) + 0.15 * np.sin(2.3 * t_norm[i])

speed = np.clip(base_speed * speed_mod, 0.01, 0.3)

# 空间噪声: 模拟湍流
noise_u = rng.normal(0, 0.02, sic.shape)
noise_v = rng.normal(0, 0.02, sic.shape)

# 方向: 典型西南漂移 (200°) + 缓慢旋转
base_dir_deg = 200.0 + 30.0 * np.sin(t_norm * 0.7)
base_dir_rad = np.radians(base_dir_deg)

# vxsi = eastward, vysi = northward
vxsi = np.zeros_like(sic)
vysi = np.zeros_like(sic)
for i in range(n_time):
    vxsi[i] = speed[i] * np.sin(base_dir_rad[i]) + noise_u[i]
    vysi[i] = speed[i] * np.cos(base_dir_rad[i]) + noise_v[i]

# 添加到数据集
ds["vxsi"] = xr.DataArray(
    vxsi.astype(np.float32),
    dims=ds["siconc"].dims,
    attrs={
        "long_name": "sea_ice_x_velocity",
        "units": "m s-1",
        "standard_name": "sea_ice_x_velocity",
        "note": "Synthetic eastward ice velocity based on SIC. For proxy-scale experiments only.",
    },
)
ds["vysi"] = xr.DataArray(
    vysi.astype(np.float32),
    dims=ds["siconc"].dims,
    attrs={
        "long_name": "sea_ice_y_velocity",
        "units": "m s-1",
        "standard_name": "sea_ice_y_velocity",
        "note": "Synthetic northward ice velocity based on SIC. For proxy-scale experiments only.",
    },
)

# 保存
ds.to_netcdf(DST)
ds.close()
print(f"\n已生成: {DST}")
print(f"变量: siconc, sithick, vxsi, vysi")
print(f"vxsi 范围: [{np.nanmin(vxsi):.4f}, {np.nanmax(vxsi):.4f}] m/s")
print(f"vysi 范围: [{np.nanmin(vysi):.4f}, {np.nanmax(vysi):.4f}] m/s")

# 验证
ds2 = xr.open_dataset(DST)
print(f"\n验证:")
for v in ds2.data_vars:
    print(f"  {v}: shape={ds2[v].shape}, mean={float(ds2[v].mean()):.4f}")
ds2.close()
print("完成 ✓")
