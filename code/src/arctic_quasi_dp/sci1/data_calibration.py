"""从已下载的真实数据校准场景参数。

使用 NSIDC Sea Ice Index CSV 数据校准场景的冰密集度参数。
将文献校准参数与真实观测数据结合，提升实验的可信度。

使用：
    from arctic_quasi_dp.sci1.data_calibration import calibrate_from_nsidc, get_arctic_sic_stats
    stats = get_arctic_sic_stats(Path("data/sci1_sources/nsidc_sea_ice_index"))
    print(stats)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv
import math

import numpy as np


@dataclass
class SICStats:
    """海冰密集度统计。"""
    month: int
    mean_extent: float       # 百万 km²
    std_extent: float
    min_extent: float
    max_extent: float
    mean_area: float
    trend_per_decade: float  # 百万 km² / 十年
    n_years: int


def load_nsidc_csv(csv_path: Path) -> List[Dict[str, float]]:
    """加载单个 NSIDC Sea Ice Index CSV 文件。过滤 -9999 缺失值。"""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                extent = float(row[" extent"])
                area = float(row["   area"])
                # 过滤缺失值 (-9999) 和异常值
                if extent < 0 or area < 0:
                    continue
                rows.append({
                    "year": int(row["year"]),
                    "month": int(row[" mo"]),
                    "extent": extent,
                    "area": area,
                })
            except (ValueError, KeyError):
                continue
    return rows


def get_arctic_sic_stats(data_dir: Path) -> Dict[int, SICStats]:
    """计算北极每月海冰范围统计。"""
    stats = {}
    for m in range(1, 13):
        csv_path = data_dir / f"N_{m:02d}_extent_v4.0.csv"
        if not csv_path.exists():
            continue
        rows = load_nsidc_csv(csv_path)
        if not rows:
            continue

        extents = [r["extent"] for r in rows]
        areas = [r["area"] for r in rows]
        years = [r["year"] for r in rows]

        # 计算趋势 (线性回归)
        n = len(extents)
        if n > 2:
            x = np.array(years, dtype=float)
            y = np.array(extents, dtype=float)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            trend_per_decade = slope * 10.0
        else:
            trend_per_decade = 0.0

        stats[m] = SICStats(
            month=m,
            mean_extent=float(np.mean(extents)),
            # M9 fix: use sample std (ddof=1) for finite year samples
            std_extent=float(np.std(extents, ddof=1)) if n > 1 else 0.0,
            min_extent=float(np.min(extents)),
            max_extent=float(np.max(extents)),
            mean_area=float(np.mean(areas)),
            trend_per_decade=trend_per_decade,
            n_years=n,
        )
    return stats


def get_typical_sic_for_month(data_dir: Path, month: int) -> Dict[str, float]:
    """获取指定月份的典型 SIC 统计值。"""
    stats = get_arctic_sic_stats(data_dir)
    if month not in stats:
        # M7 fix: use consistent key names matching the data-exists path
        return {
            "mean_extent_mkm2": 7.25,
            "std_extent_mkm2": 0.72,
            "mean_concentration_approx": 0.5,
            "trend_per_decade": 0.0,
            "n_years": 0,
        }
    s = stats[month]
    # 将 extent (百万km²) 转换为典型密集度估计
    # 北极总面积约 14.5 百万 km²
    arctic_area = 14.5
    return {
        "mean_extent_mkm2": s.mean_extent,
        "std_extent_mkm2": s.std_extent,
        "mean_concentration_approx": min(1.0, s.mean_area / arctic_area),
        "trend_per_decade": s.trend_per_decade,
        "n_years": s.n_years,
    }


def calibrate_scenario_from_nsidc(
    data_dir: Path,
    target_month: int = 9,
) -> Dict[str, float]:
    """根据真实 NSIDC 数据校准场景参数。

    Args:
        data_dir: NSIDC CSV 数据目录
        target_month: 目标月份 (9=北极最小, 3=最大)

    Returns:
        校准后的冰况参数字典
    """
    stats = get_arctic_sic_stats(data_dir)
    if target_month not in stats:
        return {}

    s = stats[target_month]
    arctic_area = 14.5  # 北极盆地面积 (百万 km²)

    # 密集度: 使用 area/total 作为标称值，std/mean 作为变异系数
    concentration_nominal = min(1.0, s.mean_area / arctic_area)
    concentration_cv = s.std_extent / max(s.mean_extent, 0.01)

    return {
        "ice_concentration": concentration_nominal,
        "ice_concentration_std": concentration_nominal * concentration_cv,
        "ice_extent_mean_mkm2": s.mean_extent,
        "ice_extent_trend_mkm2_per_decade": s.trend_per_decade,
        "source": f"NSIDC Sea Ice Index v4.0, month={target_month}, {s.n_years} years",
    }


def print_calibration_report(data_dir: Path) -> None:
    """打印校准报告。"""
    stats = get_arctic_sic_stats(data_dir)
    if not stats:
        print("No data found in", data_dir)
        return

    print("NSIDC Sea Ice Index Calibration Report")
    print("=" * 70)
    print(f"{'Month':>5}  {'Mean(Mkm2)':>10}  {'Std':>8}  {'Trend/10yr':>10}  {'Years':>5}")
    print("-" * 70)
    for m in sorted(stats.keys()):
        s = stats[m]
        print(f"{m:>5}  {s.mean_extent:>10.2f}  {s.std_extent:>8.2f}  {s.trend_per_decade:>+10.3f}  {s.n_years:>5}")

    sept = stats.get(9)
    march = stats.get(3)
    if sept and march:
        print()
        print("Key findings:")
        print(f"  Arctic Sep mean: {sept.mean_extent:.2f} Mkm2 (trend {sept.trend_per_decade:+.3f}/10yr)")
        print(f"  Arctic Mar mean: {march.mean_extent:.2f} Mkm2 (trend {march.trend_per_decade:+.3f}/10yr)")
        print(f"  Seasonal amplitude: {march.mean_extent - sept.mean_extent:.2f} Mkm2")
