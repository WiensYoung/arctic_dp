"""一键下载所有可用数据源。

运行方式：
  python scripts/download_data.py
  python scripts/download_data.py --sources nsidc copernicus era5
  python scripts/download_data.py --year 2020 --region arctic

自动检测已配置的凭据，跳过无法认证的数据源。
下载失败不阻塞其他数据源。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arctic_quasi_dp.sci1.data_sources import (
    AUTHORITATIVE_SOURCES,
    LITERATURE_CALIBRATIONS,
    write_manifest,
    update_source_status,
)


OUTPUT_DIR = Path("data/sci1_sources")


def check_earthdata() -> bool:
    """检查 Earthdata 凭据是否可用。"""
    netrc = Path.home() / ".netrc"
    if not netrc.exists():
        return False
    content = netrc.read_text()
    return "urs.earthdata.nasa.gov" in content and "login" in content


def check_cds() -> bool:
    """检查 CDS API key 是否可用。"""
    cdsapirc = Path.home() / ".cdsapirc"
    return cdsapirc.exists()


def check_copernicus_marine() -> bool:
    """检查 Copernicus Marine 凭据是否可用。"""
    config = Path.home() / ".copernicusmarine" / "copernicusmarine-datastore-credentials.ini"
    return config.exists()


def download_nsidc_cdr(year: int = 2020) -> bool:
    """下载 NSIDC CDR 海冰密集度。"""
    print(f"\n[NSIDC CDR] 下载 {year} 年 SIC 数据...")
    try:
        import earthaccess
        auth = earthaccess.login()

        granules = earthaccess.search_data(
            short_name="NSIDC-0079",
            version="v4",
            temporal=(f"{year}-01-01", f"{year}-12-31"),
            bounding_box=(-180, 60, 180, 90),
        )
        if not granules:
            print("  未找到数据")
            return False

        print(f"  找到 {len(granules)} 个文件")
        out = OUTPUT_DIR / "nsidc_cdr_sic" / str(year)
        out.mkdir(parents=True, exist_ok=True)
        files = earthaccess.download(granules[:10], str(out))  # 限制前10个测试
        print(f"  下载了 {len(files)} 个文件到 {out}")
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        return False


def download_nsidc_ice_drift(year: int = 2020) -> bool:
    """下载 NSIDC Polar Pathfinder 海冰漂移。"""
    print(f"\n[NSIDC Polar Pathfinder] 下载 {year} 年海冰漂移数据...")
    try:
        import earthaccess
        auth = earthaccess.login()

        granules = earthaccess.search_data(
            short_name="NSIDC-0116",
            version="v4",
            temporal=(f"{year}-01-01", f"{year}-12-31"),
        )
        if not granules:
            print("  未找到数据")
            return False

        print(f"  找到 {len(granules)} 个文件")
        out = OUTPUT_DIR / "nsidc_ice_drift" / str(year)
        out.mkdir(parents=True, exist_ok=True)
        files = earthaccess.download(granules[:5], str(out))
        print(f"  下载了 {len(files)} 个文件到 {out}")
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        return False


def download_era5(year: int = 2020) -> bool:
    """下载 ERA5 北极再分析数据。"""
    print(f"\n[ERA5] 下载 {year} 年北极再分析数据...")
    if not check_cds():
        print("  CDS API key 未配置，跳过")
        return False

    try:
        import cdsapi
        c = cdsapi.Client()

        out = OUTPUT_DIR / "era5"
        out.mkdir(parents=True, exist_ok=True)
        outfile = out / f"era5_arctic_{year}.nc"

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "mean_sea_level_pressure",
                    "sea_ice_cover",
                    "sea_surface_temperature",
                ],
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in [1, 15]],  # 仅每月1日和15日
                "time": ["00:00", "12:00"],
                "area": [90, -180, 60, 180],
                "format": "netcdf",
            },
            str(outfile),
        )
        print(f"  下载完成: {outfile}")
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        return False


def download_copernicus_ice(year: int = 2020) -> bool:
    """下载 Copernicus Marine 海冰再分析。"""
    print(f"\n[Copernicus Marine] 下载 {year} 年海冰再分析数据...")
    if not check_copernicus_marine():
        print("  Copernicus Marine 凭据未配置，跳过")
        return False

    try:
        import copernicusmarine

        out = OUTPUT_DIR / "copernicus_ice"
        out.mkdir(parents=True, exist_ok=True)

        copernicusmarine.subset(
            dataset_id="ARCTIC_MULTIYEAR_PHY_ICE_002_016",
            variables=["siconc", "sithick", "sivelu", "sivelv"],
            start_datetime=f"{year}-01-01",
            end_datetime=f"{year}-03-31",  # 仅Q1测试
            minimum_longitude=-180,
            maximum_longitude=180,
            minimum_latitude=60,
            maximum_latitude=90,
            output_filename=str(out / f"arctic_ice_{year}_q1.nc"),
        )
        print(f"  下载完成: {out}")
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        return False


def download_piomas() -> bool:
    """下载 PIOMAS 海冰厚度数据。"""
    print(f"\n[PIOMAS] 下载海冰厚度数据...")
    try:
        import urllib.request

        out = OUTPUT_DIR / "piomas"
        out.mkdir(parents=True, exist_ok=True)

        # PIOMAS 月均数据
        url = "https://psc.apl.uw.edu/wordpress/wp-content/uploads/schweiger/ice_volume/PIOMAS.2sst.monthly.Current.v2.1.dat"
        outfile = out / "PIOMAS_monthly.dat"

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research"})
        resp = urllib.request.urlopen(req, timeout=60)
        data = resp.read()
        outfile.write_bytes(data)
        print(f"  下载完成: {outfile} ({len(data)} bytes)")
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        print("  PIOMAS 数据可从 https://psc.apl.uw.edu/research/projects/piomas/ 手动下载")
        return False


def write_data_manifest(year: int = 2020, results: dict = None):
    """更新 data_manifest.json，记录下载状态。"""
    manifest_path = OUTPUT_DIR / "data_manifest.json"

    # 自定义 sources 列表，标记每个的下载状态
    from arctic_quasi_dp.sci1.data_sources import DataSource
    sources = list(AUTHORITATIVE_SOURCES)

    # 根据下载结果更新状态
    status_map = {
        "nsidc_cdr": "downloaded" if results and results.get("nsidc_cdr") else "not_downloaded",
        "nsidc_drift": "downloaded" if results and results.get("nsidc_drift") else "not_downloaded",
        "era5": "downloaded" if results and results.get("era5") else "not_downloaded",
        "copernicus": "downloaded" if results and results.get("copernicus") else "not_downloaded",
        "piomas": "downloaded" if results and results.get("piomas") else "not_downloaded",
    }

    write_manifest(manifest_path, sources, include_literature=True)
    print(f"\n已更新 manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="下载 SCI1 实验数据源")
    parser.add_argument("--year", type=int, default=2020, help="数据年份 (默认 2020)")
    parser.add_argument("--sources", nargs="+", default=["all"],
                        choices=["all", "nsidc", "era5", "copernicus", "piomas"],
                        help="要下载的数据源")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="输出目录")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = set(args.sources)
    download_all = "all" in sources

    print("=" * 60)
    print("SCI1 数据源下载")
    print(f"年份: {args.year}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    # 检查凭据
    print("\n凭据检查:")
    print(f"  NASA Earthdata: {'✓' if check_earthdata() else '✗ 未配置'}")
    print(f"  CDS API:        {'✓' if check_cds() else '✗ 未配置'}")
    print(f"  Copernicus M:   {'✓' if check_copernicus_marine() else '✗ 未配置'}")

    results = {}

    # NSIDC
    if download_all or "nsidc" in sources:
        if check_earthdata():
            results["nsidc_cdr"] = download_nsidc_cdr(args.year)
            results["nsidc_drift"] = download_nsidc_ice_drift(args.year)
        else:
            print("\n[NSIDC] 跳过 — 需要先运行: python scripts/setup_credentials.py")

    # ERA5
    if download_all or "era5" in sources:
        results["era5"] = download_era5(args.year)

    # Copernicus Marine
    if download_all or "copernicus" in sources:
        results["copernicus"] = download_copernicus_ice(args.year)

    # PIOMAS (免费)
    if download_all or "piomas" in sources:
        results["piomas"] = download_piomas()

    # 更新 manifest
    write_data_manifest(args.year, results)

    # 汇总
    print("\n" + "=" * 60)
    print("下载结果汇总:")
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败/跳过"
        print(f"  {name}: {status}")
    n_ok = sum(1 for v in results.values() if v)
    print(f"\n总计: {n_ok}/{len(results)} 成功")
    print("=" * 60)


if __name__ == "__main__":
    main()
