"""下载缺失的关键数据集。

用法:
    # 下载 NSIDC Polar Pathfinder v4 冰漂移数据
    python scripts/download_data.py --dataset nsidc_0116

    # 下载 ICESat-2 ATL10 冰厚/干舷数据
    python scripts/download_data.py --dataset icesat2_atl10

    # 下载全部缺失数据集
    python scripts/download_data.py --all

前置条件:
    - earthaccess 已安装 (pip install earthaccess)
    - Earthdata Login 账号 (https://urs.earthdata.nasa.gov/)
    - 首次运行需要 .netrc 认证或 earthaccess.login() 交互式登录
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "sci1_sources"


def download_nsidc_0116_ice_drift():
    """下载 NSIDC Polar Pathfinder v4 冰漂移数据。

    NSIDC-0116: Daily ice motion vectors, 25km, 1978-present.
    变量: u (eastward, cm/s), v (northward, cm/s)

    注意: NSIDC-0116 不在 NASA Earthdata Cloud 中。
    需要通过 NSIDC HTTPS 直接下载，或使用 .netrc 认证。
    """
    import requests
    from netrc import netrc

    output_dir = DATA_ROOT / "nsidc_0116_ice_drift"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 尝试方法 1: NSIDC HTTPS 直接下载 (需 .netrc)
    print("Trying NSIDC HTTPS direct download...")
    try:
        secrets = netrc(Path.home() / ".netrc")
        auth_info = secrets.authenticators("urs.earthdata.nasa.gov")
        if auth_info:
            user, _, passwd = auth_info
            # NSIDC-0116 v4 数据 URL
            base = "https://noaadata.apps.nsidc.org/NOAA/NSIDC-0116/004/2020/"
            r = requests.get(base, auth=(user, passwd), timeout=30)
            if r.status_code == 200:
                import re
                files = re.findall(r'href="([^"]+\\.nc)"', r.text)
                print(f"Found {len(files)} .nc files for 2020")
                for fname in files[:7]:  # First 7 days of Jan
                    url = base + fname
                    print(f"  Downloading {fname}...")
                    fr = requests.get(url, auth=(user, passwd), timeout=120)
                    if fr.status_code == 200:
                        (output_dir / fname).write_bytes(fr.content)
                        print(f"    OK ({len(fr.content)} bytes)")
                    else:
                        print(f"    Failed: HTTP {fr.status_code}")
                return
            else:
                print(f"  HTTP {r.status_code} - server may not be reachable from this network")
    except Exception as e:
        print(f"  HTTPS download failed: {e}")

    # 尝试方法 2: earthaccess (可能不支持)
    print("\nTrying earthaccess (may not work for this dataset)...")
    try:
        import earthaccess
        earthaccess.login(strategy="netrc")
        results = earthaccess.search_data(
            short_name="NSIDC-0116",
            version="004",
            temporal=("2020-01-01", "2020-01-07"),
            count=7,
        )
        if results:
            earthaccess.download(results, str(output_dir))
            print(f"Downloaded {len(results)} granules via earthaccess")
            return
        else:
            print("  0 granules (NSIDC-0116 not in Earthdata Cloud)")
    except Exception as e:
        print(f"  earthaccess failed: {e}")

    # 回退: 手动下载说明
    print(f"""
    ═══════════════════════════════════════════════════════════════
    NSIDC-0116 自动下载失败 (服务器不可达或不在 Cloud 中)。

    手动下载步骤:
      1. 浏览器访问: https://noaadata.apps.nsidc.org/NOAA/NSIDC-0116/004/
      2. 进入 2020/ 文件夹
      3. 下载 1月1-7日的 .nc 文件
      4. 放入: {output_dir}

    ⚠ 注意: Copernicus 2024 数据 (arctic_ice_2024_jan1_7_real.nc)
       已包含 ~4km 分辨率冰漂移 (vxsi/vysi), 精度高于 NSIDC-0116 (25km)。
       当前实验可直接使用 Copernicus 漂移数据, NSIDC-0116 仅需用于
       长期 (多年) 漂移气候态分析。
    ═══════════════════════════════════════════════════════════════
    """)


def download_icesat2_atl10():
    """下载 ICESat-2 ATL10 海冰干舷数据 (独立冰厚验证)。

    ATL10: Sea Ice Freeboard, 沿轨 ~40m 分辨率, 2018-present.
    """
    import earthaccess

    output_dir = DATA_ROOT / "icesat2_atl10"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Authenticating with Earthdata...")
    earthaccess.login()

    print("Searching ICESat-2 ATL10 v006, 2020 Jan, Arctic...")
    results = earthaccess.search_data(
        short_name="ATL10",
        version="006",
        temporal=("2020-01-01", "2020-01-31"),
        bounding_box=(75, -180, 85, 180),
        count=10,
    )
    print(f"Found {len(results)} granules")

    if results:
        earthaccess.download(results, str(output_dir))
        print(f"Downloaded to {output_dir}")

        _update_manifest_status("ICESat-2 ATL10", "downloaded", str(output_dir.relative_to(PROJECT_ROOT)))
    else:
        print("No results found. Note: ATL10 data may be seasonal (winter-only).")


def _update_manifest_status(name_keyword: str, status: str, local_path: str):
    """更新 data_manifest.json 中匹配名称的源的下载状态。"""
    import json

    manifest_path = DATA_ROOT / "data_manifest.json"
    if not manifest_path.exists():
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for src in manifest["sources"]:
        if name_keyword.lower() in src.get("name", "").lower():
            src["status"] = status
            src["local_path"] = local_path
            print(f"  Updated manifest: {src['name']} → {status}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Download missing critical datasets")
    parser.add_argument("--dataset", choices=["nsidc_0116", "icesat2_atl10"],
                        help="Specific dataset to download")
    parser.add_argument("--all", action="store_true",
                        help="Download all missing datasets")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        print("\nAvailable datasets:")
        print("  nsidc_0116    - NSIDC Polar Pathfinder v4 ice drift (P0: critical gap)")
        print("  icesat2_atl10 - ICESat-2 ATL10 sea ice freeboard (P1: thickness validation)")
        return

    try:
        if args.all or args.dataset == "nsidc_0116":
            print("\n=== Downloading NSIDC-0116 Ice Drift ===\n")
            download_nsidc_0116_ice_drift()

        if args.all or args.dataset == "icesat2_atl10":
            print("\n=== Downloading ICESat-2 ATL10 ===\n")
            download_icesat2_atl10()

        print("\nDone.")
    except ImportError:
        print("ERROR: earthaccess not installed. Run: pip install earthaccess",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
