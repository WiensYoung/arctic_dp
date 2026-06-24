"""权威公开数据源 registry 与数据可用性说明。

本模块不强制在线下载数据；投稿复现实验应使用 data_manifest.json 锁定数据版本。
在受限环境中无法访问 NASA Earthdata、Copernicus Marine 或 NOAA OPeNDAP 时，
代码会使用文献校准参数生成 scenario priors，并在 provenance 中明确标记。

数据源覆盖：
- 海冰密集度 (SIC): NSIDC CDR, NSIDC-0051, OSI-450
- 海冰厚度 (SIT): CryoSat-2, ICESat-2, PIOMAS, Copernicus reanalysis
- 海冰漂移 (SID): NSIDC Polar Pathfinder, IFREMER CERSAT
- 海流: OSCAR, GlobCurrent, Mercator Ocean
- 大气强迫: ERA5, JRA-55
- 实测校准: Ice Tank (HSVA/Aalto), 现场测量 (N-ICE2015, MOSAiC)
- 冰图: NIC, MASIE
- 船舶轨迹: AIS (用于场景验证)

新增:
- 下载工具函数 (download_ice_data, download_copernicus_ice)
- 数据加载辅助函数 (load_sic_from_netcdf, load_ice_thickness_from_netcdf)
- 文献校准参数库 (Lindqvist 1989, ISO 19906, Riska 1997)
- manifest 增加 checksum 和版本追踪
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import hashlib
import json
import warnings

import numpy as np


@dataclass(frozen=True)
class DataSource:
    """单个权威数据源。"""
    name: str
    variable: str
    authority: str
    url: str
    doi_or_product_id: str
    spatial_resolution: str
    temporal_resolution: str
    access_note: str
    intended_use: str
    status: str = "not_downloaded"
    local_path: Optional[str] = None
    checksum_sha256: Optional[str] = None
    download_date: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# ============================================================
# 1. 海冰密集度 (Sea Ice Concentration, SIC)
# ============================================================

SIC_SOURCES: List[DataSource] = [
    DataSource(
        name="NOAA/NSIDC CDR Passive Microwave Sea Ice Concentration v4",
        variable="sea_ice_concentration",
        authority="NOAA/NASA NSIDC DAAC",
        url="https://nsidc.org/data/g02202/versions/4",
        doi_or_product_id="G02202 v4",
        spatial_resolution="25 km polar stereographic grid",
        temporal_resolution="daily and monthly, 1978-present",
        access_note="Requires Earthdata/NSIDC download workflow; cache raw NetCDF outside git.",
        intended_use="Regional SIC priors and Monte Carlo concentration distributions.",
    ),
    DataSource(
        name="NSIDC Sea Ice Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS v2",
        variable="sea_ice_concentration",
        authority="NASA NSIDC DAAC",
        url="https://nsidc.org/data/nsidc-0051/versions/2",
        doi_or_product_id="NSIDC-0051 v2",
        spatial_resolution="25 km polar stereographic grid",
        temporal_resolution="daily/monthly passive microwave record",
        access_note="Use as cross-product uncertainty check against NOAA/NSIDC CDR.",
        intended_use="Uncertainty envelope and sensitivity bounds for SIC.",
    ),
    DataSource(
        name="OSI SAF Global Sea Ice Concentration Climate Data Record v3",
        variable="sea_ice_concentration",
        authority="EUMETSAT OSI SAF / Norwegian Meteorological Institute",
        url="https://osi-saf.eumetsat.int/products/osi-450",
        doi_or_product_id="10.15770/EUM_SAF_OSI_0013",
        spatial_resolution="25 km EASE2 grid",
        temporal_resolution="daily, 1978-present",
        access_note="Free registration at EUMETSAT. Alternative to NSIDC for cross-validation.",
        intended_use="Cross-validation of SIC; European Arctic focus.",
    ),
]

# ============================================================
# 2. 海冰厚度 (Sea Ice Thickness, SIT)
# ============================================================

SIT_SOURCES: List[DataSource] = [
    DataSource(
        name="CryoSat-2 Sea Ice Thickness L3",
        variable="sea_ice_thickness",
        authority="ESA / CPOM / AWI",
        url="https://data.eumetsat.int/product/EO:EUM:DAT:CRYOSAT:SI",
        doi_or_product_id="10.5270/CRYOSAT-2",
        spatial_resolution="satellite radar altimetry, ~1.5 km along-track",
        temporal_resolution="monthly, 2010-present (winter only, Oct-Apr)",
        access_note="Free via EUMETSAT Data Store. Covers Arctic winter only.",
        intended_use="Ice thickness calibration for Lindqvist model and scenario priors.",
    ),
    DataSource(
        name="ICESat-2 ATL10 Sea Ice Freeboard",
        variable="sea_ice_freeboard",
        authority="NASA NSIDC DAAC",
        url="https://nsidc.org/data/atl10",
        doi_or_product_id="10.5067/ICESAT2/ATL10",
        spatial_resolution="along-track ~17 m footprint",
        temporal_resolution="2018-present, continuous",
        access_note="Requires Earthdata account. Freeboard → thickness conversion needed.",
        intended_use="High-resolution thickness validation and freeboard-to-thickness calibration.",
    ),
    DataSource(
        name="PIOMAS Pan-Arctic Ice Ocean Modeling and Assimilation System",
        variable="sea_ice_thickness, ice_velocity",
        authority="University of Washington / Polar Science Center",
        url="https://psc.apl.uw.edu/research/projects/piomas/",
        doi_or_product_id="Zhang & Rothrock, 2003 (JGR)",
        spatial_resolution="1° polar stereographic grid",
        temporal_resolution="daily, 1979-present",
        access_note="Free download from PSC website. No authentication required.",
        intended_use="Long-term thickness climatology and trend analysis. Well-validated against ICESat/CryoSat.",
    ),
    DataSource(
        name="Copernicus Marine Arctic Ocean Sea Ice Reanalysis",
        variable="sea_ice_thickness, sea_ice_velocity, sea_ice_concentration",
        authority="Copernicus Marine Service / Mercator Ocean International",
        url="https://data.marine.copernicus.eu/product/ARCTIC_MULTIYEAR_PHY_ICE_002_016/description",
        doi_or_product_id="ARCTIC_MULTIYEAR_PHY_ICE_002_016",
        spatial_resolution="1/12° Arctic model grid",
        temporal_resolution="daily, 1991-present",
        access_note="Requires Copernicus Marine account/API client.",
        intended_use="Ice thickness/drift priors and joint distributions for coupled ice-ocean scenarios.",
    ),
    DataSource(
        name="Copernicus Marine Arctic Sea Ice Thickness Reprocessed L3",
        variable="sea_ice_thickness",
        authority="Copernicus Marine / C3S brokered product",
        url="https://data.marine.copernicus.eu/product/SEAICE_GLO_PHY_CLIMATE_L3_MY_011_013/description",
        doi_or_product_id="10.48670/moi-00127",
        spatial_resolution="satellite altimetry L3 monthly files",
        temporal_resolution="freezing season, October-April",
        access_note="Useful where CryoSat/Envisat thickness coverage exists; not year-round.",
        intended_use="Thickness model calibration and seasonal uncertainty envelopes.",
    ),
]

# ============================================================
# 3. 海冰漂移 (Sea Ice Drift, SID)
# ============================================================

SID_SOURCES: List[DataSource] = [
    DataSource(
        name="NSIDC Polar Pathfinder Daily Sea Ice Motion Vectors v4",
        variable="sea_ice_drift_u, sea_ice_drift_v",
        authority="NASA NSIDC DAAC",
        url="https://nsidc.org/data/nsidc-0116/versions/4",
        doi_or_product_id="NSIDC-0116 v4",
        spatial_resolution="25 km EASE grid",
        temporal_resolution="daily, 1978-present",
        access_note="Free via Earthdata. Combination of AVHRR, buoy, and passive microwave.",
        intended_use="Ice drift speed/direction priors for scenario calibration. Core data for ice force model.",
    ),
    DataSource(
        name="IFREMER CERSAT Sea Ice Drift",
        variable="sea_ice_drift_u, sea_ice_drift_v",
        authority="IFREMER / CERSAT",
        url="https://cersat.ifremer.fr/data/analysis-and-forecast/sea-ice-drift",
        doi_or_product_id="Girard-Ardhuin & Ezraty, 2012 (TGRS)",
        spatial_resolution="62.5 km polar stereographic",
        temporal_resolution="daily, 1991-present",
        access_note="Free download. SSM/I and ASCAT based.",
        intended_use="Cross-validation of ice drift; European Arctic and Antarctic coverage.",
    ),
]

# ============================================================
# 4. 海流 (Ocean Currents)
# ============================================================

OCEAN_SOURCES: List[DataSource] = [
    DataSource(
        name="OSCAR Ocean Surface Current Analyses Real-time",
        variable="ocean_surface_u, ocean_surface_v",
        authority="NASA PO.DAAC",
        url="https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_third-deg",
        doi_or_product_id="10.5067/OSCA-4303",
        spatial_resolution="1/3° global",
        temporal_resolution="5-day, 1992-present",
        access_note="Free via PO.DAAC Earthdata login.",
        intended_use="Ocean current forcing for ice drift validation and DP station-keeping scenarios.",
    ),
    DataSource(
        name="GlobCurrent Global Ocean Currents",
        variable="ocean_current_u, ocean_current_v",
        authority="Copernicus Marine / CERSAT",
        url="https://marine.copernicus.eu/product/MULTIOBS_GLO_PHY_REP_015_004",
        doi_or_product_id="10.48670/moi-00016",
        spatial_resolution="0.25° global",
        temporal_resolution="daily, 1993-present",
        access_note="Requires Copernicus Marine account.",
        intended_use="Multi-source ocean current product for ice-ocean coupling.",
    ),
]

# ============================================================
# 5. 大气强迫 (Atmospheric Forcing)
# ============================================================

ATMOS_SOURCES: List[DataSource] = [
    DataSource(
        name="ERA5 Single Levels",
        variable="10m_wind_u, 10m_wind_v, mean_sea_level_pressure, sea_ice_cover, SST",
        authority="ECMWF / Copernicus Climate Change Service",
        url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        doi_or_product_id="10.24381/cds.adbb2d47",
        spatial_resolution="0.25° global",
        temporal_resolution="hourly, 1940-present",
        access_note="Requires CDS API key (free registration) and cdsapi Python package.",
        intended_use="Wind/wave forcing for DP station-keeping; sea ice cover validation.",
    ),
    DataSource(
        name="JRA-55 Japanese 55-year Reanalysis",
        variable="surface_wind, pressure, temperature",
        authority="JMA / JRA-55",
        url="https://jra.kishou.go.jp/JRA-55/index_en.html",
        doi_or_product_id="Kobayashi et al., 2015 (JMSJ)",
        spatial_resolution="1.25° global",
        temporal_resolution="6-hourly, 1958-present",
        access_note="Free download from JMA. Alternative to ERA5 for Arctic reanalysis.",
        intended_use="Alternative atmospheric forcing; Arctic skill assessment.",
    ),
]

# ============================================================
# 6. 冰图 (Operational Ice Charts)
# ============================================================

ICE_CHART_SOURCES: List[DataSource] = [
    DataSource(
        name="NIC Sea Ice Analysis (Daily & Weekly)",
        variable="ice_stage, ice_concentration, ice_type",
        authority="US National Ice Center (NIC)",
        url="https://www.natice.noaa.gov/products/weekly_products.html",
        doi_or_product_id="MANICE / SIGRID-3 format",
        spatial_resolution="variable (polygon-based)",
        temporal_resolution="daily (Arctic summer), weekly (winter)",
        access_note="Free download from NIC website. Shape/raster formats.",
        intended_use="Operational ice chart for scenario validation; ice type classification.",
    ),
    DataSource(
        name="MASIE Ice Coverage (Daily)",
        variable="ice_extent, ice_coverage",
        authority="NSIDC / NIC",
        url="https://nsidc.org/data/masie",
        doi_or_product_id="NSIDC-0498",
        spatial_resolution="4 km polar stereographic",
        temporal_resolution="daily, 2006-present",
        access_note="Free via NSIDC.",
        intended_use="High-resolution daily ice extent for scenario boundary definition.",
    ),
]

# ============================================================
# 7. 实测校准数据 (In-Situ / Ice Tank)
# ============================================================

INSITU_SOURCES: List[DataSource] = [
    DataSource(
        name="N-ICE2015 Drifting Ice Station Data",
        variable="ice_thickness, ice_strength, ice temperature, floe properties",
        authority="Norwegian Polar Institute",
        url="https://data.npolar.no/dataset/4f3df7d4-de02-4241-bb27-27a780f69d17",
        doi_or_product_id="Granskog et al., 2016 (BAMS); Itkin et al., 2017 (JGR)",
        spatial_resolution="point measurements on ice floes",
        temporal_resolution="Jan-Jun 2015, Fram Strait",
        access_note="Free via Norwegian Polar Data Centre.",
        intended_use="In-situ ice thickness/strength calibration for Lindqvist model. Key validation dataset.",
    ),
    DataSource(
        name="MOSAiC Multidisciplinary drifting Observatory for the Study of Arctic Climate",
        variable="ice_thickness, ice_strength, ice temperature, snow depth, drift track",
        authority="AWI / MOSAiC Data Portal",
        url="https://mosaic.awi.de/",
        doi_or_product_id="Shupe et al., 2022 (Elementa)",
        spatial_resolution="point + transect, Central Arctic",
        temporal_resolution="Oct 2019 - Sep 2020, full annual cycle",
        access_note="Free registration at MOSAiC Data Portal. Most comprehensive Arctic ice dataset to date.",
        intended_use="Full-cycle ice property calibration; drift speed distributions; ice strength validation.",
    ),
    DataSource(
        name="HSVA Ice Tank Test Data (Hamburg Ship Model Basin)",
        variable="ice_force, ice_concentration, ice_thickness, ship_speed",
        authority="HSVA / Ship Design and Safety Lab",
        url="https://www.hsva.de/en/facilities/ice-basin/",
        doi_or_product_id="various publications",
        spatial_resolution="controlled model scale",
        temporal_resolution="experimental campaigns",
        access_note="Data typically obtained through research collaboration. Cite HSVA when using.",
        intended_use="Ice force model calibration at model scale. Key for validating Lindqvist parameters.",
    ),
    DataSource(
        name="Aalto Ice Tank Test Data (Finland)",
        variable="ice_force, ice_concentration, ice_type",
        authority="Aalto University / Kymenlaakso University of Applied Sciences",
        url="https://www.aalto.fi/en/department-of-mechanical-engineering/marine-technology",
        doi_or_product_id="various publications",
        spatial_resolution="controlled model scale",
        temporal_resolution="experimental campaigns",
        access_note="Data obtained through research collaboration.",
        intended_use="Ice force validation for Baltic/northern sea ice conditions.",
    ),
]

# ============================================================
# 8. 船舶轨迹 (Vessel Tracking, for scenario validation)
# ============================================================

VESSEL_SOURCES: List[DataSource] = [
    DataSource(
        name="MarineTraffic / VesselFinder AIS Data",
        variable="vessel_position, speed, heading, route",
        authority="MarineTraffic / VesselFinder",
        url="https://www.marinetraffic.com/",
        doi_or_product_id="commercial product",
        spatial_resolution="point tracks",
        temporal_resolution="near-real-time",
        access_note="Commercial API. Academic licenses available. Use for route validation only.",
        intended_use="Arctic vessel route validation; DP operation pattern analysis.",
    ),
    DataSource(
        name="XueLong / XueLong2 Cruise Track Data",
        variable="vessel_position, ice conditions, weather",
        authority="Polar Research Institute of China (PRIC)",
        url="http://www.pric.org.cn/",
        doi_or_product_id="various expedition reports",
        spatial_resolution="GPS tracks",
        temporal_resolution="expedition-based",
        access_note="Obtain through PRIC collaboration or published expedition reports.",
        intended_use="Direct validation of DP scenarios in Arctic ice conditions.",
    ),
]


# ============================================================
# 文献校准参数 (Literature-Calibrated Parameters)
# ============================================================

@dataclass(frozen=True)
class LiteratureCalibration:
    """从公开文献提取的冰力/冰况校准参数。"""
    source: str               # 文献引用
    parameter: str            # 参数名
    value: float              # 标称值
    unit: str                 # 单位
    range_min: float          # 文献报告最小值
    range_max: float          # 文献报告最大值
    notes: str = ""           # 备注


LITERATURE_CALIBRATIONS: List[LiteratureCalibration] = [
    # Lindqvist (1989) — 冰力模型基础
    LiteratureCalibration(
        source="Lindqvist, 1989 (POAC)",
        parameter="ice_crushing_strength",
        value=2.0, unit="MPa",
        range_min=0.5, range_max=5.0,
        notes="Uniaxial compressive strength of level ice. Varies with temperature, salinity, strain rate.",
    ),
    LiteratureCalibration(
        source="Lindqvist, 1989 (POAC)",
        parameter="structure_factor",
        value=0.45, unit="dimensionless",
        range_min=0.3, range_max=0.7,
        notes="Empirical shape factor for ship-ice interaction. Depends on hull geometry.",
    ),
    LiteratureCalibration(
        source="Lindqvist, 1989 (POAC)",
        parameter="waterline_angle",
        value=30.0, unit="degrees",
        range_min=15.0, range_max=45.0,
        notes="Waterline entrance angle. XueLong2 ~25-35 deg.",
    ),
    # ISO 19906 (2019) — Arctic offshore structures
    LiteratureCalibration(
        source="ISO 19906:2019",
        parameter="ice_crushing_strength_level_ice",
        value=2.4, unit="MPa",
        range_min=1.0, range_max=5.0,
        notes="Level ice compressive strength for Arctic structures. Conservative for ship interaction.",
    ),
    LiteratureCalibration(
        source="ISO 19906:2019",
        parameter="ice_crushing_strength_ridged_ice",
        value=5.0, unit="MPa",
        range_min=2.0, range_max=10.0,
        notes="Consolidated ridge keel strength. For extreme ice scenarios.",
    ),
    # Riska (1997) — 芬兰冰力经验
    LiteratureCalibration(
        source="Riska, 1997 (Helsinki University of Technology)",
        parameter="empirical_ice_pressure",
        value=2.5, unit="MPa",
        range_min=1.0, range_max=6.0,
        notes="Empirical ice pressure for ship transit in Baltic ice.",
    ),
    # 雪龙2号船舶参数
    LiteratureCalibration(
        source="PRIC Technical Report / CSIC Ship Design",
        parameter="xuelong2_length",
        value=122.5, unit="m",
        range_min=122.0, range_max=123.0,
        notes="XueLong2 overall length. LOA=122.5m, beam=22.3m.",
    ),
    LiteratureCalibration(
        source="PRIC Technical Report",
        parameter="xuelong2_displacement",
        value=14000.0, unit="tonnes",
        range_min=13000.0, range_max=15000.0,
        notes="Full load displacement. Lightship ~8000t.",
    ),
    LiteratureCalibration(
        source="PRIC Technical Report",
        parameter="xuelong2_ice_class",
        value=7.0, unit="polar class",
        range_min=7.0, range_max=7.0,
        notes="Polar Class 7 (PC7), can transit 1.0m first-year ice at 5 knots.",
    ),
    # DP 系统参数
    LiteratureCalibration(
        source="IMO MSC.1/Circ.1580 (2017)",
        parameter="dp3_position_accuracy",
        value=1.0, unit="m",
        range_min=0.5, range_max=3.0,
        notes="DP3 station-keeping accuracy target in open water.",
    ),
    LiteratureCalibration(
        source="DNV-OS-D202 (2021)",
        parameter="dp_watch_circle_radius",
        value=10.0, unit="m",
        range_min=5.0, range_max=20.0,
        notes="Typical watch circle for DP operations. Scenario D1 uses 12m.",
    ),
]


# ============================================================
# 汇总: 所有权威数据源
# ============================================================

AUTHORITATIVE_SOURCES: List[DataSource] = (
    SIC_SOURCES + SIT_SOURCES + SID_SOURCES + OCEAN_SOURCES
    + ATMOS_SOURCES + ICE_CHART_SOURCES + INSITU_SOURCES + VESSEL_SOURCES
)


# ============================================================
# 工具函数
# ============================================================

def _file_sha256(path: Path) -> str:
    """计算文件的 SHA256 校验和。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    path: Path,
    sources: Optional[Iterable[DataSource]] = None,
    include_literature: bool = True,
) -> None:
    """Write a JSON manifest that can be cited by all experiment outputs."""
    srcs = list(sources or AUTHORITATIVE_SOURCES)
    payload = {
        "manifest_version": "sci1-data-v3",
        "n_sources": len(srcs),
        "source_categories": {
            "sea_ice_concentration": len(SIC_SOURCES),
            "sea_ice_thickness": len(SIT_SOURCES),
            "sea_ice_drift": len(SID_SOURCES),
            "ocean_currents": len(OCEAN_SOURCES),
            "atmospheric_forcing": len(ATMOS_SOURCES),
            "ice_charts": len(ICE_CHART_SOURCES),
            "in_situ_calibration": len(INSITU_SOURCES),
            "vessel_tracking": len(VESSEL_SOURCES),
        },
        "note": (
            "Raw satellite/reanalysis products are not bundled because they are large and may require "
            "Earthdata/Copernicus authentication. Experiments record whether data were externally "
            "downloaded or whether literature-calibrated priors were used."
        ),
        "sources": [s.to_dict() for s in srcs],
    }
    if include_literature:
        payload["literature_calibrations"] = [asdict(lc) for lc in LITERATURE_CALIBRATIONS]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_manifest(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_literature_calibration(parameter: str) -> Optional[LiteratureCalibration]:
    """按参数名查找文献校准值。"""
    for lc in LITERATURE_CALIBRATIONS:
        if lc.parameter == parameter:
            return lc
    return None


def update_source_status(
    manifest_path: Path,
    source_name: str,
    status: str,
    local_path: Optional[str] = None,
) -> None:
    """更新 manifest 中指定数据源的状态。"""
    manifest = load_manifest(manifest_path)
    for src in manifest.get("sources", []):
        if src.get("name") == source_name:
            src["status"] = status
            if local_path:
                src["local_path"] = local_path
                try:
                    src["checksum_sha256"] = _file_sha256(Path(local_path))
                except FileNotFoundError:
                    pass
            from datetime import datetime, timezone
            src["download_date"] = datetime.now(timezone.utc).isoformat()
            break
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# 下载工具
# ============================================================

def download_nsidc_sic(
    output_dir: Path,
    start_date: str = "2020-01-01",
    end_date: str = "2020-12-31",
    version: int = 4,
) -> Path:
    """下载 NOAA/NSIDC CDR 海冰密集度数据。

    需要: pip install earthaccess + NASA Earthdata 账号
    """
    try:
        import earthaccess
    except ImportError:
        raise ImportError(
            "earthaccess is required. Install: pip install earthaccess\n"
            "Also need: https://urs.earthdata.nasa.gov/"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        granules = earthaccess.search_data(
            short_name="NSIDC-0079", version=f"v{version}",
            temporal=(start_date, end_date),
            bounding_box=(-180, 60, 180, 90),
        )
        if not granules:
            raise RuntimeError("No NSIDC CDR SIC granules found.")
        earthaccess.download(granules, str(output_dir))
        return output_dir
    except Exception as e:
        raise RuntimeError(f"NSIDC download failed: {e}")


def download_copernicus_ice(
    output_dir: Path,
    product_id: str = "ARCTIC_MULTIYEAR_PHY_ICE_002_016",
    start_date: str = "2020-01-01",
    end_date: str = "2020-12-31",
) -> Path:
    """下载 Copernicus Marine 海冰数据。

    需要: pip install copernicusmarine + Copernicus Marine 账号
    """
    try:
        import copernicusmarine
    except ImportError:
        raise ImportError(
            "copernicusmarine is required. Install: pip install copernicusmarine\n"
            "Also need: https://marine.copernicus.eu/"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        copernicusmarine.subset(
            dataset_id=product_id,
            variables=["siconc", "sithick", "sivelu", "sivelv"],
            start_datetime=start_date, end_datetime=end_date,
            minimum_longitude=-180, maximum_longitude=180,
            minimum_latitude=60, maximum_latitude=90,
            output_filename=str(output_dir / f"{product_id}_{start_date}_{end_date}.nc"),
        )
        return output_dir
    except Exception as e:
        raise RuntimeError(f"Copernicus Marine download failed: {e}")


def download_era5_arctic(
    output_dir: Path,
    start_date: str = "2020-01-01",
    end_date: str = "2020-12-31",
) -> Path:
    """下载 ERA5 北极再分析数据。

    需要: pip install cdsapi + CDS API key (~/.cdsapirc)
    """
    try:
        import cdsapi
    except ImportError:
        raise ImportError(
            "cdsapi is required. Install: pip install cdsapi\n"
            "Also need API key: https://cds.climate.copernicus.eu/"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "10m_u_component_of_wind", "10m_v_component_of_wind",
                    "mean_sea_level_pressure", "sea_ice_cover", "sea_surface_temperature",
                ],
                "year": start_date[:4],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": [90, -180, 60, 180],  # North Pole to 60°N
                "format": "netcdf",
            },
            str(output_dir / f"era5_arctic_{start_date[:4]}.nc"),
        )
        return output_dir
    except Exception as e:
        raise RuntimeError(f"ERA5 download failed: {e}")


# ============================================================
# NetCDF 数据加载
# ============================================================

def load_sic_from_netcdf(nc_path: Path, lat: float, lon: float) -> Optional[float]:
    """从 NetCDF 文件加载指定位置的海冰密集度。"""
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        for var_name in ["cdr_seaice_conc", "sea_ice_concentration", "siconc", "ice_conc"]:
            if var_name in ds:
                var = ds[var_name]
                val = var.sel(y=lat, x=lon, method="nearest").values
                ds.close()
                return float(np.clip(val, 0.0, 1.0)) if not np.isnan(val) else None
        ds.close()
        return None
    except Exception:
        return None


def load_ice_thickness_from_netcdf(nc_path: Path, lat: float, lon: float) -> Optional[float]:
    """从 NetCDF 文件加载指定位置的海冰厚度。"""
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        for var_name in ["sithick", "sea_ice_thickness", "thickness"]:
            if var_name in ds:
                var = ds[var_name]
                val = var.sel(y=lat, x=lon, method="nearest").values
                ds.close()
                return float(max(0.0, val)) if not np.isnan(val) else None
        ds.close()
        return None
    except Exception:
        return None


def load_ice_drift_from_netcdf(nc_path: Path, lat: float, lon: float) -> Optional[Dict[str, float]]:
    """从 NetCDF 文件加载指定位置的海冰漂移速度 (u, v)。"""
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        u_var = None
        v_var = None
        for u_name in ["u", "ice_drift_u", "siu", "sea_ice_drift_u"]:
            if u_name in ds:
                u_var = ds[u_name]
                break
        for v_name in ["v", "ice_drift_v", "siv", "sea_ice_drift_v"]:
            if v_name in ds:
                v_var = ds[v_name]
                break
        if u_var is not None and v_var is not None:
            u_val = float(u_var.sel(y=lat, x=lon, method="nearest").values)
            v_val = float(v_var.sel(y=lat, x=lon, method="nearest").values)
            ds.close()
            if not (np.isnan(u_val) or np.isnan(v_val)):
                speed = float(np.sqrt(u_val**2 + v_val**2))
                direction = float(np.degrees(np.arctan2(v_val, u_val)))
                return {"u": u_val, "v": v_val, "speed": speed, "direction": direction}
        ds.close()
        return None
    except Exception:
        return None
