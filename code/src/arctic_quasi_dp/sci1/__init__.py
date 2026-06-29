"""SCI一区投稿级实验扩展包。

该包把原 arctic_quasi_dp 原型升级为"高等级 DP 系统的冰区增强实验平台"。
核心思想：Precision DP 为主模式，Ice-aware DP 为增强模式，Quasi-DP 为安全降级，
Ice-vaning/Escape 为极端保护。
"""

from .controllers import (
    PrecisionDPController,
    IceAwarePrecisionDPController,
    QuasiDPSafetyController,
    IceVaningEscapeController,
    ModeSupervisedIceDPController,
)
from .scenarios import SCI1Scenario, build_sci1_scenarios
from .metrics import summarize_run, aggregate_summary, run_all_comparisons, statistical_comparison
from .data_sources import (
    DataSource,
    LiteratureCalibration,
    AUTHORITATIVE_SOURCES,
    LITERATURE_CALIBRATIONS,
    write_manifest,
    load_manifest,
    update_source_status,
    get_literature_calibration,
    download_nsidc_sic,
    download_copernicus_ice,
    download_era5_arctic,
)
from .thruster import ThrusterAllocator, ThrusterConfig, ThrusterUnit, ThrusterDegradationProfile
from .ice_schedule import (
    IceSchedule,
    ConstantIce,
    LinearRampIce,
    StepIce,
    SinusoidalIce,
    RandomWalkIce,
    PiecewiseIce,
)
from .sim_loop import run_simulation, VesselState, VesselParams, SimLog, WindState
from .sensor_models import IceLoadObserver, DisturbanceObserver
from .baseline_controllers import LQGController, LQGParams, DOBMPCController, DOBMPCParams
from .data_bridge import (
    DataDrivenIceSchedule,
    DataDrivenWindSchedule,
    load_copernicus_ice_data,
    load_era5_wind_data,
    extract_ice_statistics,
    generate_scenarios_from_data,
)

# NMPC 可选导入 (需要 casadi)
try:
    from .nmpc_controller import NMPCIceController, NMPCParams, check_casadi_available
    _nmpc_available = True
except ImportError:
    _nmpc_available = False

__all__ = [
    # 控制器
    "PrecisionDPController",
    "IceAwarePrecisionDPController",
    "QuasiDPSafetyController",
    "IceVaningEscapeController",
    "ModeSupervisedIceDPController",
    # 场景
    "SCI1Scenario",
    "build_sci1_scenarios",
    # 指标
    "summarize_run",
    "aggregate_summary",
    "run_all_comparisons",
    "statistical_comparison",
    # 数据源
    "DataSource",
    "LiteratureCalibration",
    "AUTHORITATIVE_SOURCES",
    "LITERATURE_CALIBRATIONS",
    "write_manifest",
    "load_manifest",
    "update_source_status",
    "get_literature_calibration",
    "download_nsidc_sic",
    "download_copernicus_ice",
    "download_era5_arctic",
    # 推进器
    "ThrusterAllocator",
    "ThrusterConfig",
    "ThrusterUnit",
    "ThrusterDegradationProfile",
    # 冰况调度
    "IceSchedule",
    "ConstantIce",
    "LinearRampIce",
    "StepIce",
    "SinusoidalIce",
    "RandomWalkIce",
    "PiecewiseIce",
    # 仿真循环
    "run_simulation",
    "VesselState",
    "VesselParams",
    "SimLog",
    # 观测器
    "IceLoadObserver",
    "DisturbanceObserver",
    # 补充基线控制器
    "LQGController",
    "LQGParams",
    "DOBMPCController",
    "DOBMPCParams",
    # 数据桥接
    "DataDrivenIceSchedule",
    "DataDrivenWindSchedule",
    "WindState",
    "load_copernicus_ice_data",
    "load_era5_wind_data",
    "extract_ice_statistics",
    "generate_scenarios_from_data",
]

if _nmpc_available:
    __all__.extend(["NMPCIceController", "NMPCParams", "check_casadi_available"])
