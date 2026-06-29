"""SCI一区投稿实验 runner。

所有场景统一使用 sim_loop 仿真循环，支持时变冰况、推进器分配和 NMPC。

命令示例：
  python scripts/run_sci1_experiments.py --profile smoke --seeds 2
  python scripts/run_sci1_experiments.py --profile paper --seeds 50 --controllers pid precision ice_aware full nmpc no_cbf no_cvar no_observer no_fallback

输出：
  results/sci1_submission/<timestamp>/
    data_manifest.json
    scenario_manifest.json
    per_seed_metrics.csv
    aggregate_metrics_ci95.csv
    statistical_comparisons.csv
    figures/*.png|*.pdf
    traces/*.csv
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import json
import os
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from ..controllers.pid import PIDController, PIDParams
try:
    from ..controllers.smc import SMCController, SMCParams
except Exception:  # pragma: no cover
    SMCController = None
    SMCParams = None

from .controllers import (
    PrecisionDPController,
    PrecisionDPParams,
    IceAwarePrecisionDPController,
    IceAwareParams,
    QuasiDPSafetyController,
    QuasiDPParams,
    IceVaningEscapeController,
    EscapeParams,
    ModeSupervisedIceDPController,
    SupervisorParams,
)
from .data_sources import write_manifest
from .figures import make_all_figures
from .metrics import summarize_run, save_tables
from .manifest import build_manifest_from_config, save_manifest, validate_manifest
from .scenarios import build_sci1_scenarios, SCI1Scenario
from .ice_schedule import ConstantIce, LinearRampIce
from .sim_loop import run_simulation
from .thruster import ThrusterConfig, ThrusterDegradationProfile
from .sensor_models import IceLoadObserver
from .vessel_config import load_vessel_config, VesselConfigBundle
from .logging_config import get_logger

_logger = get_logger(__name__)


def _set_single_thread_numeric_env() -> None:
    """Limit BLAS/OpenMP thread pools to avoid oversubscription in process parallel runs."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")




# ---------- NMPC 可用性检测 ----------

_NMPC_AVAILABLE = False
try:
    from .nmpc_controller import NMPCIceController, NMPCParams
    _NMPC_AVAILABLE = True
except ImportError:
    pass

# ---------- 补充基线控制器 ----------

from .baseline_controllers import (  # noqa: E402
    LQGController, LQGParams,
    DOBMPCController, DOBMPCParams,
    ADRCController, ADRCParams,
    RobustMPCController, RobustMPCParams,
    TubeMPCController,
    LESOADRCController, LESOADRCParams,
)


# ---------- 推进器配置映射 ----------

_THRUSTER_CONFIGS = {
    "generic_dp": ThrusterConfig.generic_dp_vessel(),
    "xuelong2": ThrusterConfig.vessel_xuelong2(),
    "none": None,
}

# 功率限制版推进器配置 (用于 C7 场景)
# 代理尺度下 C7 冰况 (c=0.5, h=0.8, v=0.3) 典型功耗 ~0.022 kW
# 设置 cap 为 0.015 kW (约 65% 典型功耗), 确保触发
_power_limited = copy.deepcopy(ThrusterConfig.generic_dp_vessel())
_power_limited.max_total_power_kw = 0.015
_THRUSTER_CONFIGS["generic_dp_power_limited"] = _power_limited

_DEGRADATION_PROFILES = {
    "no_fault": ThrusterDegradationProfile.no_fault(),
    "bow_degraded_0.7": ThrusterDegradationProfile.bow_degradation(0.7),
    "bow_degraded_0.5": ThrusterDegradationProfile.bow_degradation(0.5),
    "severe": ThrusterDegradationProfile.severe_degradation(),
    "azimuth_locked": ThrusterDegradationProfile.azimuth_locked_profile(),
}


# ---------- 控制器构建 ----------

# ---------- 控制器能力矩阵 (单一来源: tables.py) ----------

from .tables import _CONTROLLER_CAPABILITIES  # noqa: E402 — 统一能力矩阵


def build_controller(name: str, thruster_config=None):
    """构建控制器实例。未实现或缺依赖时抛出明确错误。

    Args:
        name: 控制器名称
        thruster_config: 推进器配置 (仅 NMPC 使用, None=使用默认 generic_dp_vessel)
    """
    name = name.lower()
    if name == "pid":
        return PIDController(PIDParams())
    if name == "smc" and SMCController is not None:
        return SMCController(SMCParams())
    if name == "precision":
        return PrecisionDPController(PrecisionDPParams())
    if name == "ice_aware":
        return IceAwarePrecisionDPController(IceAwareParams())
    if name == "quasi_dp":
        return QuasiDPSafetyController()
    if name == "escape":
        return IceVaningEscapeController()
    if name == "full":
        return ModeSupervisedIceDPController()
    if name == "nmpc":
        if not _NMPC_AVAILABLE:
            raise ImportError("CasADi is required for NMPC controller. Install with: pip install casadi")
        return NMPCIceController(NMPCParams())
    if name == "no_cbf":
        return ModeSupervisedIceDPController(
            ice_aware=IceAwarePrecisionDPController(use_cbf=False),
            quasi=QuasiDPSafetyController(use_cbf=False),
            escape=IceVaningEscapeController(use_cbf=False),
        )
    if name == "no_cvar":
        return ModeSupervisedIceDPController(
            ice_aware=IceAwarePrecisionDPController(use_cvar=False),
            quasi=QuasiDPSafetyController(use_cvar=False),
            escape=IceVaningEscapeController(use_cvar=False),
        )
    if name == "no_observer":
        return ModeSupervisedIceDPController(
            ice_aware=IceAwarePrecisionDPController(use_observer=False),
            quasi=QuasiDPSafetyController(use_observer=False),
            escape=IceVaningEscapeController(use_observer=False),
        )
    if name == "no_fallback":
        # 真正的无降级: 直接使用 IceAwarePrecisionDPController, 不经过 supervisor
        return IceAwarePrecisionDPController()
    if name == "oracle_full":
        # Oracle: 使用真值冰况 (无观测器噪声), 保留完整 supervisor 架构
        return ModeSupervisedIceDPController(
            ice_aware=IceAwarePrecisionDPController(use_observer=False),
            quasi=QuasiDPSafetyController(use_observer=False),
            escape=IceVaningEscapeController(use_observer=False),
        )
    if name == "lqg":
        return LQGController(LQGParams())
    if name == "dob_nmpc":
        return DOBMPCController(DOBMPCParams())
    if name == "adrc":
        return ADRCController(ADRCParams())
    if name == "leso_adrc":
        return LESOADRCController(LESOADRCParams())
    if name == "robust_mpc":
        return RobustMPCController(RobustMPCParams())
    if name == "tube_mpc":
        return TubeMPCController(RobustMPCParams(tube_margin=0.12, solver_label="tube_mpc_proxy"))

    # --- Safety filter controllers ---
    if name in ("fixed_soft_hocbf", "cvar_soft_hocbf", "no_safety_filter", "no_tail_risk", "no_cvar_proxy"):
        from .control.controller_wrappers import make_filtered_controller
        # Build nominal controller (ice_aware as default nominal).
        # The method controllers use the proxy actuator-aware polygon by default.
        nominal = IceAwarePrecisionDPController(IceAwareParams())
        filter_type = name
        if name in ("no_tail_risk", "no_cvar_proxy"):
            filter_type = "fixed_soft_hocbf"
        return make_filtered_controller(
            nominal,
            filter_type=filter_type,
            risk_gain=2.0 if name == "cvar_soft_hocbf" else 0.0,
            constraint_mode="polygon" if name != "no_safety_filter" else "box",
            # A small proxy bounded-disturbance acceleration margin supports
            # theory-oriented diagnostics without claiming full-scale proof.
            disturbance_accel_bound_mps2=0.002 if name != "no_safety_filter" else 0.0,
        )

    raise ValueError(f"Unknown controller: {name}")


def save_controller_capability_matrix(out_dir: Path) -> None:
    """Save controller capability matrix CSV."""
    from .tables import generate_table2_controller_matrix
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    df = generate_table2_controller_matrix()
    df.to_csv(summary_dir / "controller_capability_matrix.csv", index=False)


# ---------- 冰况调度 ----------

# 模块级标志: paper 模式下数据缺失是否 fail fast
_FAIL_FAST_ON_MISSING_DATA = False
_RUN_DATA_USAGE: List[Dict[str, Any]] = []


def _resolve_data_path(raw_path: str | Path | None) -> Path | None:
    """Resolve data paths robustly from code root or archive root.

    The submitted archive may contain both ``code/data`` (mock fixture) and a
    top-level ``data`` folder (real subsets).  This helper tries the supplied
    path first, then paths relative to the current working directory and the
    package archive root.  It does not silently rewrite missing paths.
    """
    if raw_path is None:
        return None
    p = Path(raw_path)
    candidates = [p, Path.cwd() / p]
    repo_root = Path(__file__).resolve().parents[4]
    candidates.extend([repo_root / p, repo_root.parent / p])
    for c in candidates:
        if c.exists():
            return c
    return p


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_ice_schedule(scenario: SCI1Scenario):
    """根据场景创建冰况调度器。"""
    # 数据驱动场景: 使用 Copernicus NetCDF 真实数据
    if getattr(scenario, 'data_driven', False):
        nc_path = getattr(scenario, 'data_nc_path', None)
        if nc_path is not None:
            from pathlib import Path as _Path
            from .data_bridge import DataDrivenIceSchedule
            nc = _resolve_data_path(nc_path) or _Path(nc_path)
            source_type = getattr(scenario, 'data_source_type', 'mock_fixture')
            allow_mock = bool(getattr(scenario, 'allow_mock_fixture', True))
            if _FAIL_FAST_ON_MISSING_DATA and source_type == 'real_subset' and allow_mock:
                raise ValueError(
                    f"Scenario {scenario.scenario_id} requests real data but allow_mock_fixture=True. "
                    "Real replay configs must set allow_mock_fixture=false."
                )
            if nc.exists():
                try:
                    checksum = _sha256_file(nc)
                except OSError:
                    checksum = None
                drift_nc = None
                drift_nc_path = getattr(scenario, 'drift_nc_path', None)
                if drift_nc_path:
                    drift_nc = _resolve_data_path(drift_nc_path)
                    if drift_nc and not drift_nc.exists():
                        drift_nc = None
                schedule = DataDrivenIceSchedule(
                    nc, lat=scenario.data_lat, lon=scenario.data_lon,
                    duration=scenario.duration,
                    drift_nc_path=drift_nc,
                )
                _RUN_DATA_USAGE.append({
                    "scenario_id": scenario.scenario_id,
                    "data_source_type": source_type,
                    "actual_data_path": str(nc),
                    "sha256": checksum,
                    "provider": getattr(scenario, 'data_provider', ''),
                    "product_id": getattr(scenario, 'data_product_id', ''),
                    "variables_used": "siconc,sithick,vxsi/vysi or sivelu/sivelv",
                    "fallback_used": False,
                    "mock_fixture_used": bool(source_type == 'mock_fixture'),
                    "ice_data_provenance": getattr(schedule, 'provenance', {}),
                })
                return schedule
            elif _FAIL_FAST_ON_MISSING_DATA:
                raise FileNotFoundError(
                    f"H-group scenario {scenario.scenario_id}: "
                    f"required data file not found: {nc_path}. "
                    f"Run download_data.py to fetch real data, "
                    f"or set runtime.fail_fast_on_missing_data=false to allow synthetic fallback."
                )
            else:
                _logger.warning(
                    "H-group scenario %s: NetCDF file not found (%s), "
                    "falling back to static ice conditions (synthetic fallback). "
                    "Run download_data.py to fetch real data.",
                    scenario.scenario_id, nc_path,
                )
                _RUN_DATA_USAGE.append({
                    "scenario_id": scenario.scenario_id,
                    "data_source_type": source_type,
                    "actual_data_path": str(nc),
                    "sha256": None,
                    "provider": getattr(scenario, 'data_provider', ''),
                    "product_id": getattr(scenario, 'data_product_id', ''),
                    "variables_used": "none",
                    "fallback_used": True,
                    "mock_fixture_used": False,
                })
    if not scenario.ice_time_varying:
        return ConstantIce(
            scenario.ice_concentration, scenario.ice_thickness,
            scenario.ice_drift_speed, scenario.ice_drift_direction,
        )
    schedule_type = getattr(scenario, 'ice_schedule_type', 'default')
    if schedule_type == "step":
        from .ice_schedule import StepIce
        return StepIce(
            c_a=scenario.ice_concentration_initial,
            h_a=scenario.ice_thickness_initial,
            v_a=scenario.ice_drift_speed_initial,
            dir_a=scenario.ice_drift_direction_initial,
            c_b=scenario.ice_concentration_final,
            h_b=scenario.ice_thickness_final,
            v_b=scenario.ice_drift_speed_final,
            dir_b=scenario.ice_drift_direction_final,
            t_change=scenario.duration * 0.4,
        )
    return LinearRampIce(
        c0=scenario.ice_concentration_initial,
        h0=scenario.ice_thickness_initial,
        v0=scenario.ice_drift_speed_initial,
        dir0=scenario.ice_drift_direction_initial,
        c1=scenario.ice_concentration_final,
        h1=scenario.ice_thickness_final,
        v1=scenario.ice_drift_speed_final,
        dir1=scenario.ice_drift_direction_final,
        duration=scenario.duration,
    )


# ---------- 推进器配置 ----------

def _get_thruster_config(scenario: SCI1Scenario):
    """从场景获取推进器配置 (deep copy 防止跨 run 污染)。"""
    name = getattr(scenario, 'thruster_config_name', 'generic_dp')
    cfg = _THRUSTER_CONFIGS.get(name)
    return copy.deepcopy(cfg) if cfg is not None else None


def _get_degradation_profile(scenario: SCI1Scenario):
    """从场景获取推进器退化配置 (deep copy 防止跨 run 污染)。"""
    name = getattr(scenario, 'degradation_name', 'no_fault')
    prof = _DEGRADATION_PROFILES.get(name, _DEGRADATION_PROFILES["no_fault"])
    return copy.deepcopy(prof)


# ---------- 传感器/冰力模型/风场构建 ----------

def _build_position_sensor(scenario: SCI1Scenario):
    """从场景构建位置传感器模型。"""
    cfg = getattr(scenario, 'position_sensor_config', None)
    if cfg is None:
        return None
    from .sensor_models import PositionSensorModel, SensorNoiseConfig
    return PositionSensorModel(SensorNoiseConfig(**cfg))


def _build_heading_sensor(scenario: SCI1Scenario):
    """从场景构建航向传感器模型。"""
    cfg = getattr(scenario, 'heading_sensor_config', None)
    if cfg is None:
        return None
    from .sensor_models import HeadingSensorModel, SensorNoiseConfig
    return HeadingSensorModel(SensorNoiseConfig(**cfg))


def _build_ice_sensor_model(scenario: SCI1Scenario):
    """从场景构建冰况传感器模型。"""
    cfg = getattr(scenario, 'ice_sensor_config', None)
    if cfg is None:
        return None
    from .sensor_models import IceConditionSensorModel
    return IceConditionSensorModel(**cfg)


def _build_ice_load_model(scenario: SCI1Scenario):
    """从场景构建冰力模型。"""
    name = getattr(scenario, 'ice_load_model_name', 'default')
    if name == 'default':
        return None  # 使用内置 _ice_force_body
    from .ice_models import EmpiricalIceLoadModel, StochasticIceLoadModel, BenchmarkIceLoadModel
    if name == 'empirical':
        return EmpiricalIceLoadModel()
    if name == 'stochastic':
        return StochasticIceLoadModel()
    if name == 'benchmark':
        return BenchmarkIceLoadModel()
    raise ValueError(f"Unknown ice_load_model_name: {name}")


def _build_wind_schedule(scenario: SCI1Scenario):
    """从场景构建风场调度器。"""
    cfg = getattr(scenario, 'wind_config', None)
    if cfg is None:
        return None
    if cfg.get("type") == "constant":
        from .sim_loop import ConstantWindSchedule
        return ConstantWindSchedule(u10=cfg.get("u10", 0.0), v10=cfg.get("v10", 0.0))
    if cfg.get("type") == "data_driven" and "era5_path" in cfg:
        from .data_bridge import DataDrivenWindSchedule
        era5_path = Path(cfg["era5_path"])
        if era5_path.exists():
            return DataDrivenWindSchedule(
                nc_path=era5_path,
                lat=cfg.get("lat", 80.0),
                lon=cfg.get("lon", 0.0),
                duration=scenario.duration,
            )
    return None


# ---------- 元数据 ----------

def _metadata() -> Dict[str, str]:
    import platform as _platform
    try:
        platform_name = os.uname().sysname
    except AttributeError:
        platform_name = _platform.system()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": os.sys.version.split()[0],
        "platform": platform_name,
        "experiment_protocol": "sci1_submission_v2",
        "simulation_loop": "sim_loop_unified",
        "nmpc_available": str(_NMPC_AVAILABLE),
        "claim": "High-level DP ice-region enhancement: precision DP primary, ice-aware risk enhancement, quasi-DP fallback, escape/ice-vaning extreme mode.",
    }


# ---------- 单次仿真运行 ----------

def _run_single(
    scenario: SCI1Scenario,
    ctrl_name: str,
    controller,
    seed: int,
    profile: str,
    vessel_bundle: Optional[VesselConfigBundle] = None,
) -> "tuple[pd.DataFrame, float]":
    """运行单个 场景×控制器×seed 的仿真。统一使用 sim_loop。"""
    ice_schedule = _make_ice_schedule(scenario)
    thruster_config = _get_thruster_config(scenario)
    degradation = _get_degradation_profile(scenario)

    # 构建传感器/冰力模型/风场
    position_sensor = _build_position_sensor(scenario)
    heading_sensor = _build_heading_sensor(scenario)
    ice_sensor_model = _build_ice_sensor_model(scenario)
    ice_load_model = _build_ice_load_model(scenario)
    wind_schedule = _build_wind_schedule(scenario)

    # 速率限制
    max_azimuth_rate = getattr(scenario, 'max_azimuth_rate', 0.0)
    max_thrust_rate = getattr(scenario, 'max_thrust_rate', 0.0)

    if hasattr(controller, 'set_cvar_seed'):
        controller.set_cvar_seed(20260625 + seed)

    # Inform safety-filter wrappers about actuator mode before the run.
    if hasattr(controller, 'set_actuator_mode'):
        actuator_mode = getattr(scenario, 'degradation_name', 'no_fault') or 'nominal'
        power_scale_factor = 1.0
        if getattr(scenario, 'azimuth_locked_angle_deg', None) is not None:
            actuator_mode = 'azimuth_locked'
        if getattr(scenario, 'thruster_config_name', '') == 'generic_dp_power_limited':
            actuator_mode = 'power_limited'
            power_scale_factor = 0.55
        if getattr(scenario, 'max_azimuth_rate', 0.0) > 0 or getattr(scenario, 'max_thrust_rate', 0.0) > 0:
            actuator_mode = 'rate_limited'
        controller.set_actuator_mode(actuator_mode, power_scale_factor=power_scale_factor)

    # 创建扰动观测器 (当控制器有 observer 能力时启用)
    disturbance_observer = None
    ctrl_caps = _CONTROLLER_CAPABILITIES.get(ctrl_name, {})
    if ctrl_caps.get("observer", False):
        disturbance_observer = IceLoadObserver(alpha=0.15)

    # 船舶参数: 优先使用 vessel_bundle, 否则使用默认
    vessel_params = vessel_bundle.vessel_params if vessel_bundle is not None else None

    # 将船舶参数传递给控制器 (消除硬编码阻尼/质量不匹配)
    if vessel_params is not None and hasattr(controller, 'set_vessel_params'):
        controller.set_vessel_params(vessel_params)

    # Oracle 模式: 跳过传感器噪声, 直接使用真值冰况
    is_oracle = ctrl_name == "oracle_full"
    effective_position_sensor = None if is_oracle else position_sensor
    effective_heading_sensor = None if is_oracle else heading_sensor
    effective_ice_sensor = None if is_oracle else ice_sensor_model

    log = run_simulation(
        controller=controller,
        duration=scenario.duration,
        dt=scenario.dt,
        ice_schedule=ice_schedule,
        wind_schedule=wind_schedule,
        vessel_params=vessel_params,
        thruster_config=thruster_config,
        degradation_profile=degradation,
        target_x=scenario.target_x,
        target_y=scenario.target_y,
        target_psi=scenario.target_psi,
        safe_region_radius=scenario.safe_region_radius,
        seed=20260625 + seed,
        log_interval=1,
        position_sensor=effective_position_sensor,
        heading_sensor=effective_heading_sensor,
        ice_sensor_model=effective_ice_sensor,
        ice_load_model=ice_load_model,
        disturbance_observer=disturbance_observer,
        target_x_final=getattr(scenario, 'target_x_final', None),
        target_y_final=getattr(scenario, 'target_y_final', None),
        target_psi_final=getattr(scenario, 'target_psi_final', None),
        target_change_time=getattr(scenario, 'target_change_time', 0.0),
        max_azimuth_rate=max_azimuth_rate,
        max_thrust_rate=max_thrust_rate,
    )
    df = log.to_dataframe()
    return df, scenario.dt


# ---------- 主实验循环 ----------

def _create_run_directories(out_dir: Path) -> Dict[str, Path]:
    """Create structured output directories."""
    dirs = {
        "root": out_dir,
        "metadata": out_dir / "metadata",
        "raw": out_dir / "raw",
        "raw_traces": out_dir / "raw" / "per_timestep_traces",
        "tasks": out_dir / "raw" / "tasks",
        "summary": out_dir / "summary",
        "figures_main": out_dir / "figures" / "main",
        "figures_supp": out_dir / "figures" / "supplementary",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@dataclass(frozen=True)
class ExperimentTask:
    """One embarrassingly parallel experiment unit."""

    scenario: Dict[str, Any]
    controller: str
    seed: int
    profile: str
    config_hash: str

    @property
    def scenario_id(self) -> str:
        return str(self.scenario.get("scenario_id", "unknown"))

    @property
    def task_hash(self) -> str:
        raw = json.dumps(
            {
                "scenario_id": self.scenario_id,
                "controller": self.controller,
                "seed": self.seed,
                "profile": self.profile,
                "config_hash": self.config_hash,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]


def _build_tasks(scenarios: List[SCI1Scenario], controllers: List[str], seeds: int, profile: str, config_hash: str) -> List[ExperimentTask]:
    return [
        ExperimentTask(s.to_dict(), c, int(seed), profile, config_hash)
        for s in scenarios
        for c in controllers
        for seed in range(seeds)
    ]


def _task_result_path(tasks_dir: Path, task_hash: str) -> Path:
    return tasks_dir / f"{task_hash}.json"


def _task_done_path(tasks_dir: Path, task_hash: str) -> Path:
    return tasks_dir / f"{task_hash}.done"


def _load_finished_task(tasks_dir: Path, task: ExperimentTask) -> Optional[Dict[str, Any]]:
    path = _task_result_path(tasks_dir, task.task_hash)
    done = _task_done_path(tasks_dir, task.task_hash)
    if not (path.exists() and done.exists()):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") == "ok" and isinstance(payload.get("row"), dict):
            return payload
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _sanitize_for_json(obj: Any) -> Any:
    """递归替换 NaN/Inf 为 None，确保输出符合 RFC 8259 JSON 标准。"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _write_task_payload(tasks_dir: Path, task_hash: str, payload: Dict[str, Any]) -> None:
    path = _task_result_path(tasks_dir, task_hash)
    tmp = path.with_suffix(".json.tmp")
    sanitized = _sanitize_for_json(payload)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)
    _task_done_path(tasks_dir, task_hash).write_text("done\n", encoding="utf-8")


def _trace_should_be_saved(save_traces: bool, save_traces_on_failure: bool, row: Dict[str, Any]) -> bool:
    if save_traces:
        return True
    if save_traces_on_failure:
        return bool(row.get("failure", 0)) or float(row.get("safety_filter_infeasible_rate", 0.0) or 0.0) > 0.0
    return False


def _write_trace_if_requested(
    df: pd.DataFrame,
    raw_traces_dir: Path,
    scenario_id: str,
    controller: str,
    seed: int,
    save_trace: bool,
    trace_downsample: int = 1,
) -> Optional[str]:
    if not save_trace:
        return None
    trace_downsample = max(1, int(trace_downsample or 1))
    trace_df = df.iloc[::trace_downsample].copy() if trace_downsample > 1 else df
    trace_path = raw_traces_dir / f"{scenario_id}__{controller}__seed{seed}.csv"
    trace_df.to_csv(trace_path, index=False)
    return str(trace_path)


def _worker_init() -> None:
    _set_single_thread_numeric_env()


def _run_task_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one task in a worker process and write an atomic task result."""
    _set_single_thread_numeric_env()
    task = ExperimentTask(**payload["task"])
    tasks_dir = Path(payload["tasks_dir"])
    raw_traces_dir = Path(payload["raw_traces_dir"])
    save_traces = bool(payload.get("save_traces", False))
    save_traces_on_failure = bool(payload.get("save_traces_on_failure", False))
    trace_downsample = int(payload.get("trace_downsample", 1) or 1)
    vessel_config_path = payload.get("vessel_config_path")
    fail_fast_on_missing_data = bool(payload.get("fail_fast_on_missing_data", False))

    scenario = SCI1Scenario(**task.scenario)
    global _FAIL_FAST_ON_MISSING_DATA
    _FAIL_FAST_ON_MISSING_DATA = fail_fast_on_missing_data
    try:
        controller = build_controller(task.controller, thruster_config=_get_thruster_config(scenario))
    except (ImportError, NotImplementedError) as e:
        result = {
            "status": "skipped",
            "task_hash": task.task_hash,
            "scenario_id": task.scenario_id,
            "controller": task.controller,
            "seed": task.seed,
            "reason": str(e),
        }
        _write_task_payload(tasks_dir, task.task_hash, result)
        return result

    vessel_bundle = None
    if vessel_config_path:
        vp = Path(vessel_config_path)
        if vp.exists():
            vessel_bundle = load_vessel_config(vp)

    try:
        df, dt = _run_single(scenario, task.controller, controller, task.seed, task.profile, vessel_bundle)
        df.insert(0, "seed", task.seed)
        df.insert(0, "controller", task.controller)
        df.insert(0, "scenario_id", scenario.scenario_id)
        _max_force = getattr(getattr(controller, 'params', None), 'max_force', 3000.0)
        _max_moment = getattr(getattr(controller, 'params', None), 'max_moment', 100000.0)
        row = summarize_run(df, scenario.scenario_id, task.controller, task.seed, dt,
                            safe_region_radius=scenario.safe_region_radius,
                            max_force=float(_max_force), max_moment=float(_max_moment))
        save_trace = _trace_should_be_saved(save_traces, save_traces_on_failure, row)
        trace_path = _write_trace_if_requested(
            df, raw_traces_dir, scenario.scenario_id, task.controller, task.seed, save_trace, trace_downsample
        )
        result = {
            "status": "ok",
            "task_hash": task.task_hash,
            "scenario_id": scenario.scenario_id,
            "controller": task.controller,
            "seed": task.seed,
            "row": row,
            "trace_path": trace_path,
            "data_usage": list(_RUN_DATA_USAGE),
        }
        _write_task_payload(tasks_dir, task.task_hash, result)
        return result
    except Exception as e:
        result = {
            "status": "failed",
            "task_hash": task.task_hash,
            "scenario_id": scenario.scenario_id,
            "controller": task.controller,
            "seed": task.seed,
            "error": repr(e),
        }
        _write_task_payload(tasks_dir, task.task_hash, result)
        raise


def _save_summary_from_run_rows(
    run_rows: List[Dict[str, Any]],
    dirs: Dict[str, Path],
    profile: str,
    save_figures: bool,
    statistics_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Write all merged summary/statistics/figure outputs from per-task rows."""
    if not run_rows:
        _logger.warning("No experiment rows available for summary generation.")
        return
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(dirs["raw"] / "per_seed_metrics.csv", index=False)

    from .metrics import save_tables
    save_tables(run_rows, dirs["summary"], statistics_cfg=statistics_cfg)

    if save_figures:
        agg_csv = dirs["summary"] / "aggregate_metrics_ci95.csv"
        if agg_csv.exists():
            make_all_figures(agg_csv, dirs["figures_main"], control_period_ms=100.0,
                             trace_dir=dirs["raw_traces"])

    from .tables import save_all_tables
    save_all_tables(run_df, dirs["summary"], profile=profile)

    # E4: 自动生成 LaTeX 表格
    try:
        from .latex_tables import generate_all_latex_tables
        latex_dir = dirs["summary"] / "latex"
        latex_files = generate_all_latex_tables(dirs["summary"], latex_dir)
        if latex_files:
            _logger.info("Generated %d LaTeX tables in %s", len(latex_files), latex_dir)
    except Exception as e:
        _logger.warning("LaTeX table generation failed: %s", e)


def _statistics_only(out_dir: Path, cfg: Dict[str, Any]) -> Path:
    dirs = _create_run_directories(out_dir)
    raw_csv = dirs["raw"] / "per_seed_metrics.csv"
    summary_csv = dirs["summary"] / "per_seed_metrics.csv"
    src = raw_csv if raw_csv.exists() else summary_csv
    if not src.exists():
        raise FileNotFoundError(f"No per-seed metrics found for statistics-only run in {out_dir}")
    run_rows = pd.read_csv(src).to_dict(orient="records")
    _save_summary_from_run_rows(
        run_rows,
        dirs,
        profile=str(cfg.get("profile", "statistics_only")),
        save_figures=bool(cfg.get("output", {}).get("save_figures", False)),
        statistics_cfg=cfg.get("statistics", {}),
    )
    return out_dir


def _save_metadata(
    dirs: Dict[str, Path],
    profile: str,
    controllers: List[str],
    seeds: int,
    scenarios: list,
    effective_cfg: Dict[str, Any],
    vessel_bundle: Optional[VesselConfigBundle] = None,
) -> None:
    """Save all metadata files."""
    meta = _metadata()

    # Run manifest
    run_manifest = {
        "metadata": meta,
        "profile": profile,
        "seeds": seeds,
        "controllers": controllers,
        "n_scenarios": len(scenarios),
        "scenario_ids": [s.scenario_id for s in scenarios],
    }
    (dirs["metadata"] / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Scenario manifest
    (dirs["metadata"] / "scenario_manifest.json").write_text(
        json.dumps({"metadata": meta, "profile": profile, "scenarios": [s.to_dict() for s in scenarios]},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Effective config
    (dirs["metadata"] / "effective_config.json").write_text(
        json.dumps(effective_cfg, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Git info — 添加 timeout 防止 git 进程挂起
    import subprocess
    try:
        _git_cwd = Path(__file__).parent.parent.parent.parent
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_git_cwd, timeout=10,
        ).decode().strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_git_cwd, timeout=10,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
        git_branch = "unknown"
    (dirs["metadata"] / "git_info.json").write_text(
        json.dumps({"hash": git_hash, "branch": git_branch}, indent=2), encoding="utf-8",
    )

    # Vessel manifest — 反映实际生效的船舶配置
    if vessel_bundle is not None:
        vp = vessel_bundle.vessel_params
        _is_full = vessel_bundle.scale_type.startswith("full_scale")
        vessel_manifest = {
            "name": vessel_bundle.name,
            "scale_type": vessel_bundle.scale_type,
            "full_scale_ready": False,
            "warning": (
                "This configuration is experimental and does not yet include fully scaled "
                "thrusters, controller gains, damping, and ice-load calibration."
                if _is_full else
                "Default experiments use proxy-scale parameters; results should not be "
                "interpreted as full-scale DP validation."
            ),
            "source_note": vessel_bundle.source_note,
            "mass_kg": vp.mass,
            "Iz_kgm2": vp.Izz,
            "length_m": vp.length,
            "beam_m": vp.beam,
            "ice_crushing_strength_mpa": vp.ice_crushing_strength_mpa,
            "ice_structure_factor": vp.ice_structure_factor,
            "max_force_n": vessel_bundle.max_force,
            "max_moment_nm": vessel_bundle.max_moment,
            "vessel_config_path": str(effective_cfg.get("vessel_config_path", "default")),
        }
    else:
        from .sim_loop import VesselParams
        vp = VesselParams()
        vessel_manifest = {
            "name": "simplified_500t",
            "scale_type": "proxy_scale",
            "full_scale_ready": False,
            "warning": "Default experiments use proxy-scale parameters; results should not be interpreted as full-scale DP validation.",
            "source_note": "Default proxy-scale model (500t). ice_crushing_strength=0.0003 MPa produces ~1 kN ice forces matching controller/thruster scale.",
            "mass_kg": vp.mass,
            "Iz_kgm2": vp.Izz,
            "length_m": vp.length,
            "beam_m": vp.beam,
            "ice_crushing_strength_mpa": vp.ice_crushing_strength_mpa,
            "ice_structure_factor": vp.ice_structure_factor,
            "max_force_n": 3000.0,
            "max_moment_nm": 100000.0,
            "vessel_config_path": "default",
        }
    (dirs["metadata"] / "vessel_manifest.json").write_text(
        json.dumps(vessel_manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Data manifest
    write_manifest(dirs["metadata"] / "data_manifest.json")
    # Actual per-run data usage manifest. This is deliberately separate from
    # the global source registry: it proves which file each replay scenario
    # actually used in this run.
    unique_usage = [dict(t) for t in {tuple(sorted(u.items())) for u in _RUN_DATA_USAGE}]
    (dirs["metadata"] / "actual_data_usage.json").write_text(
        json.dumps({"data_usage": unique_usage}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if unique_usage:
        pd.DataFrame(unique_usage).to_csv(dirs["summary"] / "actual_data_usage.csv", index=False)

    # C5: 自动生成完整运行 manifest (包含 SHA-256、环境、任务统计)
    try:
        full_manifest = build_manifest_from_config(
            config=effective_cfg,
            config_path=effective_cfg.get("_config_path"),
            data_usage=unique_usage,
            artifact_dir=dirs["root"],
        )
        full_manifest.scenario_ids = [s.scenario_id for s in scenarios]
        full_manifest.controller_ids = controllers
        full_manifest.seed_list = list(range(seeds))
        full_manifest.profile = profile
        # 不在此处验证: 任务尚未执行, task_count_completed=0 必然不等于 expected
        manifest_path = save_manifest(full_manifest, dirs["metadata"])
        _logger.info("Generated run manifest: %s", manifest_path)
    except Exception as e:
        _logger.warning("Manifest generation failed: %s", e)


def filter_scenarios(scenarios, include_groups=None, include_ids=None):
    """按 include_groups 和/或 include_ids 过滤场景列表。

    include_ids 优先: 如果指定了 include_ids, 只返回匹配的场景。
    否则按 include_groups 过滤。
    """
    if include_ids:
        ids = set(include_ids)
        return [s for s in scenarios if s.scenario_id in ids]
    if not include_groups:
        return scenarios
    groups = set(include_groups)
    return [s for s in scenarios if s.group.split("_", 1)[0] in groups]


def _dedupe_data_usage(usages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate per-task data-usage records while keeping deterministic order."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in usages:
        key = tuple(sorted((str(k), str(v)) for k, v in item.items()))
        if key not in seen:
            seen.add(key)
            out.append(dict(item))
    return out


def _write_actual_data_usage(dirs: Dict[str, Path], usages: List[Dict[str, Any]]) -> None:
    """Write actual replay data used by this run, separate from source registry."""
    unique_usage = _dedupe_data_usage(usages)
    (dirs["metadata"] / "actual_data_usage.json").write_text(
        json.dumps({"data_usage": unique_usage}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = dirs["summary"] / "actual_data_usage.csv"
    if unique_usage:
        pd.DataFrame(unique_usage).to_csv(csv_path, index=False)
    elif csv_path.exists():
        csv_path.unlink()


def _save_scale_analysis(dirs: Dict[str, Path], scenarios: List[SCI1Scenario], vessel_bundle: Optional[VesselConfigBundle]) -> None:
    """Save dimensionless scale-analysis evidence for proxy/full-scale claims."""
    try:
        from .scale_analysis import compute_dimensionless_groups
        from .sim_loop import VesselParams
        scale_rows = []
        vp_for_scale = vessel_bundle.vessel_params if vessel_bundle is not None else VesselParams()
        for sc in scenarios:
            groups = compute_dimensionless_groups(vp_for_scale, _get_thruster_config(sc), sc, control_dt=sc.dt)
            row = {
                "scenario_id": sc.scenario_id,
                "group": sc.group,
                "scale_type": vessel_bundle.scale_type if vessel_bundle is not None else "proxy_scale",
            }
            row.update(groups.to_dict())
            scale_rows.append(row)
        pd.DataFrame(scale_rows).to_csv(dirs["summary"] / "scale_analysis.csv", index=False)
    except Exception as e:
        warnings.warn(f"Scale analysis failed: {e}")


def run_experiments(
    profile: str,
    seeds: int,
    controllers: List[str],
    out_dir: Path,
    save_traces: bool = True,
    save_figures: bool = True,
    effective_cfg: Optional[Dict[str, Any]] = None,
    include_groups: Optional[List[str]] = None,
    include_ids: Optional[List[str]] = None,
    vessel_config_path: Optional[str] = None,
    jobs: int = 1,
    parallel_backend: str = "serial",
    resume: bool = False,
    save_traces_on_failure: bool = False,
    trace_downsample: int = 1,
    skip_statistics: bool = False,
) -> Path:
    """Run experiment matrix with optional scenario-controller-seed process parallelism.

    The natural parallel unit is one ``scenario x controller x seed`` task.  Each task
    writes an atomic JSON payload under ``raw/tasks`` so large Linux CPU servers can
    resume interrupted paper runs without recomputing completed simulations.
    """
    _set_single_thread_numeric_env()
    jobs = max(1, int(jobs or 1))
    trace_downsample = max(1, int(trace_downsample or 1))
    parallel_backend = (parallel_backend or "serial").lower()
    if parallel_backend not in {"serial", "process"}:
        raise ValueError(f"Unsupported parallel_backend={parallel_backend!r}; use 'serial' or 'process'.")

    dirs = _create_run_directories(out_dir)
    _RUN_DATA_USAGE.clear()
    scenarios = build_sci1_scenarios(profile)
    scenarios = filter_scenarios(scenarios, include_groups, include_ids)

    # 设置 data fail-fast 模式
    global _FAIL_FAST_ON_MISSING_DATA
    _FAIL_FAST_ON_MISSING_DATA = bool(effective_cfg.get("fail_fast_on_missing_data", False)) if effective_cfg else False

    # 加载船舶配置
    vessel_bundle: Optional[VesselConfigBundle] = None
    if vessel_config_path is not None:
        vc_path = Path(vessel_config_path)
        if vc_path.exists():
            vessel_bundle = load_vessel_config(vc_path)
            _logger.info("Loaded vessel config: %s (scale=%s, mass=%.0f kg)",
                        vessel_bundle.name, vessel_bundle.scale_type, vessel_bundle.vessel_params.mass)

            # full_scale_experimental 保护: paper profile 必须显式允许
            _is_paper = profile in ("paper", "paper_full", "paper_small", "method_paper_small")
            _allow_explicit = bool(effective_cfg.get("allow_experimental_full_scale", False)) if effective_cfg else False
            if vessel_bundle.scale_type == "full_scale_experimental" and _is_paper and not _allow_explicit:
                raise ValueError(
                    f"Vessel config '{vessel_bundle.name}' is scale_type='full_scale_experimental'. "
                    f"Profile '{profile}' does not allow experimental full-scale configs. "
                    f"Set runtime.allow_experimental_full_scale=true in YAML or "
                    f"--allow-experimental-full-scale on CLI to override."
                )
        else:
            _logger.warning("Vessel config not found: %s, using defaults", vessel_config_path)

    effective_cfg = effective_cfg or {}
    # This enables --skip-statistics while keeping aggregate and HOCBF tables.
    statistics_cfg = dict(effective_cfg.get("statistics", {}) or {})
    if skip_statistics:
        statistics_cfg["skip"] = True
        effective_cfg["statistics"] = statistics_cfg

    _save_metadata(dirs, profile, controllers, seeds, scenarios, effective_cfg, vessel_bundle)

    config_hash = str(effective_cfg.get("config_hash") or _config_hash(effective_cfg))
    tasks = _build_tasks(scenarios, controllers, seeds, profile, config_hash)
    run_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    data_usage_rows: List[Dict[str, Any]] = []

    common_payload = {
        "tasks_dir": str(dirs["tasks"]),
        "raw_traces_dir": str(dirs["raw_traces"]),
        "save_traces": bool(save_traces),
        "save_traces_on_failure": bool(save_traces_on_failure),
        "trace_downsample": int(trace_downsample),
        "vessel_config_path": str(vessel_config_path) if vessel_config_path else None,
        "fail_fast_on_missing_data": bool(_FAIL_FAST_ON_MISSING_DATA),
    }

    pending_payloads: List[Dict[str, Any]] = []
    for task in tasks:
        cached = _load_finished_task(dirs["tasks"], task) if resume else None
        if cached is not None:
            if cached.get("status") == "ok" and isinstance(cached.get("row"), dict):
                run_rows.append(cached["row"])
                data_usage_rows.extend(cached.get("data_usage", []))
            elif cached.get("status") == "skipped":
                skipped.append(cached)
            elif cached.get("status") == "failed":
                # H5 fix: re-queue failed tasks on resume instead of silently dropping
                _logger.info("Re-queuing previously failed task: %s/%s seed=%s",
                             task.scenario, task.controller, task.seed)
                cached = None  # fall through to re-queue
            if cached is not None:
                continue
        payload = dict(common_payload)
        payload["task"] = {
            "scenario": task.scenario,
            "controller": task.controller,
            "seed": task.seed,
            "profile": task.profile,
            "config_hash": task.config_hash,
        }
        pending_payloads.append(payload)

    if pending_payloads:
        _logger.info(
            "Running %d pending tasks (%d already loaded, jobs=%d, backend=%s, resume=%s)",
            len(pending_payloads), len(run_rows), jobs, parallel_backend, resume,
        )

    if jobs > 1 and parallel_backend == "process" and pending_payloads:
        try:
            ctx = mp.get_context("forkserver")
        except (ValueError, RuntimeError):  # pragma: no cover - platform fallback
            ctx = mp.get_context("spawn")
        max_workers = min(jobs, len(pending_payloads))
        _TASK_TIMEOUT_S = 600.0  # 每个任务最多10分钟
        # 不使用 with 上下文管理器, 避免 shutdown(wait=True) 永久阻塞
        pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_worker_init)
        try:
            futures = {pool.submit(_run_task_worker, payload): payload for payload in pending_payloads}
            for fut in as_completed(futures):
                try:
                    result = fut.result(timeout=_TASK_TIMEOUT_S)
                except (TimeoutError, FutureTimeoutError):
                    payload = futures[fut]
                    task_payload = payload.get("task", {})
                    failure = {
                        "status": "failed",
                        "scenario_id": task_payload.get("scenario", {}).get("scenario_id", "unknown"),
                        "controller": task_payload.get("controller", "unknown"),
                        "seed": task_payload.get("seed", -1),
                        "error": f"Task timed out after {_TASK_TIMEOUT_S}s",
                    }
                    failures.append(failure)
                    _logger.error("Task TIMEOUT: %s/%s seed=%s",
                                  failure["scenario_id"], failure["controller"], failure["seed"])
                    fut.cancel()
                    continue
                except Exception as e:
                    payload = futures[fut]
                    task_payload = payload.get("task", {})
                    failure = {
                        "status": "failed",
                        "scenario_id": task_payload.get("scenario", {}).get("scenario_id", "unknown"),
                        "controller": task_payload.get("controller", "unknown"),
                        "seed": task_payload.get("seed", -1),
                        "error": repr(e),
                    }
                    failures.append(failure)
                    _logger.error("Parallel task failed: %s", failure)
                    continue
                if result.get("status") == "ok":
                    run_rows.append(result["row"])
                    data_usage_rows.extend(result.get("data_usage", []))
                elif result.get("status") == "skipped":
                    skipped.append(result)
                else:
                    failures.append(result)
        finally:
            # 强制关闭: cancel 未执行的任务, 不等待 hung workers
            pool.shutdown(wait=False, cancel_futures=True)
            # 终止所有 worker 进程 (防止 IPOPT/OSQP 卡死导致永久挂起)
            # _processes 是私有属性, 不同 Python 版本行为不同, 需安全访问
            procs_dict = getattr(pool, "_processes", None)
            if procs_dict:
                for proc in procs_dict.values():
                    try:
                        if proc.is_alive():
                            _logger.warning("Force-terminating hung worker PID=%s", proc.pid)
                            proc.terminate()
                            proc.join(timeout=5.0)
                            if proc.is_alive():
                                proc.kill()
                    except Exception:
                        pass  # Python 3.12+ 内部结构可能变化, 静默跳过
    else:
        for payload in pending_payloads:
            try:
                result = _run_task_worker(payload)
            except Exception as e:
                task_payload = payload.get("task", {})
                result = {
                    "status": "failed",
                    "scenario_id": task_payload.get("scenario", {}).get("scenario_id", "unknown"),
                    "controller": task_payload.get("controller", "unknown"),
                    "seed": task_payload.get("seed", -1),
                    "error": repr(e),
                }
                failures.append(result)
                _logger.error("Task failed: %s", result)
                continue
            if result.get("status") == "ok":
                run_rows.append(result["row"])
                data_usage_rows.extend(result.get("data_usage", []))
            elif result.get("status") == "skipped":
                skipped.append(result)
            else:
                failures.append(result)

    if skipped:
        _logger.info("Skipped %d tasks due to optional missing dependencies/controllers.", len(skipped))
    if failures:
        _logger.warning("%d tasks failed. See metadata/task_failure_report.json.", len(failures))
    (dirs["metadata"] / "task_failure_report.json").write_text(
        json.dumps({"failed": failures, "skipped": skipped}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Backward-compatible skip report expected by older tests and scripts.
    if skipped or failures:
        (dirs["metadata"] / "skip_report.json").write_text(
            json.dumps({
                "skipped": skipped,
                "failed": failures,
                "skipped_all": not bool(run_rows),
                "skipped_controllers": sorted({str(x.get("controller", "")) for x in skipped}),
                "reason": "missing dependencies/controllers or task failures",
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Save controller capability matrix even when no task ran.
    from .tables import generate_table2_controller_matrix
    generate_table2_controller_matrix().to_csv(dirs["summary"] / "controller_capability_matrix.csv", index=False)

    _write_actual_data_usage(dirs, data_usage_rows)
    _save_scale_analysis(dirs, scenarios, vessel_bundle)

    if not run_rows:
        _logger.warning("No experiments were executed.")
        return out_dir

    _save_summary_from_run_rows(
        run_rows,
        dirs,
        profile=profile,
        save_figures=save_figures,
        statistics_cfg=statistics_cfg,
    )
    return out_dir


# ---------- YAML 配置加载 ----------

_KNOWN_CONFIG_KEYS = {
    "profile", "seeds", "controllers", "output", "runtime",
    "scenarios", "protocol", "simulation", "vessel", "statistics", "data",
}


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(cfg)}")
    return cfg


def _check_unknown_keys(cfg: Dict[str, Any], strict: bool = True) -> None:
    """检查未知配置键。"""
    unknown = set(cfg.keys()) - _KNOWN_CONFIG_KEYS
    if unknown:
        msg = f"Unknown config keys: {unknown}"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)


def _config_hash(cfg: Dict[str, Any]) -> str:
    """计算配置的 SHA256 hash。"""
    raw = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def merge_config(yaml_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """合并 YAML 配置和 CLI 参数。CLI 参数优先 (None 表示未指定)。"""
    cfg = dict(yaml_cfg)

    # CLI 覆盖 (None = 未指定, 使用 YAML 值)
    if args.profile is not None:
        cfg["profile"] = args.profile
    elif "profile" not in cfg:
        cfg["profile"] = "smoke"

    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    elif "seeds" not in cfg:
        cfg["seeds"] = 2

    if args.controllers is not None:
        cfg["controllers"] = args.controllers
    elif "controllers" not in cfg:
        cfg["controllers"] = [
            "pid", "precision", "ice_aware", "full", "nmpc",
            "no_cbf", "no_cvar", "no_observer", "no_fallback",
        ]

    if args.no_traces or getattr(args, "no_save_traces", False):
        if "output" not in cfg:
            cfg["output"] = {}
        cfg["output"]["save_traces"] = False

    if getattr(args, "trace_downsample", None) is not None:
        if "output" not in cfg:
            cfg["output"] = {}
        cfg["output"]["trace_downsample"] = int(args.trace_downsample)

    if getattr(args, "save_traces_on_failure", False):
        if "output" not in cfg:
            cfg["output"] = {}
        cfg["output"]["save_traces_on_failure"] = True

    if getattr(args, "jobs", None) is not None:
        if "runtime" not in cfg:
            cfg["runtime"] = {}
        cfg["runtime"]["jobs"] = int(args.jobs)

    if getattr(args, "parallel_backend", None) is not None:
        if "runtime" not in cfg:
            cfg["runtime"] = {}
        cfg["runtime"]["parallel_backend"] = args.parallel_backend

    if getattr(args, "resume", False):
        if "runtime" not in cfg:
            cfg["runtime"] = {}
        cfg["runtime"]["resume"] = True

    if getattr(args, "skip_statistics", False):
        if "runtime" not in cfg:
            cfg["runtime"] = {}
        cfg["runtime"]["skip_statistics"] = True

    if getattr(args, "statistics_only", False):
        if "runtime" not in cfg:
            cfg["runtime"] = {}
        cfg["runtime"]["statistics_only"] = True

    if args.out is not None:
        if "output" not in cfg:
            cfg["output"] = {}
        cfg["output"]["root"] = str(args.out)

    return cfg


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCI一区投稿级冰区 DP 实验矩阵")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seeds", type=int, default=None, help="Smoke: 1-3; paper: 30-100")
    parser.add_argument("--controllers", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--no-save-traces", action="store_true", help="Alias for --no-traces")
    parser.add_argument("--save-traces-on-failure", action="store_true", default=False,
                        help="Save full traces only for failed/safety-filter-infeasible tasks when save_traces=false")
    parser.add_argument("--trace-downsample", type=int, default=None,
                        help="Write every Nth trace row when traces are enabled")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Number of scenario-controller-seed worker processes; use 24-28 on 32-core EPYC")
    parser.add_argument("--parallel-backend", choices=["serial", "process"], default=None,
                        help="Parallel backend; process is recommended for CPU-bound paper runs")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Reuse completed raw/tasks/*.done payloads")
    parser.add_argument("--skip-statistics", action="store_true", default=False,
                        help="Run simulations and aggregate metrics, but defer Wilcoxon/bootstrap statistics")
    parser.add_argument("--statistics-only", action="store_true", default=False,
                        help="Recompute summary/statistics from existing raw/summary per_seed_metrics")
    parser.add_argument("--allow-experimental-full-scale", action="store_true", default=False,
                        help="Allow full_scale_experimental vessel config in paper profiles")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Strict mode: error on unknown config keys (default: True)")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    # 加载 YAML 配置 (如果有)
    yaml_cfg: Dict[str, Any] = {}
    if args.config is not None:
        yaml_cfg = load_yaml_config(args.config)
        _check_unknown_keys(yaml_cfg, strict=args.strict)

    # 合并配置
    cfg = merge_config(yaml_cfg, args)

    profile = cfg.get("profile", "smoke")
    seeds = int(cfg.get("seeds", 2))
    controllers = cfg.get("controllers", [
        "pid", "precision", "ice_aware", "full", "nmpc",
        "no_cbf", "no_cvar", "no_observer", "no_fallback",
    ])
    output_cfg = cfg.get("output", {})
    save_traces = output_cfg.get("save_traces", True)
    save_figures = output_cfg.get("save_figures", True)
    save_traces_on_failure = bool(output_cfg.get("save_traces_on_failure", False))
    trace_downsample = int(output_cfg.get("trace_downsample", 1) or 1)

    # 输出目录
    if args.out is not None:
        out = args.out
    elif "root" in output_cfg:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(output_cfg["root"]) / f"{profile}_{stamp}"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("results") / "sci1_submission" / f"{profile}_{stamp}"

    # 场景过滤 (include_ids 优先于 include_groups)
    scenario_cfg = cfg.get("scenarios", {})
    include_groups = scenario_cfg.get("include_groups", None) if isinstance(scenario_cfg, dict) else None
    include_ids = scenario_cfg.get("include_ids", None) if isinstance(scenario_cfg, dict) else None

    # 船舶配置路径 (从 YAML vessel 字段读取)
    vessel_cfg = cfg.get("vessel", {})
    vessel_config_path = None
    if isinstance(vessel_cfg, dict):
        vessel_config_path = vessel_cfg.get("config_path") or vessel_cfg.get("config")
    elif isinstance(vessel_cfg, str):
        vessel_config_path = vessel_cfg

    # data fail-fast 配置 (兼容 fail_fast 和 fail_fast_on_missing_data)
    runtime_cfg = cfg.get("runtime", {})
    if isinstance(runtime_cfg, dict):
        fail_fast_on_missing_data = runtime_cfg.get("fail_fast_on_missing_data",
                                                     runtime_cfg.get("fail_fast", False))
        allow_explicit = runtime_cfg.get("allow_experimental_full_scale", False)
        jobs = int(runtime_cfg.get("jobs", runtime_cfg.get("parallel_recommended_jobs", 1)) or 1)
        parallel_backend = str(runtime_cfg.get("parallel_backend", "process" if jobs > 1 else "serial"))
        resume = bool(runtime_cfg.get("resume", False))
        skip_statistics = bool(runtime_cfg.get("skip_statistics", False))
        statistics_only = bool(runtime_cfg.get("statistics_only", False))
    else:
        fail_fast_on_missing_data = False
        allow_explicit = False
        jobs = 1
        parallel_backend = "serial"
        resume = False
        skip_statistics = False
        statistics_only = False
    # CLI 覆盖
    if getattr(args, 'allow_experimental_full_scale', False):
        allow_explicit = True

    effective_cfg = {
        "profile": profile,
        "seeds": seeds,
        "controllers": controllers,
        "include_groups": include_groups,
        "vessel_config_path": vessel_config_path,
        "fail_fast_on_missing_data": fail_fast_on_missing_data,
        "allow_experimental_full_scale": allow_explicit,
        "statistics": cfg.get("statistics", {}),
        "data": cfg.get("data", {}),
        "output": {
            "root": str(out),
            "save_traces": save_traces,
            "save_figures": save_figures,
            "save_traces_on_failure": save_traces_on_failure,
            "trace_downsample": trace_downsample,
        },
        "runtime": {
            "jobs": jobs,
            "parallel_backend": parallel_backend,
            "resume": resume,
            "skip_statistics": skip_statistics,
            "statistics_only": statistics_only,
        },
        "config_hash": _config_hash(cfg),
        "source_config": str(args.config) if args.config else "cli_only",
    }

    if statistics_only:
        result_dir = _statistics_only(out, cfg)
    else:
        result_dir = run_experiments(
            profile, seeds, controllers, out,
            save_traces=save_traces,
            save_figures=save_figures,
            effective_cfg=effective_cfg,
            include_groups=include_groups,
            include_ids=include_ids,
            vessel_config_path=vessel_config_path,
            jobs=jobs,
            parallel_backend=parallel_backend,
            resume=resume,
            save_traces_on_failure=save_traces_on_failure,
            trace_downsample=trace_downsample,
            skip_statistics=skip_statistics,
        )
    _logger.info("SCI1 experiment outputs written to: %s", result_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
