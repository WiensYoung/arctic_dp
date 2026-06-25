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
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

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
from .scenarios import build_sci1_scenarios, SCI1Scenario
from .ice_schedule import ConstantIce, LinearRampIce
from .sim_loop import run_simulation
from .thruster import ThrusterConfig, ThrusterDegradationProfile


# ---------- NMPC 可用性检测 ----------

_NMPC_AVAILABLE = False
try:
    from .nmpc_controller import NMPCIceController, NMPCParams
    _NMPC_AVAILABLE = True
except ImportError:
    pass


# ---------- 推进器配置映射 ----------

_THRUSTER_CONFIGS = {
    "generic_dp": ThrusterConfig.generic_dp_vessel(),
    "xuelong2": ThrusterConfig.vessel_xuelong2(),
    "none": None,
}

_DEGRADATION_PROFILES = {
    "no_fault": ThrusterDegradationProfile.no_fault(),
    "bow_degraded_0.5": ThrusterDegradationProfile.bow_degradation(0.5),
    "severe": ThrusterDegradationProfile.severe_degradation(),
}


# ---------- 控制器构建 ----------

def build_controller(name: str):
    """构建控制器实例。NMPC 不可用时抛出明确错误。"""
    name = name.lower()
    if name == "pid":
        return PIDController(PIDParams(Kp_pos=150, Kd_pos=90, Ki_pos=0.2, Kp_heading=600, Kd_heading=260, Ki_heading=0.5))
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
            raise ImportError(
                "CasADi is required for NMPC controller. "
                "Install with: pip install casadi"
            )
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
        return ModeSupervisedIceDPController(params=SupervisorParams(high_risk_enter=2.0, extreme_risk_enter=3.0))
    raise ValueError(f"Unknown controller: {name}")


# ---------- 冰况调度 ----------

def _make_ice_schedule(scenario: SCI1Scenario):
    """根据场景创建冰况调度器。"""
    if not scenario.ice_time_varying:
        return ConstantIce(
            scenario.ice_concentration, scenario.ice_thickness,
            scenario.ice_drift_speed, scenario.ice_drift_direction,
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


# ---------- 元数据 ----------

def _metadata() -> Dict[str, str]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": os.sys.version.split()[0],
        "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
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
) -> "tuple[pd.DataFrame, float]":
    """运行单个 场景×控制器×seed 的仿真。统一使用 sim_loop。"""
    ice_schedule = _make_ice_schedule(scenario)
    thruster_config = _get_thruster_config(scenario)
    degradation = _get_degradation_profile(scenario)

    if hasattr(controller, 'set_cvar_seed'):
        controller.set_cvar_seed(20260625 + seed)

    log = run_simulation(
        controller=controller,
        duration=scenario.duration,
        dt=scenario.dt,
        ice_schedule=ice_schedule,
        thruster_config=thruster_config,
        degradation_profile=degradation,
        target_x=scenario.target_x,
        target_y=scenario.target_y,
        target_psi=scenario.target_psi,
        safe_region_radius=scenario.safe_region_radius,
        seed=20260625 + seed,
        log_interval=1,
    )
    df = log.to_dataframe()
    return df, scenario.dt


# ---------- 主实验循环 ----------

def run_experiments(
    profile: str,
    seeds: int,
    controllers: List[str],
    out_dir: Path,
    save_traces: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    if save_traces:
        traces_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "data_manifest.json")
    scenarios = build_sci1_scenarios(profile)
    (out_dir / "scenario_manifest.json").write_text(
        json.dumps({"metadata": _metadata(), "profile": profile, "scenarios": [s.to_dict() for s in scenarios]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_rows: List[Dict] = []
    skipped: List[str] = []

    for scenario in scenarios:
        for ctrl_name in controllers:
            for seed in range(seeds):
                # 构建控制器 (NMPC 不可用时跳过)
                try:
                    controller = build_controller(ctrl_name)
                except ImportError as e:
                    key = f"{scenario.scenario_id}:{ctrl_name}"
                    if key not in skipped:
                        skipped.append(key)
                        print(f"  SKIP {ctrl_name} (missing dependency: {e})")
                    continue

                df, dt = _run_single(scenario, ctrl_name, controller, seed, profile)

                df.insert(0, "seed", seed)
                df.insert(0, "controller", ctrl_name)
                df.insert(0, "scenario_id", scenario.scenario_id)
                if save_traces:
                    trace_path = traces_dir / f"{scenario.scenario_id}__{ctrl_name}__seed{seed}.csv"
                    df.to_csv(trace_path, index=False)
                run_rows.append(summarize_run(df, scenario.scenario_id, ctrl_name, seed, dt, safe_region_radius=scenario.safe_region_radius))

    if skipped:
        print(f"  Skipped {len(skipped)} scenario-controller combinations due to missing dependencies.")

    save_tables(run_rows, out_dir)
    make_all_figures(out_dir / "aggregate_metrics_ci95.csv", out_dir / "figures", control_period_ms=100.0)
    return out_dir


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    # 默认控制器列表 (NMPC 在列，不可用时自动跳过)
    default_controllers = [
        "pid", "precision", "ice_aware", "full", "nmpc",
        "no_cbf", "no_cvar", "no_observer", "no_fallback",
    ]
    parser = argparse.ArgumentParser(description="Run SCI一区投稿级冰区 DP 实验矩阵")
    parser.add_argument("--profile", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--seeds", type=int, default=2, help="Smoke: 1-3; paper: 30-100")
    parser.add_argument("--controllers", nargs="+", default=default_controllers)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-traces", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or Path("results") / "sci1_submission" / f"{args.profile}_{stamp}"
    result_dir = run_experiments(
        args.profile, args.seeds, args.controllers, out,
        save_traces=not args.no_traces,
    )
    print(f"SCI1 experiment outputs written to: {result_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
