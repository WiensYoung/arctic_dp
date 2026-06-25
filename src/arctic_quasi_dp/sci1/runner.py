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
import json
import os
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

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

# ---------- 控制器能力矩阵 ----------

_CONTROLLER_CAPABILITIES = {
    "pid": {"observer": False, "cvar": False, "cbf": False, "thruster_deg": False,
            "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "smc": {"observer": False, "cvar": False, "cbf": False, "thruster_deg": False,
            "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "precision": {"observer": False, "cvar": False, "cbf": False, "thruster_deg": False,
                  "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "ice_aware": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": False,
                  "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "quasi_dp": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": False,
                 "mode_supervisor": False, "quasi_dp": True, "escape": False, "oracle": False, "casadi": False},
    "escape": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": False,
               "mode_supervisor": False, "quasi_dp": False, "escape": True, "oracle": False, "casadi": False},
    "full": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": True,
             "mode_supervisor": True, "quasi_dp": True, "escape": True, "oracle": False, "casadi": False},
    "nmpc": {"observer": False, "cvar": False, "cbf": True, "thruster_deg": False,
             "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": True},
    "no_cbf": {"observer": True, "cvar": True, "cbf": False, "thruster_deg": True,
               "mode_supervisor": True, "quasi_dp": True, "escape": True, "oracle": False, "casadi": False},
    "no_cvar": {"observer": True, "cvar": False, "cbf": True, "thruster_deg": True,
                "mode_supervisor": True, "quasi_dp": True, "escape": True, "oracle": False, "casadi": False},
    "no_observer": {"observer": False, "cvar": True, "cbf": True, "thruster_deg": True,
                    "mode_supervisor": True, "quasi_dp": True, "escape": True, "oracle": False, "casadi": False},
    "no_fallback": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": True,
                    "mode_supervisor": True, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "oracle_full": {"observer": True, "cvar": True, "cbf": True, "thruster_deg": True,
                    "mode_supervisor": True, "quasi_dp": True, "escape": True, "oracle": True, "casadi": False},
    "lqg": {"observer": True, "cvar": False, "cbf": False, "thruster_deg": False,
            "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": False},
    "dob_nmpc": {"observer": True, "cvar": False, "cbf": True, "thruster_deg": False,
                 "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": True},
    "robust_mpc": {"observer": True, "cvar": False, "cbf": True, "thruster_deg": False,
                   "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": True},
    "tube_mpc": {"observer": True, "cvar": False, "cbf": True, "thruster_deg": False,
                 "mode_supervisor": False, "quasi_dp": False, "escape": False, "oracle": False, "casadi": True},
}


def build_controller(name: str):
    """构建控制器实例。未实现或缺依赖时抛出明确错误。"""
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
        return ModeSupervisedIceDPController(params=SupervisorParams(high_risk_enter=2.0, extreme_risk_enter=3.0))
    if name == "oracle_full":
        # oracle_full uses true ice (upper-bound ablation only)
        ctrl = ModeSupervisedIceDPController()
        ctrl._oracle_mode = True  # Flag for oracle ice access
        return ctrl
    if name in ("lqg", "dob_nmpc", "robust_mpc", "tube_mpc"):
        raise NotImplementedError(
            f"{name} is not yet implemented. "
            f"It will be skipped. See skip_report.json for details."
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
                try:
                    controller = build_controller(ctrl_name)
                except (ImportError, NotImplementedError) as e:
                    key = f"{scenario.scenario_id}:{ctrl_name}"
                    if key not in skipped:
                        skipped.append(key)
                        print(f"  SKIP {ctrl_name}: {e}")
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
        print(f"  Skipped {len(skipped)} scenario-controller combinations.")

    if skipped:
        (out_dir / "skip_report.json").write_text(
            json.dumps({"skipped": skipped, "reason": "missing dependencies or not implemented"}, indent=2),
            encoding="utf-8",
        )

    # Save controller capability matrix
    save_controller_capability_matrix(out_dir)

    if not run_rows:
        print("  WARNING: No experiments were executed. "
              "Check controller dependencies, e.g. install casadi for nmpc.")
        (out_dir / "skip_report.json").write_text(
            json.dumps({
                "skipped_all": True,
                "skipped_controllers": controllers,
                "reason": "All controllers were skipped due to missing dependencies.",
            }, indent=2),
            encoding="utf-8",
        )
        return out_dir

    save_tables(run_rows, out_dir)
    make_all_figures(out_dir / "aggregate_metrics_ci95.csv", out_dir / "figures", control_period_ms=100.0)
    return out_dir


# ---------- YAML 配置加载 ----------

_KNOWN_CONFIG_KEYS = {
    "profile", "seeds", "controllers", "output", "runtime",
    "scenarios", "protocol", "simulation", "vessel",
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

    if args.no_traces:
        if "output" not in cfg:
            cfg["output"] = {}
        cfg["output"]["save_traces"] = False

    if args.out is not None:
        if "output" not in cfg:
            cfg["output"] = {}
        cfg["output"]["root"] = str(args.out)

    return cfg


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCI一区投稿级冰区 DP 实验矩阵")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--profile", choices=["smoke", "paper"], default=None)
    parser.add_argument("--seeds", type=int, default=None, help="Smoke: 1-3; paper: 30-100")
    parser.add_argument("--controllers", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-traces", action="store_true")
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

    # 输出目录
    if args.out is not None:
        out = args.out
    elif "root" in output_cfg:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(output_cfg["root"]) / f"{profile}_{stamp}"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("results") / "sci1_submission" / f"{profile}_{stamp}"

    # 保存 effective config
    out.mkdir(parents=True, exist_ok=True)
    effective_cfg = {
        "profile": profile,
        "seeds": seeds,
        "controllers": controllers,
        "output": {"root": str(out), "save_traces": save_traces, "save_figures": save_figures},
        "config_hash": _config_hash(cfg),
        "source_config": str(args.config) if args.config else "cli_only",
    }
    (out / "effective_config.json").write_text(
        json.dumps(effective_cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result_dir = run_experiments(
        profile, seeds, controllers, out,
        save_traces=save_traces,
    )
    print(f"SCI1 experiment outputs written to: {result_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
