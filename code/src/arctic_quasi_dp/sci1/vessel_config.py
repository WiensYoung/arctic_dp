"""船舶配置加载模块 — 从 YAML 构造 VesselParams、ThrusterConfig 和控制器参数。

使用：
    from arctic_quasi_dp.sci1.vessel_config import load_vessel_config
    vc = load_vessel_config("configs/vessels/xuelong2_like.yaml")
    params = vc.vessel_params
    thruster = vc.thruster_config
    max_force = vc.max_force
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .sim_loop import VesselParams
from .thruster import ThrusterConfig, ThrusterUnit


@dataclass
class VesselConfigBundle:
    """从 YAML 加载的完整船舶配置包。"""
    name: str = "unknown"
    scale_type: str = "proxy_scale"  # "proxy_scale" or "full_scale"
    source_note: str = ""
    vessel_params: VesselParams = field(default_factory=VesselParams)
    thruster_config: Optional[ThrusterConfig] = None
    max_force: float = 3000.0
    max_moment: float = 100000.0
    raw_yaml: Dict[str, Any] = field(default_factory=dict)


def load_vessel_config(path: str | Path) -> VesselConfigBundle:
    """从 YAML 文件加载船舶配置。

    构造:
    - VesselParams (质量、惯性、阻尼、冰力参数)
    - ThrusterConfig (推进器布局和容量)
    - max_force / max_moment (控制器力限)

    Args:
        path: YAML 配置文件路径

    Returns:
        VesselConfigBundle 包含所有派生配置
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vessel config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Vessel config must be a YAML mapping, got {type(cfg)}")

    bundle = VesselConfigBundle()
    bundle.raw_yaml = cfg
    bundle.name = cfg.get("name", path.stem)

    # 读取冰力参数 (始终从 YAML 读取)
    ice_strength = float(cfg.get("ice_crushing_strength_mpa", 0.0003))

    # 读取尺度类型 (优先从 YAML 显式字段, 否则从 ice_crushing_strength 推断)
    explicit_scale = cfg.get("scale_type")
    if explicit_scale:
        bundle.scale_type = explicit_scale
    else:
        bundle.scale_type = "full_scale_experimental" if ice_strength >= 1.0 else "proxy_scale"
    bundle.source_note = cfg.get("source_note", "")

    # 构造 VesselParams
    mass = cfg.get("mass_kg", 500000.0)
    Iz = cfg.get("Iz_kgm2")
    if Iz is None:
        # 从长度和质量估算: Iz ≈ 0.15 * L² * m
        length = cfg.get("length_m", 122.5)
        Iz = 0.15 * length ** 2 * mass

    bundle.vessel_params = VesselParams(
        mass=mass,
        Izz=float(Iz),
        Xu=float(cfg.get("Xu", 500.0)),
        Yv=float(cfg.get("Yv", 800.0)),
        Nr=float(cfg.get("Nr", 200000.0)),
        Xu_abs=float(cfg.get("Xu_abs", 200.0)),
        Yv_abs=float(cfg.get("Yv_abs", 300.0)),
        Nr_abs=float(cfg.get("Nr_abs", 50000.0)),
        length=float(cfg.get("length_m", 122.5)),
        beam=float(cfg.get("beam_m", 22.0)),
        ice_crushing_strength_mpa=float(ice_strength),
        ice_structure_factor=float(cfg.get("ice_structure_factor", 0.45)),
        waterline_angle_deg=float(cfg.get("waterline_angle_deg", 30.0)),
    )

    # 推进器力限 (从 actuator_limits 读取, 或从 ice_strength 推导)
    act_limits = cfg.get("actuator_limits", {})
    if "max_force_n" in act_limits:
        bundle.max_force = float(act_limits["max_force_n"])
    else:
        # 基于冰力推导: 中等冰况力的 2-3 倍作为控制器力限
        from .ice_force_common import compute_ice_force_body_from_dict
        test_ice = {"concentration": 0.6, "thickness": 1.0, "drift_speed": 0.4, "drift_direction": 135.0}
        f = compute_ice_force_body_from_dict(test_ice, 0.0, bundle.vessel_params.ice_crushing_strength_mpa,
                                              bundle.vessel_params.beam, bundle.vessel_params.length)
        ice_mag = float((f[0] ** 2 + f[1] ** 2) ** 0.5)
        bundle.max_force = max(3000.0, 2.5 * ice_mag)

    if "max_moment_nm" in act_limits:
        bundle.max_moment = float(act_limits["max_moment_nm"])
    else:
        bundle.max_moment = bundle.max_force * bundle.vessel_params.length * 0.5

    # 推进器配置 (如果 YAML 中有 thrusters 字段)
    if "thrusters" in cfg:
        units = []
        for t in cfg["thrusters"]:
            units.append(ThrusterUnit(
                name=t["name"],
                x=float(t["x"]),
                y=float(t["y"]),
                max_thrust=float(t["max_thrust"]),
                min_thrust=float(t.get("min_thrust", 0.0)),
                azimuth=float(t.get("azimuth", 0.0)),
                azimuthable=bool(t.get("azimuthable", False)),
                efficiency=float(t.get("efficiency", 1.0)),
            ))
        bundle.thruster_config = ThrusterConfig(
            name=cfg.get("name", "custom"),
            thrusters=units,
            max_total_power_kw=float(cfg.get("max_total_power_kw", 0.0)),
        )

    return bundle
