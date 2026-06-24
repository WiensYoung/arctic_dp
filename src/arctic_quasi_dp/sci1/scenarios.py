"""SCI一区投稿实验场景矩阵。

场景覆盖：开阔水域精确定位、冰区扰动、推进器退化/故障、传感器退化、
安全降级与实时性。每个场景都可被 Monte Carlo runner 调用。

新增场景：
- B4: 非平稳冰况 (时变密集度/漂移速度)
- F1: 边缘冰区 (MIZ) 典型作业场景

推进器配置：
- 所有场景默认使用 generic_dp_vessel 推进器配置
- C1 场景使用 bow_degradation 退化配置
- A 组场景不使用推进器分配 (开阔水域无冰力)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SCI1Scenario:
    scenario_id: str
    group: str
    description: str
    duration: float
    dt: float
    target_x: float = 0.0
    target_y: float = 0.0
    target_psi: float = 0.0
    ice_concentration: float = 0.0
    ice_thickness: float = 0.0
    ice_drift_speed: float = 0.0
    ice_drift_direction: float = 0.0
    safe_region_radius: float = 10.0
    fault_profile: Optional[Dict[str, Any]] = None
    evidence_level: str = "literature-calibrated synthetic"
    primary_claim: str = ""
    # 时变冰况参数 (可选)
    ice_time_varying: bool = False
    ice_concentration_initial: float = 0.0
    ice_concentration_final: float = 0.0
    ice_thickness_initial: float = 0.0
    ice_thickness_final: float = 0.0
    ice_drift_speed_initial: float = 0.0
    ice_drift_speed_final: float = 0.0
    ice_drift_direction_initial: float = 0.0
    ice_drift_direction_final: float = 0.0
    # 推进器配置
    thruster_config_name: str = "generic_dp"  # "generic_dp", "xuelong2", "none"
    degradation_name: str = "no_fault"         # "no_fault", "bow_degraded_0.5", "severe"

    def to_config_kwargs(self) -> Dict[str, Any]:
        return {
            "duration": self.duration,
            "dt": self.dt,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_psi": self.target_psi,
            "ice_concentration": self.ice_concentration,
            "ice_thickness": self.ice_thickness,
            "ice_drift_speed": self.ice_drift_speed,
            "ice_drift_direction": self.ice_drift_direction,
            "verbose": False,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def ice_conditions_at(self, t: float) -> Dict[str, float]:
        """返回时刻 t 的冰况参数。常数场景直接返回固定值，时变场景线性插值。"""
        if not self.ice_time_varying:
            return {
                "concentration": self.ice_concentration,
                "thickness": self.ice_thickness,
                "drift_speed": self.ice_drift_speed,
                "drift_direction": self.ice_drift_direction,
            }
        frac = float(np.clip(t / max(self.duration, 1e-6), 0.0, 1.0))
        return {
            "concentration": self.ice_concentration_initial + frac * (self.ice_concentration_final - self.ice_concentration_initial),
            "thickness": self.ice_thickness_initial + frac * (self.ice_thickness_final - self.ice_thickness_initial),
            "drift_speed": self.ice_drift_speed_initial + frac * (self.ice_drift_speed_final - self.ice_drift_speed_initial),
            "drift_direction": self.ice_drift_direction_initial + frac * (self.ice_drift_direction_final - self.ice_drift_direction_initial),
        }


def build_sci1_scenarios(profile: str = "smoke") -> List[SCI1Scenario]:
    """Build scenario list.

    profile="smoke" uses short durations for CI; profile="paper" is intended for
    paper runs with 30-100 seeds per scenario.
    """
    short = profile == "smoke"
    dur = 20.0 if short else 300.0
    dt = 0.2 if short else 0.1
    return [
        # === A 组: 开阔水域精确定位基线 (无推进器分配) ===
        SCI1Scenario(
            scenario_id="A1_open_water_station_keeping",
            group="A_precision",
            description="Open-water station keeping sanity check for Precision DP.",
            duration=dur, dt=dt, ice_concentration=0.0, ice_thickness=0.0,
            ice_drift_speed=0.0, ice_drift_direction=0.0,
            thruster_config_name="none",
            primary_claim="Proposed architecture does not sacrifice normal DP precision.",
        ),
        SCI1Scenario(
            scenario_id="A2_low_speed_offset_tracking",
            group="A_precision",
            description="Low-speed offset target used as DP tracking proxy.",
            duration=dur, dt=dt, target_x=4.0, target_y=-3.0, target_psi=5.0,
            ice_concentration=0.0, ice_thickness=0.0, ice_drift_speed=0.0,
            thruster_config_name="none",
            primary_claim="Position/heading tracking remains competitive in benign conditions.",
        ),
        # === B 组: 冰区增强 (使用推进器分配) ===
        SCI1Scenario(
            scenario_id="B1_moderate_drifting_ice",
            group="B_ice_enhancement",
            description="Moderate drifting broken ice; core ice-aware precision DP test.",
            duration=dur, dt=dt, ice_concentration=0.45, ice_thickness=0.7,
            ice_drift_speed=0.25, ice_drift_direction=125.0,
            primary_claim="At comparable precision, ice-aware risk control lowers tail risk.",
        ),
        SCI1Scenario(
            scenario_id="B2_ice_concentration_jump",
            group="B_ice_enhancement",
            description="Concentration jump proxy; use fault manager in full integration.",
            duration=dur, dt=dt, ice_concentration=0.62, ice_thickness=1.0,
            ice_drift_speed=0.35, ice_drift_direction=145.0,
            primary_claim="Risk layer improves recovery after nonstationary ice disturbance.",
        ),
        SCI1Scenario(
            scenario_id="B3_drift_direction_change",
            group="B_ice_enhancement",
            description="Cross-drift ice forcing where heading-to-ice matters.",
            duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
            ice_drift_speed=0.42, ice_drift_direction=90.0,
            primary_claim="Ice-aware DP reduces cross-ice load without abandoning precision.",
        ),
        # B4: 非平稳冰况 — 密集度、厚度和漂移速度随时间变化
        SCI1Scenario(
            scenario_id="B4_nonstationary_ice_ramp",
            group="B_ice_enhancement",
            description=(
                "Nonstationary ice: concentration ramps from 0.3 to 0.8, thickness from 0.7 to 1.1, "
                "drift speed from 0.15 to 0.55 over the scenario duration."
            ),
            duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
            ice_drift_speed=0.35, ice_drift_direction=135.0,
            ice_time_varying=True,
            ice_concentration_initial=0.3, ice_concentration_final=0.8,
            ice_thickness_initial=0.7, ice_thickness_final=1.1,
            ice_drift_speed_initial=0.15, ice_drift_speed_final=0.55,
            ice_drift_direction_initial=120.0, ice_drift_direction_final=160.0,
            primary_claim="Supervisor mode switching adapts to time-varying ice conditions.",
        ),
        # === C 组: 故障容错 (推进器退化) ===
        SCI1Scenario(
            scenario_id="C1_thruster_degradation_proxy",
            group="C_fault_tolerance",
            description="High ice + thruster-degradation; force the allocator near saturation.",
            duration=dur, dt=dt, ice_concentration=0.72, ice_thickness=1.25,
            ice_drift_speed=0.5, ice_drift_direction=160.0,
            thruster_config_name="generic_dp",
            degradation_name="bow_degraded_0.5",
            primary_claim="Safety degradation reduces loss-of-position probability under degraded thrust.",
        ),
        # === D 组: 安全降级 ===
        SCI1Scenario(
            scenario_id="D1_extreme_ice_escape",
            group="D_safety_degradation",
            description="Extreme ice loading where quasi-DP and ice-vaning/escape should activate.",
            duration=dur, dt=dt, ice_concentration=0.86, ice_thickness=1.55,
            ice_drift_speed=0.65, ice_drift_direction=180.0,
            safe_region_radius=12.0,
            primary_claim="Fallback prevents unsafe force saturation and reduces safety violations.",
        ),
        # === E 组: 实时性 ===
        SCI1Scenario(
            scenario_id="E1_runtime_feasibility",
            group="E_realtime",
            description="Runtime feasibility profile; same physical setting as B1 but all solver times are logged.",
            duration=dur, dt=dt, ice_concentration=0.45, ice_thickness=0.7,
            ice_drift_speed=0.25, ice_drift_direction=125.0,
            primary_claim="P95 solve time remains below control update period.",
        ),
        # === F 组: 边缘冰区 (MIZ) ===
        SCI1Scenario(
            scenario_id="F1_marginal_ice_zone",
            group="F_miz",
            description=(
                "Marginal ice zone (MIZ) scenario: moderate concentration with high variability, "
                "thin first-year ice, and moderate drift. Typical of Arctic operational conditions."
            ),
            duration=dur, dt=dt, ice_concentration=0.35, ice_thickness=0.4,
            ice_drift_speed=0.30, ice_drift_direction=75.0,
            safe_region_radius=10.0,
            primary_claim="Ice-aware DP maintains precision in marginal ice zone conditions.",
        ),
    ]
