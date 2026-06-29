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

import re

import numpy as np


def scenario_group_from_id(scenario_id: str) -> str:
    """从 scenario_id 提取字母组标识。

    'A1_open_water_station_keeping' → 'A'
    'G3_model_sensitivity' → 'G'
    'H1_copernicus_replay' → 'H'
    """
    m = re.match(r"^([A-Z])\d+", scenario_id)
    if m:
        return m.group(1)
    return scenario_id.split("_", 1)[0]


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
    # 传感器退化配置 (E 组场景)
    position_sensor_config: Optional[Dict[str, Any]] = None   # e.g. {"gaussian_std": 2.5}
    heading_sensor_config: Optional[Dict[str, Any]] = None    # e.g. {"gaussian_std": 0.01, "bias": 0.05}
    ice_sensor_config: Optional[Dict[str, Any]] = None        # e.g. {"concentration_std": 0.15}
    # 冰力模型选择 (G 组场景)
    ice_load_model_name: str = "default"  # "default", "empirical", "stochastic", "benchmark"
    # 风场配置 (A2 场景)
    wind_config: Optional[Dict[str, Any]] = None  # e.g. {"type": "constant", "u10": 8.0, "v10": 5.0}
    # 推进器速率限制 (C5/C6 场景)
    max_azimuth_rate: float = 0.0   # rad/s, 0=无限制
    max_thrust_rate: float = 0.0    # N/s, 0=无限制
    # 冰况调度类型
    ice_schedule_type: str = "default"  # "default", "linear", "step"
    # 数据驱动场景 (可选)
    data_driven: bool = False           # 是否使用 DataDrivenIceSchedule
    data_nc_path: Optional[str] = None  # NetCDF 数据路径 (SIC/SIT)
    drift_nc_path: Optional[str] = None  # 独立冰漂移数据路径 (NSIDC-0116), 当主文件不含漂移时使用
    data_lat: float = 80.0              # 数据提取纬度
    data_lon: float = 0.0               # 数据提取经度
    data_source_type: str = "none"      # mock_fixture / real_subset / none
    data_provider: str = ""             # Copernicus Marine / ERA5 / project fixture
    data_product_id: str = ""           # product id for provenance
    allow_mock_fixture: bool = True      # real replay configs must set this false
    # 动态目标跟踪 (A5 场景)
    target_x_final: Optional[float] = None  # 最终目标 x (None=不使用动态目标)
    target_y_final: Optional[float] = None
    target_psi_final: Optional[float] = None
    target_change_time: float = 0.0     # 目标切换时刻 (s)
    # 推进器方位角锁定 (C4 场景)
    azimuth_locked_angle_deg: Optional[float] = None  # None=不锁定

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
        # 角度插值: 使用最短路径 (处理 350° → 10° 绕回 0° 的情况)
        dir_diff = (self.ice_drift_direction_final - self.ice_drift_direction_initial + 180.0) % 360.0 - 180.0
        dir_interp = (self.ice_drift_direction_initial + frac * dir_diff) % 360.0
        return {
            "concentration": self.ice_concentration_initial + frac * (self.ice_concentration_final - self.ice_concentration_initial),
            "thickness": self.ice_thickness_initial + frac * (self.ice_thickness_final - self.ice_thickness_initial),
            "drift_speed": self.ice_drift_speed_initial + frac * (self.ice_drift_speed_final - self.ice_drift_speed_initial),
            "drift_direction": dir_interp,
        }


def build_sci1_scenarios(profile: str = "smoke") -> List[SCI1Scenario]:
    """Build scenario list.

    profile="smoke" uses short durations for CI; profile="paper" is intended for
    paper runs with 30-100 seeds per scenario.

    Groups:
    A: Open-water Precision DP (5 scenarios)
    B: Ice-aware DP under ice disturbance (7 scenarios)
    C: Thruster degradation and fault tolerance (8 scenarios)
    D: Safety degradation and fallback (4 scenarios)
    E: Sensor degradation and observer robustness (8 scenarios)
    F: Runtime feasibility (5 scenarios)
    G: Ice model sensitivity and data-source robustness (7 scenarios)
    H: Data-driven Copernicus/ERA5 replay (5 scenarios)
    I: Safety filter method validation (5 scenarios)
    """
    short_profiles = {"smoke", "artifact_check", "method_smoke", "scale_comparison"}
    short = profile in short_profiles
    dur = 20.0 if short else 300.0
    dt = 0.2 if short else 0.1
    scenarios: List[SCI1Scenario] = []

    # === A 组: Open-water Precision DP ===
    scenarios.append(SCI1Scenario(
        scenario_id="A1_open_water_station_keeping", group="A_precision",
        description="Open-water station keeping. Sanity check for Precision DP baseline.",
        duration=dur, dt=dt, ice_concentration=0.0, ice_thickness=0.0,
        ice_drift_speed=0.0, ice_drift_direction=0.0, thruster_config_name="none",
        primary_claim="Proposed architecture does not sacrifice normal DP precision.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="A2_wind_current_disturbance", group="A_precision",
        description="Open water with wind disturbance (current model not implemented).",
        duration=dur, dt=dt, target_x=0.0, target_y=0.0, target_psi=0.0,
        ice_concentration=0.0, ice_thickness=0.0, ice_drift_speed=0.0,
        thruster_config_name="generic_dp",
        wind_config={"type": "constant", "u10": 8.0, "v10": 5.0},
        primary_claim="Precision DP rejects wind/current disturbance without ice enhancement.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="A3_heading_keeping", group="A_precision",
        description="Heading-only keeping at fixed position.",
        duration=dur, dt=dt, target_psi=45.0,
        ice_concentration=0.0, ice_thickness=0.0, ice_drift_speed=0.0,
        thruster_config_name="generic_dp",
        primary_claim="Heading control remains accurate under proposed framework.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="A4_low_speed_tracking", group="A_precision",
        description="Low-speed offset tracking with gentle maneuvers.",
        duration=dur, dt=dt, target_x=4.0, target_y=-3.0, target_psi=5.0,
        ice_concentration=0.0, ice_thickness=0.0, ice_drift_speed=0.0,
        thruster_config_name="generic_dp",
        primary_claim="Position/heading tracking remains competitive in benign conditions.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="A5_dynamic_target_tracking", group="A_precision",
        description="Dynamic target: step change from (0,0,0°) to (8,5,15°) at t=60% of duration.",
        duration=dur, dt=dt,
        target_x=0.0, target_y=0.0, target_psi=0.0,
        target_x_final=8.0, target_y_final=5.0, target_psi_final=15.0,
        target_change_time=dur * 0.6,
        ice_concentration=0.0, ice_thickness=0.0, ice_drift_speed=0.0,
        thruster_config_name="generic_dp",
        primary_claim="System handles target transitions without instability.",
    ))

    # === B 组: Ice-aware DP under ice disturbance ===
    scenarios.append(SCI1Scenario(
        scenario_id="B1_moderate_drifting_ice", group="B_ice_enhancement",
        description="Moderate drifting broken ice; core ice-aware precision DP test.",
        duration=dur, dt=dt, ice_concentration=0.45, ice_thickness=0.7,
        ice_drift_speed=0.25, ice_drift_direction=125.0,
        primary_claim="At comparable precision, ice-aware risk control lowers tail risk.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="B2_ice_concentration_jump", group="B_ice_enhancement",
        description="Step change in ice concentration mid-run.",
        duration=dur, dt=dt,
        ice_concentration=0.62, ice_thickness=1.0,
        ice_drift_speed=0.35, ice_drift_direction=145.0,
        ice_time_varying=True,
        ice_concentration_initial=0.20, ice_concentration_final=0.62,
        ice_thickness_initial=1.0, ice_thickness_final=1.0,
        ice_drift_speed_initial=0.35, ice_drift_speed_final=0.35,
        ice_drift_direction_initial=145.0, ice_drift_direction_final=145.0,
        ice_schedule_type="step",
        primary_claim="Risk layer improves recovery after nonstationary ice disturbance.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="B3_ice_thickness_jump", group="B_ice_enhancement",
        description="Step change in ice thickness.",
        duration=dur, dt=dt,
        ice_concentration=0.50, ice_thickness=1.3,
        ice_drift_speed=0.30, ice_drift_direction=130.0,
        ice_time_varying=True,
        ice_concentration_initial=0.50, ice_concentration_final=0.50,
        ice_thickness_initial=0.4, ice_thickness_final=1.3,
        ice_drift_speed_initial=0.30, ice_drift_speed_final=0.30,
        ice_drift_direction_initial=130.0, ice_drift_direction_final=130.0,
        ice_schedule_type="step",
        primary_claim="Ice-aware DP adapts to sudden thickness increase.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="B4_ice_drift_speed_jump", group="B_ice_enhancement",
        description="Step change in ice drift speed.",
        duration=dur, dt=dt,
        ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.60, ice_drift_direction=120.0,
        ice_time_varying=True,
        ice_concentration_initial=0.55, ice_concentration_final=0.55,
        ice_thickness_initial=0.9, ice_thickness_final=0.9,
        ice_drift_speed_initial=0.15, ice_drift_speed_final=0.60,
        ice_drift_direction_initial=120.0, ice_drift_direction_final=120.0,
        ice_schedule_type="step",
        primary_claim="Controller handles sudden drift speed increase.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="B5_drift_direction_change", group="B_ice_enhancement",
        description="Cross-drift ice forcing where heading-to-ice matters.",
        duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.42, ice_drift_direction=90.0,
        primary_claim="Ice-aware DP reduces cross-ice load without abandoning precision.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="B6_ice_impact_pulse", group="B_ice_enhancement",
        description="Short high-intensity ice impact pulse.",
        duration=dur, dt=dt, ice_concentration=0.70, ice_thickness=1.2,
        ice_drift_speed=0.50, ice_drift_direction=150.0,
        primary_claim="CBF constraint prevents safety violation during impact.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="B7_brash_ice_channel", group="B_ice_enhancement",
        description="Brash ice channel: low concentration, thin ice, moderate drift.",
        duration=dur, dt=dt, ice_concentration=0.25, ice_thickness=0.3,
        ice_drift_speed=0.15, ice_drift_direction=100.0,
        primary_claim="System maintains precision in light ice conditions.",
    ))

    # === C 组: Thruster degradation and fault tolerance ===
    scenarios.append(SCI1Scenario(
        scenario_id="C1_single_thruster_30pct_loss", group="C_fault_tolerance",
        description="Single thruster 30% thrust loss under moderate ice.",
        duration=dur, dt=dt, ice_concentration=0.50, ice_thickness=0.8,
        ice_drift_speed=0.30, ice_drift_direction=140.0,
        degradation_name="bow_degraded_0.7",
        primary_claim="Graceful degradation under partial thruster loss.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="C2_single_thruster_50pct_loss", group="C_fault_tolerance",
        description="Single thruster 50% thrust loss under moderate ice.",
        duration=dur, dt=dt, ice_concentration=0.60, ice_thickness=1.0,
        ice_drift_speed=0.35, ice_drift_direction=150.0,
        degradation_name="bow_degraded_0.5",
        primary_claim="Safety maintained under significant thrust reduction.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="C3_single_thruster_failure", group="C_fault_tolerance",
        description="Complete single thruster failure under moderate ice.",
        duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.30, ice_drift_direction=140.0,
        degradation_name="severe",
        primary_claim="Fault-tolerant allocation redistributes thrust after failure.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="C4_azimuth_angle_stuck", group="C_fault_tolerance",
        description="Azimuthing thrusters locked at 0° (forward).",
        duration=dur, dt=dt, ice_concentration=0.50, ice_thickness=0.8,
        ice_drift_speed=0.30, ice_drift_direction=130.0,
        degradation_name="azimuth_locked",
        # H6 fix: set locked angle so runner passes actuator mode to safety filter
        azimuth_locked_angle_deg=0.0,
        primary_claim="System adapts to constrained azimuth capability.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="C5_azimuth_rate_limit", group="C_fault_tolerance",
        description="Azimuth rotation rate limited.",
        duration=dur, dt=dt, ice_concentration=0.45, ice_thickness=0.7,
        ice_drift_speed=0.25, ice_drift_direction=120.0,
        max_azimuth_rate=0.5,  # rad/s (~28.6 deg/s)
        primary_claim="Rate-limited azimuth still provides adequate control.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="C6_thrust_rate_limit", group="C_fault_tolerance",
        description="Thrust change rate limited per timestep.",
        duration=dur, dt=dt, ice_concentration=0.50, ice_thickness=0.8,
        ice_drift_speed=0.30, ice_drift_direction=135.0,
        max_thrust_rate=200.0,  # N/s
        primary_claim="Controller respects actuator rate limits.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="C7_total_power_limitation", group="C_fault_tolerance",
        description="Total power budget constrained.",
        duration=dur, dt=dt, ice_concentration=0.50, ice_thickness=0.8,
        ice_drift_speed=0.30, ice_drift_direction=130.0,
        thruster_config_name="generic_dp_power_limited",
        primary_claim="Power-aware allocation prevents overload.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="C8_ice_thruster_combined_fault", group="C_fault_tolerance",
        description="High ice load combined with thruster degradation.",
        duration=dur, dt=dt, ice_concentration=0.72, ice_thickness=1.25,
        ice_drift_speed=0.5, ice_drift_direction=160.0,
        degradation_name="bow_degraded_0.5",
        primary_claim="Combined stress: safety degradation reduces loss-of-position probability.",
    ))

    # === D 组: Safety degradation and fallback ===
    # 冰参数降低到执行器可部分抵抗的水平 (冰力 ~2000-2500N vs 3000N 执行器)
    # 这样不同控制器的降级策略会产生可区分的结果
    scenarios.append(SCI1Scenario(
        scenario_id="D1_precision_only_extreme_ice", group="D_safety_degradation",
        description="Precision DP only under heavy ice (no fallback). Baseline.",
        duration=dur, dt=dt, ice_concentration=0.70, ice_thickness=1.2,
        ice_drift_speed=0.50, ice_drift_direction=180.0, safe_region_radius=12.0,
        primary_claim="Precision-only struggles under heavy ice (baseline for D2-D4).",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="D2_ice_aware_no_fallback", group="D_safety_degradation",
        description="Ice-aware DP without Quasi-DP/Escape fallback.",
        duration=dur, dt=dt, ice_concentration=0.70, ice_thickness=1.2,
        ice_drift_speed=0.50, ice_drift_direction=180.0, safe_region_radius=12.0,
        primary_claim="Ice-aware alone outperforms precision-only under heavy ice.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="D3_ice_aware_quasi_dp", group="D_safety_degradation",
        description="Ice-aware DP with Quasi-DP fallback (no Escape).",
        duration=dur, dt=dt, ice_concentration=0.70, ice_thickness=1.2,
        ice_drift_speed=0.50, ice_drift_direction=180.0, safe_region_radius=12.0,
        primary_claim="Quasi-DP fallback prevents saturation-induced position loss.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="D4_full_escape_mode", group="D_safety_degradation",
        description="Full system with Quasi-DP + Ice-vaning/Escape.",
        duration=dur, dt=dt, ice_concentration=0.70, ice_thickness=1.2,
        ice_drift_speed=0.50, ice_drift_direction=180.0, safe_region_radius=12.0,
        primary_claim="Complete fallback chain provides maximum safety margin.",
    ))

    # === E 组: Sensor degradation and observer robustness ===
    scenarios.append(SCI1Scenario(
        scenario_id="E1_gnss_position_noise", group="E_sensor_degradation",
        description="GNSS/BeiDou position noise increased 5x.",
        duration=dur, dt=dt, ice_concentration=0.45, ice_thickness=0.7,
        ice_drift_speed=0.25, ice_drift_direction=125.0,
        position_sensor_config={"gaussian_std": 2.5, "low_pass_alpha": 0.8},
        primary_claim="Observer-based system tolerates position sensor degradation.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="E2_ins_drift", group="E_sensor_degradation",
        description="INS drift accumulating over time.",
        duration=dur, dt=dt, ice_concentration=0.45, ice_thickness=0.7,
        ice_drift_speed=0.25, ice_drift_direction=125.0,
        heading_sensor_config={"random_walk_std": 0.005, "gaussian_std": 0.005},
        primary_claim="System handles slowly drifting heading reference.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="E3_gyro_compass_bias", group="E_sensor_degradation",
        description="Gyro/compass constant bias on heading.",
        duration=dur, dt=dt, ice_concentration=0.50, ice_thickness=0.8,
        ice_drift_speed=0.30, ice_drift_direction=130.0,
        heading_sensor_config={"bias": 0.087, "gaussian_std": 0.005},  # ~5 deg bias
        primary_claim="Heading bias does not destabilize ice-aware control.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="E4_ice_concentration_error", group="E_sensor_degradation",
        description="Ice concentration estimation error (20% overestimate).",
        duration=dur, dt=dt, ice_concentration=0.50, ice_thickness=0.8,
        ice_drift_speed=0.30, ice_drift_direction=130.0,
        ice_sensor_config={"concentration_std": 0.15, "observer_alpha": 0.18},
        primary_claim="Overestimated ice concentration causes conservatism, not failure.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="E5_ice_direction_error", group="E_sensor_degradation",
        description="Ice drift direction estimation error (30 deg offset).",
        duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.35, ice_drift_direction=140.0,
        ice_sensor_config={"drift_direction_std_deg": 30.0, "observer_alpha": 0.18},
        primary_claim="Direction error degrades feedforward but CBF maintains safety.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="E6_ice_observer_delay", group="E_sensor_degradation",
        description="Ice load observer with significant time delay.",
        duration=dur, dt=dt, ice_concentration=0.50, ice_thickness=0.8,
        ice_drift_speed=0.30, ice_drift_direction=130.0,
        ice_sensor_config={"observer_alpha": 0.05},  # slow observer = effective delay
        primary_claim="Delayed observer still provides useful feedforward.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="E7_sensor_dropout", group="E_sensor_degradation",
        description="Intermittent sensor dropout (position/heading).",
        duration=dur, dt=dt, ice_concentration=0.45, ice_thickness=0.7,
        ice_drift_speed=0.25, ice_drift_direction=125.0,
        position_sensor_config={"dropout_probability": 0.15, "gaussian_std": 0.5},
        primary_claim="System degrades gracefully during sensor dropout.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="E8_combined_sensor_degradation", group="E_sensor_degradation",
        description="Combined position noise + heading bias + ice estimation error.",
        duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.35, ice_drift_direction=140.0,
        position_sensor_config={"gaussian_std": 1.5, "dropout_probability": 0.05},
        heading_sensor_config={"bias": 0.05, "random_walk_std": 0.002},
        ice_sensor_config={"concentration_std": 0.10, "drift_direction_std_deg": 15.0},
        primary_claim="Robust under realistic combined sensor imperfections.",
    ))

    # === F 组: Runtime feasibility ===
    scenarios.append(SCI1Scenario(
        scenario_id="F1_nominal_runtime", group="F_runtime",
        description="Nominal runtime measurement (same as B1).",
        duration=dur, dt=dt, ice_concentration=0.45, ice_thickness=0.7,
        ice_drift_speed=0.25, ice_drift_direction=125.0,
        primary_claim="P95 solve time remains below control update period.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="F2_heavy_disturbance_runtime", group="F_runtime",
        description="Runtime under heavy ice disturbance.",
        duration=dur, dt=dt, ice_concentration=0.75, ice_thickness=1.3,
        ice_drift_speed=0.55, ice_drift_direction=160.0,
        primary_claim="Solver remains feasible under heavy disturbance.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="F3_high_uncertainty_runtime", group="F_runtime",
        description="Runtime under high ice uncertainty.",
        duration=dur, dt=dt, ice_concentration=0.60, ice_thickness=1.0,
        ice_drift_speed=0.40, ice_drift_direction=145.0,
        primary_claim="High uncertainty increases CVaR samples but stays real-time.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="F4_nmpc_infeasibility_stress", group="F_runtime",
        description="NMPC infeasibility stress test under extreme conditions.",
        duration=dur, dt=dt, ice_concentration=0.80, ice_thickness=1.4,
        ice_drift_speed=0.60, ice_drift_direction=170.0,
        primary_claim="NMPC fallback to PD maintains safety during infeasibility.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="F5_fallback_trigger_stress", group="F_runtime",
        description="Repeated fallback trigger stress test.",
        duration=dur, dt=dt, ice_concentration=0.70, ice_thickness=1.2,
        ice_drift_speed=0.50, ice_drift_direction=155.0,
        primary_claim="Frequent mode switching does not destabilize the system.",
    ))

    # === G 组: Ice model sensitivity and data-source robustness ===
    scenarios.append(SCI1Scenario(
        scenario_id="G1_empirical_ice_model", group="G_ice_sensitivity",
        description="Empirical Lindqvist 1989 ice load model (baseline ice model).",
        duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.35, ice_drift_direction=135.0,
        ice_load_model_name="empirical",
        primary_claim="Results hold under standard empirical ice model.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="G2_stochastic_ice_model", group="G_ice_sensitivity",
        description="Stochastic ice load with burst events.",
        duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.35, ice_drift_direction=135.0,
        ice_load_model_name="stochastic",
        primary_claim="Stochastic ice loads do not invalidate safety claims.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="G3_benchmark_ice_model", group="G_ice_sensitivity",
        description="Benchmark literature-calibrated ice load model.",
        duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.35, ice_drift_direction=135.0,
        ice_load_model_name="benchmark",
        primary_claim="Benchmark model confirms generality of approach.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="G4_high_concentration_sensitivity", group="G_ice_sensitivity",
        description="Sensitivity: high concentration (c=0.9).",
        duration=dur, dt=dt, ice_concentration=0.90, ice_thickness=1.0,
        ice_drift_speed=0.35, ice_drift_direction=140.0,
        primary_claim="System remains stable at near-complete ice cover.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="G5_high_thickness_sensitivity", group="G_ice_sensitivity",
        description="Sensitivity: thick ice (h=2.0m).",
        duration=dur, dt=dt, ice_concentration=0.60, ice_thickness=2.0,
        ice_drift_speed=0.35, ice_drift_direction=140.0,
        primary_claim="Thick ice triggers fallback but maintains safety.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="G6_high_drift_speed_sensitivity", group="G_ice_sensitivity",
        description="Sensitivity: high drift speed (v=0.8 m/s).",
        duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.80, ice_drift_direction=140.0,
        primary_claim="High drift speed is the primary driver of safety risk.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="G7_drift_angle_sensitivity", group="G_ice_sensitivity",
        description="Sensitivity: ice drift perpendicular to vessel heading.",
        duration=dur, dt=dt, ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.40, ice_drift_direction=90.0,
        primary_claim="Cross-drift direction maximizes lateral ice load.",
    ))

    # === H 组: Data-driven replay scenarios ===
    # The repository bundles a small Copernicus-style mock NetCDF so artifact
    # checks can exercise the replay path without network access.  It is not a
    # Packaged mock fixture for offline artifact checks. This is NOT a real
    # Copernicus validation product.
    _copernicus_nc = "data/sci1_sources/copernicus/arctic_ice_2020_jan1_7.nc"
    _real_copernicus_nc = "data/sci1_sources/copernicus/arctic_ice_2024_jan1_7_real.nc"
    scenarios.append(SCI1Scenario(
        scenario_id="H1_mock_copernicus_fixture", group="H_data_driven",
        description="Data-driven replay: bundled Copernicus-style mock artifact subset at 80°N, 0°E.",
        duration=dur, dt=dt,
        ice_concentration=0.65, ice_thickness=1.2,  # 典型值, 运行时被数据覆盖
        ice_drift_speed=0.15, ice_drift_direction=200.0,
        data_driven=True, data_nc_path=_copernicus_nc, data_lat=80.0, data_lon=0.0,
        data_source_type="mock_fixture", data_provider="Project-generated fixture",
        data_product_id="mock-artifact-fixture-v1", allow_mock_fixture=True,
        evidence_level="synthetic",
        primary_claim="Replay pipeline is closed on a packaged data-driven ice time series; replace file for real Copernicus claims.",
    ))
    # H1_data_driven_80N 已移除: 与 H1_mock_copernicus_fixture 完全重复，会产生重复数据行

    scenarios.append(SCI1Scenario(
        scenario_id="H1_real_copernicus_era5_replay", group="H_data_driven",
        description="Real-data replay path using bundled Copernicus/ERA5-style real subset; fail-fast if the file is missing.",
        duration=dur, dt=dt,
        ice_concentration=0.65, ice_thickness=1.2,
        ice_drift_speed=0.15, ice_drift_direction=200.0,
        data_driven=True, data_nc_path=_real_copernicus_nc, data_lat=80.0, data_lon=0.0,
        data_source_type="real_subset", data_provider="Copernicus Marine Service",
        data_product_id="cmems_mod_arc_phy_anfc_6km_detided_P1D-m", allow_mock_fixture=False,
        evidence_level="reanalysis",
        primary_claim="Real subset replay path with recorded data provenance; not a full-scale DP validation claim.",
    ))
    # H1_nsidc: 使用 NSIDC-0116 独立冰漂移数据验证漂移对 DP 性能的影响
    _nsidc_drift_path = "data/sci1_sources/nsidc_0116_ice_drift/"
    scenarios.append(SCI1Scenario(
        scenario_id="H1_nsidc_drift_replay", group="H_data_driven",
        description="Real-data replay: Copernicus SIC/SIT + NSIDC-0116 ice drift at 80N, 0E.",
        duration=dur, dt=dt,
        ice_concentration=0.65, ice_thickness=1.2,
        ice_drift_speed=0.15, ice_drift_direction=200.0,
        data_driven=True, data_nc_path=_real_copernicus_nc,
        drift_nc_path=_nsidc_drift_path, data_lat=80.0, data_lon=0.0,
        data_source_type="real_subset", data_provider="Copernicus+NSIDC",
        data_product_id="cmems_mod_arc_phy_anfc_6km_detided_P1D-m + NSIDC-0116 v4",
        allow_mock_fixture=False,
        evidence_level="reanalysis",
        primary_claim="NSIDC-0116 ice drift data fills the last data gap; compare with H1_real to quantify drift-source sensitivity.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="H2_data_driven_85N", group="H_data_driven",
        description="Data-driven replay: bundled Copernicus-style mock artifact subset at 85°N, 0°E.",
        duration=dur, dt=dt,
        ice_concentration=0.80, ice_thickness=1.8,
        ice_drift_speed=0.10, ice_drift_direction=210.0,
        data_driven=True, data_nc_path=_copernicus_nc, data_lat=85.0, data_lon=0.0,
        data_source_type="mock_fixture", data_provider="Project-generated fixture",
        data_product_id="mock-artifact-fixture-v1", allow_mock_fixture=True,
        evidence_level="synthetic",
        primary_claim="System handles data-driven replay format in high-latitude-like conditions.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="H3_data_driven_75N", group="H_data_driven",
        description="Data-driven replay: bundled Copernicus-style mock artifact subset at 75°N (MIZ-like).",
        duration=dur, dt=dt,
        ice_concentration=0.35, ice_thickness=0.5,
        ice_drift_speed=0.20, ice_drift_direction=180.0,
        data_driven=True, data_nc_path=_copernicus_nc, data_lat=75.0, data_lon=0.0,
        data_source_type="mock_fixture", data_provider="Project-generated fixture",
        data_product_id="mock-artifact-fixture-v1", allow_mock_fixture=True,
        evidence_level="synthetic",
        primary_claim="MIZ-like data-driven replay path is exercised; not a real-data validation claim.",
    ))

    # === I 组: Safety filter method validation ===
    scenarios.append(SCI1Scenario(
        scenario_id="I1_safe_boundary_nominal", group="I_safety_filter",
        description="Nominal ice condition near safe boundary. Tests HOCBF activation.",
        duration=dur, dt=dt,
        ice_concentration=0.50, ice_thickness=0.8,
        ice_drift_speed=0.30, ice_drift_direction=130.0,
        safe_region_radius=12.0,
        primary_claim="Soft-HOCBF-QP prevents safety boundary violation.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="I2_safe_boundary_ice_impulse", group="I_safety_filter",
        description="Ice impulse near safe boundary. Tests HOCBF under sudden disturbance.",
        duration=dur, dt=dt,
        ice_concentration=0.70, ice_thickness=1.2,
        ice_drift_speed=0.50, ice_drift_direction=150.0,
        safe_region_radius=12.0,
        primary_claim="HOCBF slack handles sudden ice impulse without infeasibility.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="I3_safe_boundary_thruster_degraded", group="I_safety_filter",
        description="Safe boundary with thruster degradation. Tests HOCBF under reduced capacity.",
        duration=dur, dt=dt,
        ice_concentration=0.55, ice_thickness=0.9,
        ice_drift_speed=0.35, ice_drift_direction=140.0,
        safe_region_radius=12.0,
        degradation_name="bow_degraded_0.7",
        primary_claim="Safety filter adapts to reduced thruster capacity.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="I4_safe_boundary_sensor_noise", group="I_safety_filter",
        description="Safe boundary with sensor noise. Tests HOCBF robustness.",
        duration=dur, dt=dt,
        ice_concentration=0.50, ice_thickness=0.8,
        ice_drift_speed=0.30, ice_drift_direction=130.0,
        safe_region_radius=12.0,
        position_sensor_config={"gaussian_std": 1.5},
        primary_claim="HOCBF maintains safety under sensor noise.",
    ))
    scenarios.append(SCI1Scenario(
        scenario_id="I5_safe_boundary_power_cap", group="I_safety_filter",
        description="Safe boundary with power cap. Tests HOCBF under power constraint.",
        duration=dur, dt=dt,
        ice_concentration=0.50, ice_thickness=0.8,
        ice_drift_speed=0.30, ice_drift_direction=130.0,
        safe_region_radius=12.0,
        thruster_config_name="generic_dp_power_limited",
        primary_claim="Safety filter respects power cap while maintaining safety.",
    ))

    return scenarios
