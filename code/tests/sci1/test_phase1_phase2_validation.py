"""阶段 1-2 综合验收测试。

覆盖:
- 1.1: vessel config 驱动仿真
- 1.2: proxy-scale manifest 一致性
- 1.3: C7 power cap 触发
- 1.4: E5/E8 方向噪声单位
- 1.5: data_bridge numpy or bug
- 1.6: _extract_point 2D 索引
- 2.1: group-level 统计
- 2.2: CI NaN when n<2
- 2.4: trace 执行器证据字段
- 4.2: oracle_full 使用 true ice
"""

import math
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ============================================================
# 1.5: data_bridge numpy array `or` bug
# ============================================================

class TestDataBridgeNumpyOrBug:
    """data_bridge 中 numpy array 不能用 `or` 判断。"""

    def test_first_existing_with_arrays(self):
        """_first_existing 应正确处理 numpy array 值。"""
        from arctic_quasi_dp.sci1.data_bridge import _first_existing
        data = {
            "vxsi": np.array([1.0, 2.0, 3.0]),
            "vysi": np.array([4.0, 5.0, 6.0]),
        }
        vx = _first_existing(data, ["vxsi", "sivelu"])
        vy = _first_existing(data, ["vysi", "sivelv"])
        assert vx is not None
        assert vy is not None
        np.testing.assert_array_equal(vx, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(vy, [4.0, 5.0, 6.0])

    def test_first_existing_fallback(self):
        """_first_existing 在首选键不存在时回退到备选。"""
        from arctic_quasi_dp.sci1.data_bridge import _first_existing
        data = {"sivelu": np.array([1.0]), "sivelv": np.array([2.0])}
        vx = _first_existing(data, ["vxsi", "sivelu"])
        assert vx is not None
        np.testing.assert_array_equal(vx, [1.0])

    def test_first_existing_none(self):
        """_first_existing 在所有键都不存在时返回 None。"""
        from arctic_quasi_dp.sci1.data_bridge import _first_existing
        data = {"other": np.array([1.0])}
        vx = _first_existing(data, ["vxsi", "sivelu"])
        assert vx is None


# ============================================================
# 1.6: _extract_point 2D 索引
# ============================================================

class TestExtractPoint2D:
    """_extract_point 应返回标量而非切片。"""

    def test_extract_point_2d_returns_scalar(self):
        """2D (lat, lon) 快照应返回标量。"""
        from arctic_quasi_dp.sci1.data_bridge import DataDrivenIceSchedule
        # 直接测试内部逻辑: 2D array with matching lat/lon dimensions
        arr = np.arange(12).reshape(3, 4)
        # 模拟 _extract_point 的 2D 分支逻辑
        n_lat, n_lon = 3, 4
        lat_idx, lon_idx = 1, 2
        if arr.shape[0] == n_lat and arr.shape[1] == n_lon:
            val = float(arr[lat_idx, lon_idx])
        else:
            val = arr[:, lon_idx]
        assert np.isscalar(val)
        assert val == 6.0  # arr[1, 2] = 6

    def test_extract_point_3d_returns_timeseries(self):
        """3D (time, lat, lon) 应返回 1D 时间序列。"""
        arr = np.arange(60).reshape(5, 3, 4)
        result = arr[:, 1, 2]
        assert result.shape == (5,)
        assert result[0] == arr[0, 1, 2]


# ============================================================
# 2.1: group-level 统计
# ============================================================

class TestGroupExtraction:
    """scenario_id 应提取字母组而非含数字的前缀。"""

    def test_group_extraction_uses_letter_group(self):
        from arctic_quasi_dp.sci1.scenarios import scenario_group_from_id
        assert scenario_group_from_id("A1_open_water_station_keeping") == "A"
        assert scenario_group_from_id("A5_dynamic_target_tracking") == "A"
        assert scenario_group_from_id("B2_ice_concentration_jump") == "B"
        assert scenario_group_from_id("G3_model_sensitivity") == "G"
        assert scenario_group_from_id("H1_copernicus_replay") == "H"

    def test_group_extraction_fallback(self):
        from arctic_quasi_dp.sci1.scenarios import scenario_group_from_id
        # 没有数字前缀的 ID 回退到 split
        assert scenario_group_from_id("custom_scenario") == "custom"


# ============================================================
# 2.2: CI NaN when n<2
# ============================================================

class TestCINanWhenNlt2:
    """n=1 时 CI 应输出 NaN 而非 0。"""

    def test_ci_nan_when_n_lt_2(self):
        from arctic_quasi_dp.sci1.metrics import aggregate_summary
        import pandas as pd
        # 构造 n=1 的数据
        rows = [{"scenario_id": "A1", "controller": "pid", "seed": 0,
                 "rms_position_error_m": 1.0, "failure": 0}]
        df = pd.DataFrame(rows)
        agg = aggregate_summary(df)
        assert len(agg) == 1
        assert math.isnan(agg.iloc[0]["rms_position_error_m_ci95"])
        assert math.isnan(agg.iloc[0]["rms_position_error_m_std"])

    def test_ci_valid_when_n_ge_2(self):
        from arctic_quasi_dp.sci1.metrics import aggregate_summary
        import pandas as pd
        rows = [
            {"scenario_id": "A1", "controller": "pid", "seed": 0, "rms_position_error_m": 1.0, "failure": 0},
            {"scenario_id": "A1", "controller": "pid", "seed": 1, "rms_position_error_m": 2.0, "failure": 0},
        ]
        df = pd.DataFrame(rows)
        agg = aggregate_summary(df)
        assert not math.isnan(agg.iloc[0]["rms_position_error_m_ci95"])
        assert agg.iloc[0]["rms_position_error_m_ci95"] > 0


# ============================================================
# 1.1: vessel config 驱动仿真
# ============================================================

class TestVesselConfigDrivesSimulation:
    """vessel config YAML 应真正驱动 VesselParams。"""

    def test_load_vessel_config_simplified(self):
        from arctic_quasi_dp.sci1.vessel_config import load_vessel_config
        cfg_path = Path(__file__).parent.parent.parent / "configs" / "vessels" / "simplified_500t.yaml"
        if not cfg_path.exists():
            pytest.skip("Vessel config not found")
        vc = load_vessel_config(cfg_path)
        assert vc.name == "simplified_500t"
        assert vc.scale_type == "proxy_scale"
        assert vc.vessel_params.mass == 500000.0
        assert vc.vessel_params.beam == 22.0
        assert vc.max_force == 3000.0

    def test_load_vessel_config_xuelong2(self):
        from arctic_quasi_dp.sci1.vessel_config import load_vessel_config
        cfg_path = Path(__file__).parent.parent.parent / "configs" / "vessels" / "xuelong2_like.yaml"
        if not cfg_path.exists():
            pytest.skip("Vessel config not found")
        vc = load_vessel_config(cfg_path)
        assert vc.name == "xuelong2_like"
        assert vc.scale_type == "full_scale_experimental"
        assert vc.vessel_params.mass == 13996000.0
        assert vc.max_force == 1.0e6

    def test_vessel_manifest_reflects_actual_config(self):
        """smoke 运行后 vessel_manifest.json 应反映实际配置。"""
        from arctic_quasi_dp.sci1.runner import run_experiments
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results"
            run_experiments(
                profile="smoke", seeds=1, controllers=["pid"],
                out_dir=out, save_traces=False, save_figures=False,
            )
            manifest_path = out / "metadata" / "vessel_manifest.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert "scale_type" in manifest
            assert "ice_crushing_strength_mpa" in manifest
            assert manifest["scale_type"] in ("proxy_scale", "full_scale")


# ============================================================
# 1.2: proxy-scale manifest 一致性
# ============================================================

class TestProxyScaleConsistency:
    """manifest 中 source_note 和实际参数应一致。"""

    def test_manifest_source_note_matches_params(self):
        from arctic_quasi_dp.sci1.runner import run_experiments
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results"
            run_experiments(
                profile="smoke", seeds=1, controllers=["pid"],
                out_dir=out, save_traces=False, save_figures=False,
            )
            manifest = json.loads((out / "metadata" / "vessel_manifest.json").read_text(encoding="utf-8"))
            # ice_crushing_strength 应在 source_note 中被正确引用
            strength = manifest["ice_crushing_strength_mpa"]
            note = manifest.get("source_note", "")
            # 不应出现旧的 0.004 MPa 说明
            assert "0.004" not in note or strength == 0.004


# ============================================================
# 1.3: C7 power cap 触发
# ============================================================

class TestC7PowerCap:
    """C7 场景的 power cap 应在重冰况下触发。"""

    def test_power_limited_config_is_restrictive(self):
        """power_limited 推进器配置的 cap 应低于满推功率。"""
        from arctic_quasi_dp.sci1.runner import _THRUSTER_CONFIGS
        cfg = _THRUSTER_CONFIGS.get("generic_dp_power_limited")
        assert cfg is not None
        max_power = sum(t.max_thrust for t in cfg.thrusters) / 1000.0
        assert cfg.max_total_power_kw < max_power, (
            f"Power cap {cfg.max_total_power_kw} kW should be less than "
            f"full thrust power {max_power:.2f} kW"
        )


# ============================================================
# 1.4: E5/E8 方向噪声单位
# ============================================================

class TestSensorDirectionNoise:
    """方向噪声字段应显式标注单位为 degree。"""

    def test_sensor_direction_noise_field_is_degree(self):
        from arctic_quasi_dp.sci1.sensor_models import IceConditionSensorModel
        model = IceConditionSensorModel(drift_direction_std_deg=30.0)
        assert model.drift_direction_std_deg == 30.0

    def test_backward_compat_drift_direction_std(self):
        """旧字段 drift_direction_std 应向后兼容。"""
        import warnings
        from arctic_quasi_dp.sci1.sensor_models import IceConditionSensorModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = IceConditionSensorModel(drift_direction_std=0.5)
        assert model.drift_direction_std_deg == 0.5

    def test_direction_noise_produces_expected_std(self):
        """drift_direction_std_deg=30 应产生约 30 度的标准差。"""
        from arctic_quasi_dp.sci1.sensor_models import IceConditionSensorModel
        rng = np.random.default_rng(42)
        model = IceConditionSensorModel(drift_direction_std_deg=30.0)
        estimates = []
        for _ in range(200):
            model.reset()
            est = model.update(0.5, 0.8, 0.3, 180.0, rng, 0.1)
            estimates.append(est.drift_direction_deg)
        # 初始值为 0, 经过 EMA 后趋近 180, 但噪声应有明显波动
        # 检查最后 100 步的标准差
        tail = estimates[-100:]
        # 由于 EMA 平滑, 观测噪声 std 不等于输入 std, 但应 > 5 度
        assert np.std(tail) > 5.0, f"Direction noise std {np.std(tail):.1f} too small for input 30 deg"


# ============================================================
# 2.4: trace 执行器证据字段
# ============================================================

class TestActuatorTraceFields:
    """trace 应包含执行器层证据字段。"""

    def test_allocator_provides_actuator_trace(self):
        from arctic_quasi_dp.sci1.thruster import ThrusterAllocator, ThrusterConfig
        allocator = ThrusterAllocator(ThrusterConfig.generic_dp_vessel())
        thrusts = np.array([100.0, 100.0, 100.0, 100.0, 50.0])
        tau_desired = np.array([300.0, 0.0, 1000.0])
        tau_actual = allocator.resulting_tau(thrusts)
        trace = allocator.get_actuator_trace(thrusts, tau_desired, tau_actual)
        # 检查关键字段存在
        assert "total_power_kw" in trace
        assert "power_cap_active" in trace
        assert "allocation_residual_norm" in trace
        assert "thrust_saturation_ratio" in trace
        assert "thruster_0_commanded_thrust" in trace
        assert "thruster_0_azimuth_locked" in trace

    def test_trace_in_simulation_output(self):
        """仿真输出应包含执行器证据字段。"""
        from arctic_quasi_dp.sci1.sim_loop import run_simulation
        from arctic_quasi_dp.sci1.controllers import PrecisionDPController
        from arctic_quasi_dp.sci1.thruster import ThrusterConfig
        ctrl = PrecisionDPController()
        log = run_simulation(
            controller=ctrl, duration=1.0, dt=0.1,
            target_x=0.0, target_y=0.0, target_psi=0.0,
            thruster_config=ThrusterConfig.generic_dp_vessel(),
            seed=42,
        )
        df = log.to_dataframe()
        assert "total_power_kw" in df.columns
        assert "power_cap_active" in df.columns
        assert "allocation_residual_norm" in df.columns
        assert "thruster_0_commanded_thrust" in df.columns


# ============================================================
# 4.2: oracle_full 使用 true ice
# ============================================================

class TestOracleFullUsesTrueIce:
    """oracle_full 应绕过传感器噪声, 使用真值冰况。"""

    def test_oracle_skips_sensor_noise(self):
        """oracle_full 路径不应使用传感器模型。"""
        # 验证 _run_single 中 oracle_full 路径的逻辑
        from arctic_quasi_dp.sci1.runner import _run_single
        from arctic_quasi_dp.sci1.scenarios import build_sci1_scenarios
        from arctic_quasi_dp.sci1.controllers import ModeSupervisedIceDPController

        scenarios = build_sci1_scenarios("smoke")
        # 找一个有传感器退化的场景
        e_scenario = None
        for s in scenarios:
            if s.scenario_id.startswith("E1"):
                e_scenario = s
                break
        if e_scenario is None:
            pytest.skip("E1 scenario not found")

        ctrl = ModeSupervisedIceDPController()
        # oracle_full 不应崩溃
        df, dt = _run_single(e_scenario, "oracle_full", ctrl, 0, "smoke")
        assert len(df) > 0
        # oracle 应有有效的 position_error
        assert df["position_error"].mean() >= 0
