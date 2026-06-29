"""第二轮修复验收测试。

覆盖:
- Phase 1: angle-aware EMA
- Phase 2: PositionSensorModel delay warmup
- Phase 3: xuelong2_like full-scale protection
- Phase 4: thruster tau_weight 等权
- Phase 5: metrics/statistics 职责边界
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ============================================================
# Phase 1: angle-aware EMA
# ============================================================

class TestAngleEMA:
    """角度 EMA 必须正确处理 ±180° 环绕。"""

    def test_angle_ema_wraparound_positive_to_negative(self):
        from arctic_quasi_dp.utils.math_utils import angle_ema_deg
        val = angle_ema_deg(179.0, -179.0, 0.5)
        # 179 和 -179 之间最短路径经过 ±180, 结果应接近 180 或 -180
        assert abs(abs(val) - 180.0) < 2.0, f"Expected ~±180, got {val}"

    def test_angle_ema_wraparound_negative_to_positive(self):
        from arctic_quasi_dp.utils.math_utils import angle_ema_deg
        val = angle_ema_deg(-179.0, 179.0, 0.5)
        assert abs(abs(val) - 180.0) < 2.0, f"Expected ~±180, got {val}"

    def test_shortest_angle_diff_deg(self):
        from arctic_quasi_dp.utils.math_utils import shortest_angle_diff_deg
        # -179 → 179: 最短差是 +2° (经过 ±180)
        assert abs(shortest_angle_diff_deg(-179.0, 179.0) - 2.0) < 1e-9
        # 179 → -179: 最短差是 -2°
        assert abs(shortest_angle_diff_deg(179.0, -179.0) + 2.0) < 1e-9

    def test_wrap_angle_deg(self):
        from arctic_quasi_dp.utils.math_utils import wrap_angle_deg
        assert abs(wrap_angle_deg(180.0) - (-180.0)) < 1e-9
        assert abs(wrap_angle_deg(360.0) - 0.0) < 1e-9
        assert abs(wrap_angle_deg(-360.0) - 0.0) < 1e-9
        assert abs(wrap_angle_deg(270.0) - (-90.0)) < 1e-9

    def test_angle_ema_no_wraparound(self):
        from arctic_quasi_dp.utils.math_utils import angle_ema_deg
        # 普通情况: 10° 和 20°, alpha=0.5 → 15°
        val = angle_ema_deg(10.0, 20.0, 0.5)
        assert abs(val - 15.0) < 1e-9

    def test_heading_sensor_angle_ema(self):
        """HeadingSensorModel 的 EMA 必须使用角度感知平均 (不产生 0° 穿越)。"""
        from arctic_quasi_dp.sci1.sensor_models import HeadingSensorModel, SensorNoiseConfig
        cfg = SensorNoiseConfig(gaussian_std=0.0, bias=0.0, low_pass_alpha=0.9)
        sensor = HeadingSensorModel(cfg)
        rng = np.random.default_rng(42)

        # 先用多步收敛到 179°
        for _ in range(20):
            sensor.measure(math.radians(179.0), rng)
        # 然后跳到 -179°
        result = sensor.measure(math.radians(-179.0), rng)
        result_deg = math.degrees(result)
        # 结果应该接近 ±180° (经过环绕), 而不是 0° (线性平均的错误结果)
        # 如果是线性平均: (179 + -179)/2 = 0° — 这是错误的
        # 角度 EMA: 应该在 ~178° 或 ~-178°
        assert abs(result_deg) > 90.0, \
            f"Angle EMA should not cross to 0° (linear avg bug), got {result_deg}°"

    def test_ice_sensor_direction_angle_ema(self):
        """IceConditionSensorModel 的方向 EMA 必须使用角度感知平均 (不产生 0° 穿越)。"""
        from arctic_quasi_dp.sci1.sensor_models import IceConditionSensorModel
        model = IceConditionSensorModel(
            concentration_std=0.0, thickness_std=0.0,
            drift_speed_std=0.0, drift_direction_std_deg=0.0,
            observer_alpha=0.9,
        )
        rng = np.random.default_rng(42)

        # 先收敛到 170°
        for _ in range(20):
            model.update(0.5, 0.8, 0.3, 170.0, rng, 0.1)
        # 跳到 -170°
        est = model.update(0.5, 0.8, 0.3, -170.0, rng, 0.1)
        # 结果应接近 ±180°, 不是 0° (线性平均的错误结果)
        assert abs(est.drift_direction_deg) > 90.0, \
            f"Angle EMA should not cross to 0° (linear avg bug), got {est.drift_direction_deg}°"


# ============================================================
# Phase 2: PositionSensorModel delay warmup
# ============================================================

class TestPositionSensorDelayWarmup:
    """delay buffer 填充期不应返回原点。"""

    def test_delay_warmup_returns_not_zeros(self):
        """warmup 期返回值不应是 np.zeros(2)。"""
        from arctic_quasi_dp.sci1.sensor_models import PositionSensorModel, SensorNoiseConfig
        cfg = SensorNoiseConfig(gaussian_std=0.0, bias=0.0, time_delay_steps=5)
        sensor = PositionSensorModel(cfg)
        rng = np.random.default_rng(42)
        true_pos = np.array([10.0, -3.0])
        obs = sensor.measure(true_pos, rng)
        # 不应返回原点
        assert not np.allclose(obs, np.zeros(2)), "warmup should not return zeros"
        # 应返回接近真实位置
        assert np.allclose(obs, true_pos, atol=0.1), f"Expected ~{true_pos}, got {obs}"

    def test_delay_after_buffer_filled(self):
        """buffer 满后应返回延迟的观测。"""
        from arctic_quasi_dp.sci1.sensor_models import PositionSensorModel, SensorNoiseConfig
        cfg = SensorNoiseConfig(gaussian_std=0.0, bias=0.0, time_delay_steps=2)
        sensor = PositionSensorModel(cfg)
        rng = np.random.default_rng(42)

        p0 = np.array([1.0, 1.0])
        p1 = np.array([2.0, 2.0])
        p2 = np.array([3.0, 3.0])

        sensor.measure(p0, rng)
        sensor.measure(p1, rng)
        obs = sensor.measure(p2, rng)
        # buffer 满后, 返回的是最早入队的 (p0)
        assert np.allclose(obs, p0, atol=0.1), f"Expected ~{p0}, got {obs}"


# ============================================================
# Phase 3: xuelong2_like full-scale protection
# ============================================================

class TestFullScaleProtection:
    """full_scale_experimental 配置不能静默进入 paper profile。"""

    def test_xuelong2_rejected_by_paper_profile(self):
        """paper profile + xuelong2_like 应抛出 ValueError。"""
        from arctic_quasi_dp.sci1.runner import run_experiments
        cfg_path = Path(__file__).parent.parent.parent / "configs" / "vessels" / "xuelong2_like.yaml"
        if not cfg_path.exists():
            pytest.skip("xuelong2_like.yaml not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results"
            with pytest.raises(ValueError, match="full_scale_experimental"):
                run_experiments(
                    profile="paper", seeds=1, controllers=["pid"],
                    out_dir=out, save_traces=False, save_figures=False,
                    vessel_config_path=str(cfg_path),
                )

    def test_xuelong2_allowed_with_explicit_flag(self):
        """设置 allow_experimental_full_scale=True 后应允许运行。"""
        from arctic_quasi_dp.sci1.runner import run_experiments
        cfg_path = Path(__file__).parent.parent.parent / "configs" / "vessels" / "xuelong2_like.yaml"
        if not cfg_path.exists():
            pytest.skip("xuelong2_like.yaml not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results"
            # 不应抛出异常
            run_experiments(
                profile="smoke", seeds=1, controllers=["pid"],
                out_dir=out, save_traces=False, save_figures=False,
                vessel_config_path=str(cfg_path),
                effective_cfg={"allow_experimental_full_scale": True},
            )
            manifest = (out / "metadata" / "vessel_manifest.json").read_text(encoding="utf-8")
            assert "full_scale_experimental" in manifest

    def test_proxy_scale_manifest_contains_warning(self):
        """proxy_scale manifest 应包含 warning 字段。"""
        from arctic_quasi_dp.sci1.runner import run_experiments
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results"
            run_experiments(
                profile="smoke", seeds=1, controllers=["pid"],
                out_dir=out, save_traces=False, save_figures=False,
            )
            manifest = json.loads((out / "metadata" / "vessel_manifest.json").read_text(encoding="utf-8"))
            assert manifest["scale_type"] == "proxy_scale"
            assert "warning" in manifest
            assert "proxy-scale" in manifest["warning"].lower()
            assert manifest["full_scale_ready"] is False


# ============================================================
# Phase 4: thruster tau_weight 等权
# ============================================================

class TestThrusterTauWeight:
    """推进器分配任务权重应默认等权。"""

    def test_default_tau_weight_is_equal(self):
        from arctic_quasi_dp.sci1.thruster import ThrusterAllocator, ThrusterConfig
        allocator = ThrusterAllocator(ThrusterConfig.generic_dp_vessel())
        np.testing.assert_allclose(allocator.tau_weight, [1.0, 1.0, 1.0])

    def test_custom_tau_weight(self):
        """应支持自定义 tau_weight。"""
        from arctic_quasi_dp.sci1.thruster import ThrusterAllocator, ThrusterConfig
        w = np.array([1.0, 1.0, 0.5])
        allocator = ThrusterAllocator(ThrusterConfig.generic_dp_vessel(), tau_weight=w)
        np.testing.assert_allclose(allocator.tau_weight, [1.0, 1.0, 0.5])

    def test_tau_weight_affects_allocation(self):
        """不同的 tau_weight 应产生不同的分配结果。"""
        from arctic_quasi_dp.sci1.thruster import ThrusterAllocator, ThrusterConfig
        # 使用非对称力+力矩, 确保打破推进器对称性
        tau = np.array([200.0, 100.0, 500.0])

        a1 = ThrusterAllocator(ThrusterConfig.generic_dp_vessel())
        thrusts1, _ = a1.allocate(tau)

        a2 = ThrusterAllocator(ThrusterConfig.generic_dp_vessel(), tau_weight=np.array([1.0, 1.0, 0.1]))
        thrusts2, _ = a2.allocate(tau)

        # 降低偏航权重应改变推力分配
        assert not np.allclose(thrusts1, thrusts2, atol=1e-3)


# ============================================================
# Phase 5: metrics/statistics 职责边界
# ============================================================

class TestMetricsStatisticsBoundary:
    """metrics.py 的跨控制器统计应委托给 statistics.py。"""

    def test_metrics_statistical_comparison_uses_statistics_module(self):
        """metrics.statistical_comparison 应产生与 statistics.paired_comparison一致的结果。"""
        import pandas as pd
        from arctic_quasi_dp.sci1.metrics import statistical_comparison
        from arctic_quasi_dp.sci1.statistics import paired_comparison

        # 构造测试数据
        rows = []
        for seed in range(10):
            rows.append({"scenario_id": "B1", "controller": "pid", "seed": seed,
                         "rms_position_error_m": 1.0 + seed * 0.1, "failure": 0})
            rows.append({"scenario_id": "B1", "controller": "full", "seed": seed,
                         "rms_position_error_m": 0.5 + seed * 0.05, "failure": 0})
        df = pd.DataFrame(rows)

        m_result = statistical_comparison(df, "rms_position_error_m", "pid", "full")
        s_result = paired_comparison(df, "rms_position_error_m", "pid", "full", lower_is_better=True)

        # p 值应一致 (都使用 Wilcoxon signed-rank)
        if not math.isnan(m_result["p_value"]) and not math.isnan(s_result["p_value"]):
            assert abs(m_result["p_value"] - s_result["p_value"]) < 1e-6

    def test_summarize_run_does_not_output_p_value(self):
        """metrics.summarize_run (单 run) 不应输出跨控制器统计字段。"""
        import pandas as pd
        from arctic_quasi_dp.sci1.metrics import summarize_run

        df = pd.DataFrame({
            "time": [0.1, 0.2],
            "position_error": [1.0, 2.0],
            "heading_error": [0.1, 0.2],
            "violation": [0.0, 0.0],
            "energy": [0.0, 1.0],
        })
        result = summarize_run(df, "B1", "pid", 0, 0.1)
        # 单 run summary 不应包含 p_value / effect_size_holm
        assert "p_value" not in result
        assert "effect_size_holm" not in result
        assert "cohens_d" not in result
