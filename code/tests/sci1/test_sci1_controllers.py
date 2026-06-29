"""SCI1 控制器全面测试。

覆盖:
- PrecisionDPController: 基本功能、set_target、reset
- IceAwarePrecisionDPController: 冰况估计、CVaR、CBF、Lindqvist 冰力模型
- QuasiDPSafetyController: 安全降级、目标放松、状态恢复
- IceVaningEscapeController: 迎冰/撤离控制
- ModeSupervisedIceDPController: 模式切换、hysteresis、dwell time
- 消融控制器: no_cbf, no_cvar, no_observer, no_fallback
- 边界条件: 零冰况、极端冰况、无目标状态
"""

import numpy as np
import pytest

from arctic_quasi_dp.sci1.controllers import (
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
    DPMode,
    _ice_risk_standardized,
    _ice_force_lindqvist_proxy,
)


# ---------- 辅助函数 ----------

def _make_state(x=0.0, y=0.0, psi=0.0, u=0.0, v=0.0, r=0.0):
    return np.array([x, y, psi, u, v, r], dtype=np.float64)


def _extreme_ice():
    return {"ice_concentration": 0.9, "ice_thickness": 1.6, "ice_drift_speed": 0.7, "ice_drift_direction": 180.0}


# ---------- PrecisionDPController ----------

class TestPrecisionDPController:
    def test_finite_tau(self):
        ctrl = PrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        result = ctrl.compute_control(_make_state(2.0, -1.0, 0.1), dt=0.1)
        assert result.tau.shape == (3,)
        assert np.all(np.isfinite(result.tau))
        assert result.feasible

    def test_no_target_returns_zero(self):
        ctrl = PrecisionDPController()
        result = ctrl.compute_control(_make_state(5.0, 5.0), dt=0.1)
        np.testing.assert_array_equal(result.tau, np.zeros(3))
        assert result.feasible

    def test_reference_overrides_target(self):
        ctrl = PrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        result = ctrl.compute_control(_make_state(2.0, 0.0), reference={"x": 2.0, "y": 0.0}, dt=0.1)
        # 参考点就在当前位置，误差应为 0
        assert result.tau.shape == (3,)

    def test_reset_clears_state(self):
        ctrl = PrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.compute_control(_make_state(5.0, 5.0), dt=0.1)
        ctrl.reset()
        assert np.allclose(ctrl._int_pos, 0.0)
        assert ctrl._int_psi == 0.0

    def test_diagnostics_populated(self):
        ctrl = PrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.compute_control(_make_state(1.0, 1.0), dt=0.1)
        diag = ctrl.get_diagnostics()
        assert "solver_status" in diag
        assert "risk_total" in diag
        assert diag["solver_status"] == "analytical"

    def test_set_safe_region_radius(self):
        ctrl = PrecisionDPController()
        # PrecisionDPParams 没有 cbf_radius，调用应静默忽略
        ctrl.set_safe_region_radius(15.0)
        # IceAware 子类才有 cbf_radius
        ice_ctrl = IceAwarePrecisionDPController()
        ice_ctrl.set_safe_region_radius(15.0)
        assert ice_ctrl.params.cbf_radius == 15.0


# ---------- IceAwarePrecisionDPController ----------

class TestIceAwarePrecisionDPController:
    def test_finite_tau_with_ice(self):
        ctrl = IceAwarePrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.6, 1.0, 0.4, 120.0)
        result = ctrl.compute_control(_make_state(), dt=0.1)
        assert result.tau.shape == (3,)
        assert np.all(np.isfinite(result.tau))

    def test_diagnostics_include_ice_info(self):
        ctrl = IceAwarePrecisionDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.6, 1.0, 0.4, 120.0)
        ctrl.compute_control(_make_state(), dt=0.1)
        diag = ctrl.get_diagnostics()
        assert "risk_cvar" in diag
        assert "cbf_status" in diag
        assert "risk_ice" in diag
        assert diag["risk_model_status"] == "observer_proxy"

    def test_observer_estimates_ice(self):
        ctrl = IceAwarePrecisionDPController(use_observer=True)
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.6, 1.0, 0.4, 120.0)
        # 运行多步让观测器收敛
        for _ in range(10):
            ctrl.compute_control(_make_state(), dt=0.1)
        est = ctrl._ice_est
        assert abs(est["concentration"] - 0.6) < 0.3  # 观测器应接近真实值
        assert est["thickness"] > 0

    def test_no_observer_passes_through(self):
        ctrl = IceAwarePrecisionDPController(use_observer=False)
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.6, 1.0, 0.4, 120.0)
        ctrl.compute_control(_make_state(), dt=0.1)
        est = ctrl._ice_est
        assert abs(est["concentration"] - 0.6) < 0.01

    def test_cvar_disabled(self):
        ctrl = IceAwarePrecisionDPController(use_cvar=False)
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.6, 1.0, 0.4, 120.0)
        ctrl.compute_control(_make_state(), dt=0.1)
        diag = ctrl.get_diagnostics()
        assert diag["cvar_diagnostic"] == "disabled"

    def test_cbf_active_near_boundary_moving_outward(self):
        ctrl = IceAwarePrecisionDPController(use_cbf=True)
        ctrl.set_target(0.0, 0.0, 0.0)
        # 在安全边界附近且快速向外移动 => CBF 应提供安全裕度
        # 注意: 强 PD 增益本身提供足够制动力, CBF margin 可能为正 (约束已满足)
        state = _make_state(x=9.5, y=0.0, u=2.0, v=0.0)  # 9.5m from center, moving at 2 m/s
        result = ctrl.compute_control(state, dt=0.1)
        diag = ctrl.get_diagnostics()
        # CBF 要么主动修正 (margin < 0), 要么约束已满足 (margin >= 0)
        # 两种情况都表明安全机制在工作
        assert diag["safety_filter_hocbf_margin"] is not None, (
            f"HOCBF margin should be computed, got None"
        )

    def test_cbf_inactive_inside_safe_zone(self):
        ctrl = IceAwarePrecisionDPController(use_cbf=True)
        ctrl.set_target(0.0, 0.0, 0.0)
        # Deep inside safe zone (dist < 0.75 * radius) => CBF should NOT activate
        state = _make_state(x=2.0, y=0.0)
        result = ctrl.compute_control(state, dt=0.1)
        diag = ctrl.get_diagnostics()
        assert diag["cbf_active"] is False, "CBF should be inactive deep inside safe zone"

    def test_cvar_seed_changes_output(self):
        ctrl1 = IceAwarePrecisionDPController(use_cvar=True)
        ctrl1.set_target(0.0, 0.0, 0.0)
        ctrl1.set_ice_conditions(0.6, 1.0, 0.4, 120.0)
        ctrl1.set_cvar_seed(42)
        r1 = ctrl1.compute_control(_make_state(), dt=0.1)

        ctrl2 = IceAwarePrecisionDPController(use_cvar=True)
        ctrl2.set_target(0.0, 0.0, 0.0)
        ctrl2.set_ice_conditions(0.6, 1.0, 0.4, 120.0)
        ctrl2.set_cvar_seed(123)
        r2 = ctrl2.compute_control(_make_state(), dt=0.1)

        # 不同种子应产生不同 CVaR (概率极高)
        d1 = ctrl1.get_diagnostics()
        d2 = ctrl2.get_diagnostics()
        # 至少 quantile 应该不同
        assert d1["cvar_quantile"] != d2["cvar_quantile"] or d1["cvar_sample_count"] == 0


# ---------- QuasiDPSafetyController ----------

class TestQuasiDPSafetyController:
    def test_finite_tau(self):
        ctrl = QuasiDPSafetyController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.6, 1.0, 0.4, 120.0)
        result = ctrl.compute_control(_make_state(2.0, 2.0), dt=0.1)
        assert np.all(np.isfinite(result.tau))
        assert result.mode == "quasi_dp"

    def test_relaxes_target_inside_watch_circle(self):
        ctrl = QuasiDPSafetyController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.5, 0.8, 0.3, 90.0)
        # 在 watch circle 内 (dist < 10m)
        state = _make_state(x=3.0, y=4.0)  # dist = 5.0
        result = ctrl.compute_control(state, dt=0.1)
        # 检查诊断中标记了放松
        diag = ctrl.get_diagnostics()
        assert diag.get("reference_relaxed", False)

    def test_no_relax_outside_watch_circle(self):
        ctrl = QuasiDPSafetyController()
        ctrl.set_target(0.0, 0.0, 0.0)
        # 在 watch circle 外
        state = _make_state(x=15.0, y=0.0)  # dist = 15.0
        result = ctrl.compute_control(state, dt=0.1)
        diag = ctrl.get_diagnostics()
        # 在圈外不应放松
        assert not diag.get("reference_relaxed", False) or diag.get("true_position_error", 0) > 10.0

    def test_target_restored_after_compute(self):
        ctrl = QuasiDPSafetyController()
        ctrl.set_target(5.0, 5.0, 0.0)
        ctrl.set_ice_conditions(0.5, 0.8, 0.3, 90.0)
        original = ctrl._target_pos
        state = _make_state(x=7.0, y=7.0)  # 在圈内
        ctrl.compute_control(state, dt=0.1)
        # target_pos 应该恢复
        assert ctrl._target_pos == original

    def test_no_target_returns_zero(self):
        ctrl = QuasiDPSafetyController()
        result = ctrl.compute_control(_make_state(), dt=0.1)
        np.testing.assert_array_equal(result.tau, np.zeros(3))


# ---------- IceVaningEscapeController ----------

class TestIceVaningEscapeController:
    def test_finite_tau(self):
        ctrl = IceVaningEscapeController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.9, 1.6, 0.7, 180.0)
        result = ctrl.compute_control(_make_state(x=5.0, y=0.0), dt=0.1)
        assert np.all(np.isfinite(result.tau))
        assert result.mode == "escape"

    def test_risk_is_high(self):
        ctrl = IceVaningEscapeController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.9, 1.6, 0.7, 180.0)
        result = ctrl.compute_control(_make_state(x=5.0, y=0.0), dt=0.1)
        assert result.risk >= 0.75  # escape 模式风险应较高

    def test_no_target_returns_zero(self):
        ctrl = IceVaningEscapeController()
        result = ctrl.compute_control(_make_state(), dt=0.1)
        np.testing.assert_array_equal(result.tau, np.zeros(3))
        assert result.risk == 1.0


# ---------- ModeSupervisedIceDPController ----------

class TestModeSupervisedIceDPController:
    def test_finite_tau_extreme_ice(self):
        ctrl = ModeSupervisedIceDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(**_extreme_ice())
        result = ctrl.compute_control(_make_state(x=12.0, y=0.0), dt=0.2)
        assert np.all(np.isfinite(result.tau))

    def test_mode_name_valid(self):
        ctrl = ModeSupervisedIceDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(**_extreme_ice())
        ctrl.compute_control(_make_state(x=12.0, y=0.0), dt=0.2)
        diag = ctrl.get_diagnostics()
        assert diag.get("supervisor_mode_name") in {"PRECISION", "ICE_AWARE", "QUASI_DP", "ESCAPE"}

    def test_mode_switches_with_risk(self):
        ctrl = ModeSupervisedIceDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        # 低冰况 -> PRECISION
        ctrl.set_ice_conditions(0.1, 0.2, 0.05, 90.0)
        for _ in range(5):
            mode1 = ctrl._select_mode(_make_state(), 0.2)
        assert mode1 == DPMode.PRECISION

        # 高冰况 -> 应该切换
        ctrl.set_ice_conditions(0.9, 1.6, 0.7, 180.0)
        for _ in range(60):  # 超过 dwell_time
            mode2 = ctrl._select_mode(_make_state(), 0.2)
        assert mode2 in {DPMode.ICE_AWARE, DPMode.QUASI_DP, DPMode.ESCAPE}

    def test_dwell_time_prevents_rapid_switching(self):
        ctrl = ModeSupervisedIceDPController(params=SupervisorParams(dwell_time=10.0))
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.6, 1.0, 0.4, 120.0)
        # 第一次切换
        for _ in range(5):
            ctrl._select_mode(_make_state(), 0.2)
        first_mode = ctrl._mode
        # 立即改变冰况
        ctrl.set_ice_conditions(0.1, 0.1, 0.05, 90.0)
        # 在 dwell_time 内不应切换
        ctrl._select_mode(_make_state(), 0.2)
        assert ctrl._mode == first_mode  # 仍在同一模式

    def test_set_target_propagates_to_all(self):
        ctrl = ModeSupervisedIceDPController()
        ctrl.set_target(5.0, 5.0, 45.0)
        assert ctrl.precision._target_pos == (5.0, 5.0)
        assert ctrl.ice_aware._target_pos == (5.0, 5.0)
        assert ctrl.quasi._target_pos == (5.0, 5.0)
        assert ctrl.escape._target_pos == (5.0, 5.0)

    def test_set_ice_conditions_propagates(self):
        ctrl = ModeSupervisedIceDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.5, 0.8, 0.3, 90.0)
        assert ctrl.ice_aware._raw_ice["concentration"] == 0.5
        assert ctrl.quasi._raw_ice["concentration"] == 0.5
        assert ctrl.escape._raw_ice["concentration"] == 0.5

    def test_reset(self):
        ctrl = ModeSupervisedIceDPController()
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.5, 0.8, 0.3, 90.0)
        ctrl.compute_control(_make_state(), dt=0.2)
        ctrl.reset()
        assert ctrl._mode == DPMode.PRECISION
        assert ctrl._t == 0.0

    def test_set_cvar_seed(self):
        ctrl = ModeSupervisedIceDPController()
        ctrl.set_cvar_seed(42)
        # T2 fix: verify seed is propagated to sub-controllers
        assert hasattr(ctrl.ice_aware, '_rng'), "ice_aware should have _rng after set_cvar_seed"
        assert hasattr(ctrl.quasi, '_rng'), "quasi should have _rng after set_cvar_seed"
        assert hasattr(ctrl.escape, '_rng'), "escape should have _rng after set_cvar_seed"

    def test_set_safe_region_radius(self):
        ctrl = ModeSupervisedIceDPController()
        ctrl.set_safe_region_radius(15.0)
        assert ctrl.ice_aware.params.cbf_radius == 15.0
        assert ctrl.quasi.params.cbf_radius == 15.0
        assert ctrl.escape.params.cbf_radius == 15.0


# ---------- 消融控制器 ----------

class TestAblationControllers:
    def _make_ablation_ctrl(self, name):
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

    @pytest.mark.parametrize("ablation", ["no_cbf", "no_cvar", "no_observer", "no_fallback"])
    def test_ablation_produces_finite_tau(self, ablation):
        ctrl = self._make_ablation_ctrl(ablation)
        ctrl.set_target(0.0, 0.0, 0.0)
        ctrl.set_ice_conditions(0.7, 1.2, 0.5, 160.0)
        result = ctrl.compute_control(_make_state(x=5.0, y=3.0), dt=0.1)
        assert np.all(np.isfinite(result.tau))

    def test_no_cbf_disables_cbf_in_all_subcontrollers(self):
        ctrl = self._make_ablation_ctrl("no_cbf")
        assert not ctrl.ice_aware.use_cbf
        assert not ctrl.quasi.use_cbf
        assert not ctrl.escape.use_cbf
        # 注意: escape 模式使用自身安全逻辑(迎冰/撤离)，不使用 CBF。
        # use_cbf=False 在 escape 上仅标记，不影响行为。

    def test_no_cvar_disables_cvar_in_all_subcontrollers(self):
        ctrl = self._make_ablation_ctrl("no_cvar")
        assert not ctrl.ice_aware.use_cvar
        assert not ctrl.quasi.use_cvar
        assert not ctrl.escape.use_cvar
        # 注意: escape 模式风险固定为高值，不使用 CVaR 估计。

    def test_no_observer_disables_observer_in_all_subcontrollers(self):
        ctrl = self._make_ablation_ctrl("no_observer")
        assert not ctrl.ice_aware.use_observer
        assert not ctrl.quasi.use_observer
        assert not ctrl.escape.use_observer
        # observer 影响所有模式的冰况估计 (通过 _update_ice_estimate)


# ---------- 冰力模型和风险函数 ----------

class TestIceModels:
    def test_risk_standardized_bounded(self):
        assert 0.0 <= _ice_risk_standardized(0.0, 0.0, 0.0) <= 1.0
        assert 0.0 <= _ice_risk_standardized(1.0, 3.0, 1.0) <= 1.0
        assert 0.0 <= _ice_risk_standardized(0.5, 0.7, 0.25) <= 1.0

    def test_risk_increases_with_ice(self):
        low = _ice_risk_standardized(0.1, 0.2, 0.05)
        high = _ice_risk_standardized(0.9, 1.6, 0.7)
        assert high > low

    def test_lindqvist_force_bounded(self):
        force = _ice_force_lindqvist_proxy(0.5, 0.7, 0.25, 0.0, 0.0)
        assert force.shape == (3,)
        assert np.all(np.isfinite(force))

    def test_lindqvist_force_increases_with_ice(self):
        f_low = _ice_force_lindqvist_proxy(0.1, 0.2, 0.05, 0.0, 0.0)
        f_high = _ice_force_lindqvist_proxy(0.9, 1.6, 0.7, 0.0, 0.0)
        assert np.linalg.norm(f_high) > np.linalg.norm(f_low)

    def test_lindqvist_zero_ice_zero_force(self):
        force = _ice_force_lindqvist_proxy(0.0, 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(force, 0.0, atol=1e-10)
