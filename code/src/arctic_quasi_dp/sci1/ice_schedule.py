"""时变冰况调度器。

提供多种时变冰况模型：
- 线性渐变 (ramp)
- 阶跃变化 (step)
- 周期变化 (sinusoidal)
- 随机扰动 (random walk)
- 分段组合 (piecewise)

与 sim_loop.py 配合使用，在每个仿真 timestep 更新冰况。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import math

import numpy as np


@dataclass
class IceState:
    """时刻 t 的冰况状态。

    角度约定: drift_direction 以度为单位 (用户接口)。
    内部计算使用 drift_direction_rad 属性 (弧度)。
    """
    concentration: float   # [0, 1]
    thickness: float       # m
    drift_speed: float     # m/s
    drift_direction: float # deg (用户接口)

    @property
    def drift_direction_rad(self) -> float:
        """漂移方向 (弧度), 供内部计算使用。"""
        return math.radians(self.drift_direction)

    def to_dict(self) -> Dict[str, float]:
        return {
            "concentration": self.concentration,
            "thickness": self.thickness,
            "drift_speed": self.drift_speed,
            "drift_direction": self.drift_direction,
        }


def drift_dir_deg_to_rad(deg: float) -> float:
    """将冰漂移方向从度转换为弧度 — 角度单位转换的唯一规范入口。"""
    return math.radians(deg)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _wrap_angle_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class IceSchedule:
    """冰况调度基类。所有时变冰况模型的接口。"""

    def at(self, t: float) -> IceState:
        """返回时刻 t 的冰况。子类必须实现。"""
        raise NotImplementedError

    def __call__(self, t: float) -> Dict[str, float]:
        """方便作为函数调用。"""
        return self.at(t).to_dict()


class ConstantIce(IceSchedule):
    """常数冰况 (无时变)。"""

    def __init__(self, concentration: float, thickness: float, drift_speed: float, drift_direction: float):
        self._state = IceState(concentration, thickness, drift_speed, drift_direction)

    def at(self, t: float) -> IceState:
        return self._state


class LinearRampIce(IceSchedule):
    """线性渐变冰况。

    参数从 (c0, h0, v0, dir0) 线性变化到 (c1, h1, v1, dir1)。
    """

    def __init__(
        self,
        c0: float, h0: float, v0: float, dir0: float,
        c1: float, h1: float, v1: float, dir1: float,
        duration: float,
    ):
        self.c0, self.h0, self.v0, self.dir0 = c0, h0, v0, dir0
        self.c1, self.h1, self.v1, self.dir1 = c1, h1, v1, dir1
        self.duration = max(duration, 1e-6)

    def at(self, t: float) -> IceState:
        frac = float(np.clip(t / self.duration, 0.0, 1.0))
        # H9 fix: shortest-path angle interpolation (handle 0/360 wraparound)
        diff_dir = (self.dir1 - self.dir0 + 180.0) % 360.0 - 180.0
        interp_dir = (self.dir0 + frac * diff_dir) % 360.0
        return IceState(
            concentration=float(np.clip(_lerp(self.c0, self.c1, frac), 0.0, 1.0)),
            thickness=max(0.0, _lerp(self.h0, self.h1, frac)),
            drift_speed=max(0.0, _lerp(self.v0, self.v1, frac)),
            drift_direction=interp_dir,
        )


class StepIce(IceSchedule):
    """阶跃变化冰况。

    在 t_change 时刻从状态 A 突变到状态 B。
    """

    def __init__(
        self,
        c_a: float, h_a: float, v_a: float, dir_a: float,
        c_b: float, h_b: float, v_b: float, dir_b: float,
        t_change: float,
    ):
        self.state_a = IceState(c_a, h_a, v_a, dir_a)
        self.state_b = IceState(c_b, h_b, v_b, dir_b)
        self.t_change = t_change

    def at(self, t: float) -> IceState:
        return self.state_b if t >= self.t_change else self.state_a


class SinusoidalIce(IceSchedule):
    """周期变化冰况。

    参数围绕均值做正弦振荡。
    """

    def __init__(
        self,
        c_mean: float, c_amp: float,
        h_mean: float, h_amp: float,
        v_mean: float, v_amp: float,
        dir_mean: float, dir_amp: float,
        period: float,
        phase: float = 0.0,
    ):
        self.c_mean, self.c_amp = c_mean, c_amp
        self.h_mean, self.h_amp = h_mean, h_amp
        self.v_mean, self.v_amp = v_mean, v_amp
        self.dir_mean, self.dir_amp = dir_mean, dir_amp
        self.period = max(period, 1e-6)
        self.phase = phase

    def at(self, t: float) -> IceState:
        s = math.sin(2.0 * math.pi * t / self.period + self.phase)
        return IceState(
            concentration=float(np.clip(self.c_mean + self.c_amp * s, 0.0, 1.0)),
            thickness=max(0.0, self.h_mean + self.h_amp * s),
            drift_speed=max(0.0, self.v_mean + self.v_amp * s),
            drift_direction=_wrap_angle_deg(self.dir_mean + self.dir_amp * s),
        )


class RandomWalkIce(IceSchedule):
    """随机游走冰况 (带均值回复)。

    冰况参数围绕标称值做 Ornstein-Uhlenbeck 随机过程。
    使用确定性种子保证可复现。
    """

    def __init__(
        self,
        c_nom: float, h_nom: float, v_nom: float, dir_nom: float,
        c_vol: float = 0.05, h_vol: float = 0.05, v_vol: float = 0.03, dir_vol: float = 5.0,
        mean_reversion: float = 0.1,
        dt_sample: float = 1.0,
        seed: int = 2026,
    ):
        self.c_nom = c_nom
        self.h_nom = h_nom
        self.v_nom = v_nom
        self.dir_nom = dir_nom
        self.c_vol = c_vol
        self.h_vol = h_vol
        self.v_vol = v_vol
        self.dir_vol = dir_vol
        self.mean_reversion = mean_reversion
        self.dt_sample = dt_sample
        self._rng = np.random.default_rng(seed)
        self._cache: Dict[int, IceState] = {}

    def _generate_at(self, step: int) -> IceState:
        """迭代生成冰况 (避免递归深度限制)。"""
        # 找到最近的已缓存 step
        start = 0
        prev_state = IceState(self.c_nom, self.h_nom, self.v_nom, self.dir_nom)
        for s in range(step, -1, -1):
            if s in self._cache:
                prev_state = self._cache[s]
                start = s
                break

        alpha = self.mean_reversion
        sqrt_dt = math.sqrt(self.dt_sample)

        for s in range(start + 1, step + 1):
            noise_c = self._rng.normal(0, self.c_vol)
            noise_h = self._rng.normal(0, self.h_vol)
            noise_v = self._rng.normal(0, self.v_vol)
            noise_dir = self._rng.normal(0, self.dir_vol)

            c = prev_state.concentration + alpha * (self.c_nom - prev_state.concentration) + noise_c * sqrt_dt
            h = prev_state.thickness + alpha * (self.h_nom - prev_state.thickness) + noise_h * sqrt_dt
            v = prev_state.drift_speed + alpha * (self.v_nom - prev_state.drift_speed) + noise_v * sqrt_dt
            d = prev_state.drift_direction + alpha * (self.dir_nom - prev_state.drift_direction) + noise_dir * sqrt_dt

            prev_state = IceState(
                concentration=float(np.clip(c, 0.0, 1.0)),
                thickness=max(0.0, h),
                drift_speed=max(0.0, v),
                drift_direction=_wrap_angle_deg(d),
            )
            self._cache[s] = prev_state

        return prev_state

    def at(self, t: float) -> IceState:
        if not math.isfinite(t):
            t = 0.0
        step = int(t / self.dt_sample)
        if step not in self._cache:
            self._cache[step] = self._generate_at(step)
        return self._cache[step]


class PiecewiseIce(IceSchedule):
    """分段组合冰况。

    按时间段应用不同的 IceSchedule。
    """

    def __init__(self, segments: List[Tuple[float, IceSchedule]]):
        """
        Args:
            segments: [(start_time, schedule), ...] 列表，按时间排序。
                      最后一个 segment 的 schedule 持续到仿真结束。
        """
        self.segments = sorted(segments, key=lambda x: x[0])

    def at(self, t: float) -> IceState:
        schedule = self.segments[0][1]
        for start_t, seg_schedule in self.segments:
            if t >= start_t:
                schedule = seg_schedule
            else:
                break
        return schedule.at(t)


# ---------- 预定义场景冰况调度 ----------

def make_b4_ice_schedule(duration: float) -> LinearRampIce:
    """B4 非平稳冰况: 密集度从 0.3 线性渐变到 0.8。"""
    return LinearRampIce(
        c0=0.3, h0=0.7, v0=0.15, dir0=120.0,
        c1=0.8, h1=1.1, v1=0.55, dir1=160.0,
        duration=duration,
    )


def make_step_jump_ice(duration: float) -> StepIce:
    """模拟冰况阶跃突变 (如冰脊遭遇)。"""
    return StepIce(
        c_a=0.4, h_a=0.6, v_a=0.2, dir_a=90.0,
        c_b=0.85, h_b=1.4, v_b=0.6, dir_b=180.0,
        t_change=duration * 0.4,
    )


def make_oscillating_ice(duration: float) -> SinusoidalIce:
    """模拟潮汐/风驱冰况周期变化。"""
    return SinusoidalIce(
        c_mean=0.5, c_amp=0.15,
        h_mean=0.8, h_amp=0.1,
        v_mean=0.3, v_amp=0.1,
        dir_mean=135.0, dir_amp=20.0,
        period=duration / 3.0,
    )
