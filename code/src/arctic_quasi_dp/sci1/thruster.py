"""推进器分配模型。

实现全回转推进器 (azimuthing thruster) 的力/力矩分配，包括：
- 推进器配置 (位置、方位角、最大推力)
- 推进器分配矩阵 (Thruster Allocation Matrix, TAM)
- 伪逆分配 + 饱和裁剪
- 推进器故障/退化注入
- 能耗模型

典型使用：
    config = ThrusterConfig.vessel_xuelong2()
    allocator = ThrusterAllocator(config)
    tau_cmd = np.array([500.0, 0.0, 2000.0])  # [Fx, Fy, Mz]
    thrusts, feasible = allocator.allocate(tau_cmd)
    actual_tau = allocator.resulting_tau(thrusts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
from numpy.typing import NDArray


@dataclass
class ThrusterUnit:
    """单个推进器参数。"""
    name: str
    x: float                        # 船体坐标系 x 位置 (m, 艏正)
    y: float                        # 船体坐标系 y 位置 (m, 右正)
    max_thrust: float               # 最大推力 (N)
    min_thrust: float = 0.0         # 最小推力 (N, 通常 0)
    azimuth: float = 0.0            # 固定方位角 (rad, 0=向艏)
    azimuthable: bool = False       # 是否可旋转
    efficiency: float = 1.0         # 效率因子 [0, 1]
    degraded: float = 1.0           # 退化因子 [0, 1], 1.0=正常
    faulty: bool = False            # 是否故障 (推力强制为 0)
    azimuth_locked: bool = False    # 方位角锁定 (不可旋转)


@dataclass
class ThrusterConfig:
    """推进器配置。"""
    name: str
    thrusters: List[ThrusterUnit]
    max_total_power_kw: float = 0.0  # 总功率限制 (kW), 0=无限制

    @classmethod
    def vessel_xuelong2(cls) -> "ThrusterConfig":
        """雪龙2号推进器配置 (简化)。

        雪龙2号有 2 个全回转推进器 (船艏) + 2 个固定推进器 (船艉)。
        参考: 中国船舶科学研究中心公开资料。
        """
        return cls(
            name="xuelong2",
            thrusters=[
                ThrusterUnit("bow_port",   x=55.0, y=-5.0, max_thrust=1000.0, azimuthable=True),
                ThrusterUnit("bow_stbd",   x=55.0, y=5.0,  max_thrust=1000.0, azimuthable=True),
                ThrusterUnit("stern_port",  x=-50.0, y=-4.5, max_thrust=800.0, azimuth=0.0),
                ThrusterUnit("stern_stbd",  x=-50.0, y=4.5,  max_thrust=800.0, azimuth=0.0),
            ],
            max_total_power_kw=8000.0,
        )

    @classmethod
    def generic_dp_vessel(cls) -> "ThrusterConfig":
        """通用 DP 船推进器配置。"""
        return cls(
            name="generic_dp",
            thrusters=[
                ThrusterUnit("fwd_port",   x=40.0, y=-6.0, max_thrust=800.0, azimuthable=True),
                ThrusterUnit("fwd_stbd",   x=40.0, y=6.0,  max_thrust=800.0, azimuthable=True),
                ThrusterUnit("aft_port",   x=-35.0, y=-5.0, max_thrust=700.0, azimuthable=True),
                ThrusterUnit("aft_stbd",   x=-35.0, y=5.0,  max_thrust=700.0, azimuthable=True),
                ThrusterUnit("tunnel",     x=50.0, y=0.0,  max_thrust=300.0, azimuth=np.pi / 2),
            ],
            max_total_power_kw=6000.0,
        )


class ThrusterAllocator:
    """推进器分配器。

    将期望的 [Fx, Fy, Mz] 分配到各推进器推力。
    支持全回转推进器的方位角优化。

    Args:
        config: 推进器配置
        max_azimuth_rate: 方位角速率限制 (rad/s, 0=无限制)
        tau_weight: 任务权重 [Fx, Fy, Mz], 默认等权 [1, 1, 1]
    """

    def __init__(
        self,
        config: ThrusterConfig,
        max_azimuth_rate: float = 0.0,
        tau_weight: Optional[NDArray[np.float64]] = None,
    ):
        self.config = config
        self.n = len(config.thrusters)
        self.max_azimuth_rate = max_azimuth_rate  # rad/s, 0=无限制
        self.tau_weight = np.asarray(tau_weight if tau_weight is not None else [1.0, 1.0, 1.0],
                                     dtype=np.float64).reshape(3,)
        self._prev_azimuths = np.array([t.azimuth for t in config.thrusters], dtype=np.float64)
        self._build_tam()

    def _build_tam(self) -> None:
        """构建推进器分配矩阵 (Thrust Allocation Matrix)。

        TAM: T @ u = tau
        其中 u = [u1, u2, ..., un] 为各推进器推力,
        tau = [Fx, Fy, Mz] 为期望的力/力矩。

        对于可旋转推进器，TAM 依赖于当前方位角。
        """
        self._tam = np.zeros((3, self.n), dtype=np.float64)
        self._update_tam()

    def _update_tam(self, azimuths: Optional[NDArray[np.float64]] = None) -> None:
        """更新 TAM (基于当前方位角)。

        注意: 退化因子 degraded 仅在推力上限中应用 (_clip_thrusts),
        不在 TAM 中重复应用, 避免 degraded² 效应。
        """
        for i, t in enumerate(self.config.thrusters):
            alpha = t.azimuth
            if azimuths is not None and t.azimuthable:
                alpha = azimuths[i]
            cos_a = math.cos(alpha)
            sin_a = math.sin(alpha)
            # 力分量: Fx = u*cos(alpha), Fy = u*sin(alpha)
            self._tam[0, i] = t.efficiency * cos_a
            self._tam[1, i] = t.efficiency * sin_a
            # 力矩: Mz = x*Fy - y*Fx = u*(x*sin(alpha) - y*cos(alpha))
            self._tam[2, i] = t.efficiency * (t.x * sin_a - t.y * cos_a)

    def optimize_azimuths(self, tau_desired: NDArray[np.float64]) -> NDArray[np.float64]:
        """优化可旋转推进器的方位角。

        使用简化的梯度方法：对于每个可旋转推进器，
        找到使其对 tau_desired 贡献最大的方位角。

        Args:
            tau_desired: 期望的 [Fx, Fy, Mz]

        Returns:
            各推进器的最优方位角数组
        """
        azimuths = np.array([t.azimuth for t in self.config.thrusters], dtype=np.float64)
        fx, fy = tau_desired[0], tau_desired[1]

        for i, t in enumerate(self.config.thrusters):
            if not t.azimuthable or t.azimuth_locked:
                continue
            # 最优方位角: 指向 (Fx, Fy) 方向, 同时考虑力臂
            # 简化: 直接朝期望力方向
            if abs(fx) > 1e-6 or abs(fy) > 1e-6:
                azimuths[i] = math.atan2(fy, fx)
            # 考虑力臂: 如果 x 很大 (船艏推进器), 偏航力矩主要由横向力产生
            if abs(t.x) > 20.0:  # 船艏/船艉推进器
                # 加入偏航修正
                mz = tau_desired[2]
                if abs(mz) > 1e-3 and abs(fx) + abs(fy) < abs(mz) / 10.0:
                    # 偏航主导: 调整方位角以产生更多横向力
                    sign_mz = 1.0 if mz > 0 else -1.0
                    sign_x = 1.0 if t.x > 0 else -1.0
                    azimuths[i] = sign_mz * sign_x * math.pi / 2

        return azimuths

    def allocate(
        self,
        tau_desired: NDArray[np.float64],
        optimize_azimuth: bool = True,
        dt: float = 0.1,
    ) -> Tuple[NDArray[np.float64], bool]:
        """将期望力/力矩分配到各推进器。

        使用伪逆 + 饱和裁剪迭代。

        Args:
            tau_desired: 期望的 [Fx, Fy, Mz] (N, N, N·m)
            optimize_azimuth: 是否优化方位角

        Returns:
            (各推进器推力数组, 是否可行)
        """
        tau_desired = np.asarray(tau_desired, dtype=np.float64).reshape(3,)

        # 1. 优化方位角
        if optimize_azimuth:
            azimuths = self.optimize_azimuths(tau_desired)
            # 方位角速率限制
            if self.max_azimuth_rate > 0:
                max_delta = self.max_azimuth_rate * dt
                for i in range(self.n):
                    if self.config.thrusters[i].azimuthable:
                        delta = azimuths[i] - self._prev_azimuths[i]
                        # 处理角度环绕
                        delta = (delta + math.pi) % (2 * math.pi) - math.pi
                        delta = np.clip(delta, -max_delta, max_delta)
                        azimuths[i] = self._prev_azimuths[i] + delta
            self._prev_azimuths = azimuths.copy()
            self._update_tam(azimuths)
        else:
            self._update_tam()

        # 保存 commanded azimuths (在 TAM 更新后)
        self._last_commanded_azimuths = np.array(
            [self._prev_azimuths[i] if self.config.thrusters[i].azimuthable
             else self.config.thrusters[i].azimuth
             for i in range(self.n)], dtype=np.float64,
        )

        # 2. 伪逆分配
        T = self._tam
        # 排除故障推进器
        active = np.array([not t.faulty for t in self.config.thrusters], dtype=bool)
        if not np.any(active):
            return np.zeros(self.n), False

        T_active = T[:, active]
        # 加权伪逆 (Wn: 推进器权重, Wt: 任务权重)
        Wn = np.diag([self.config.thrusters[i].max_thrust ** 2
                       for i in range(self.n) if active[i]])
        Wt = np.diag(self.tau_weight)

        # T_active_pinv = Wn @ T_active.T @ inv(T_active @ Wn @ T_active.T + lambda*I)
        TW = T_active @ Wn
        TWT = TW @ T_active.T
        # Tikhonov 正则化 (防奇异)
        lam = 1e-4 * np.trace(TWT) / 3.0
        TWT_reg = TWT + lam * np.eye(3)
        try:
            T_pinv = Wn @ T_active.T @ np.linalg.inv(TWT_reg)
        except np.linalg.LinAlgError:
            return np.zeros(self.n), False

        u_active = T_pinv @ Wt @ tau_desired

        # 3. 饱和裁剪 (方向感知迭代重分配)
        u_active = self._clip_thrusts(u_active, active, tau_desired=tau_desired)

        # 3b. 总功率约束
        self._last_power_kw_before_cap = 0.0
        self._last_power_scale_factor = 1.0
        if self.config.max_total_power_kw > 0:
            u_tmp = np.zeros(self.n)
            u_tmp[active] = u_active
            total_kw = self.total_power_kw(u_tmp)
            self._last_power_kw_before_cap = total_kw
            if total_kw > self.config.max_total_power_kw:
                # P ~ |u|^1.5 => scale = (Pmax / P)^(2/3)
                scale = (self.config.max_total_power_kw / max(total_kw, 1e-9)) ** (2.0 / 3.0)
                self._last_power_scale_factor = float(np.clip(scale, 0.0, 1.0))
                u_active = u_active * self._last_power_scale_factor

        # 4. 映射回全推进器数组
        u = np.zeros(self.n)
        u[active] = u_active

        # 5. 检查可行性
        actual_tau = self.resulting_tau(u)
        tau_err = np.linalg.norm(tau_desired - actual_tau)
        tau_norm = np.linalg.norm(tau_desired)
        if tau_norm < 1e-6:
            feasible = True  # zero desired tau is always feasible
        else:
            feasible = tau_err < 0.3 * tau_norm

        return u, feasible

    def _clip_thrusts(
        self,
        u: NDArray[np.float64],
        active: NDArray[np.bool_],
        tau_desired: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """方向感知迭代饱和裁剪。

        当推进器饱和时，固定其在限制值，然后重新求解剩余推进器的分配
        以最小化残差力矩 ||tau_desired - T @ u||。这比标量重分配更精确，
        因为它考虑了各推进器对 [Fx, Fy, Mz] 的方向贡献。

        Args:
            u: 推力数组 (仅含 active 推进器)
            active: 活跃推进器掩码
            tau_desired: 期望的广义力 [Fx, Fy, Mz]，用于方向感知重分配。
                        若为 None，回退到标量重分配。
        """
        u = u.copy()
        active_map = np.where(active)[0]

        for _ in range(3):
            u_before = u.copy()
            saturated = np.zeros(len(u), dtype=bool)
            for j in range(len(u)):
                t = self.config.thrusters[active_map[j]]
                t_max = t.max_thrust * t.degraded
                t_min = t.min_thrust
                if u[j] > t_max:
                    u[j] = t_max
                    saturated[j] = True
                elif u[j] < t_min:
                    u[j] = t_min
                    saturated[j] = True
            if not np.any(saturated):
                break

            remaining = np.where(~saturated)[0]
            if len(remaining) == 0:
                break

            if tau_desired is not None:
                # 方向感知重分配: 固定饱和推进器，重新求解剩余推进器
                T_active = self._tam[:, active]
                T_sat = T_active[:, saturated]
                T_rem = T_active[:, remaining]
                u_sat = u[saturated]

                # 残差力矩 = 期望力 - 饱和推进器贡献
                tau_residual = tau_desired - T_sat @ u_sat
                if np.linalg.norm(tau_residual) < 1e-6:
                    break

                # 对剩余推进器求解加权伪逆
                Wn_rem = np.diag([self.config.thrusters[active_map[j]].max_thrust ** 2
                                  for j in remaining])
                TWT = T_rem @ Wn_rem @ T_rem.T
                lam = 1e-4 * np.trace(TWT) / 3.0
                try:
                    T_pinv_rem = Wn_rem @ T_rem.T @ np.linalg.inv(TWT + lam * np.eye(3))
                    u_rem_new = T_pinv_rem @ tau_residual
                except np.linalg.LinAlgError:
                    break

                # 裁剪剩余推进器到限制内
                for idx, j in enumerate(remaining):
                    t = self.config.thrusters[active_map[j]]
                    u_rem_new[idx] = np.clip(u_rem_new[idx], t.min_thrust, t.max_thrust * t.degraded)

                u[remaining] = u_rem_new
            else:
                # 标量重分配 (无 tau_desired 时的回退)
                excess = sum(u_before[j] - u[j] for j in range(len(u)) if saturated[j])
                if abs(excess) < 1e-6:
                    break
                capacity = np.zeros(len(remaining))
                for idx, j in enumerate(remaining):
                    t = self.config.thrusters[active_map[j]]
                    t_max = t.max_thrust * t.degraded
                    t_min = t.min_thrust
                    capacity[idx] = max(0.0, (t_max - u[j]) if excess > 0 else (u[j] - t_min))
                total_cap = capacity.sum()
                if total_cap < 1e-6:
                    break
                for idx, j in enumerate(remaining):
                    u[j] += excess * (capacity[idx] / total_cap)

        return u

    def resulting_tau(self, thrusts: NDArray[np.float64]) -> NDArray[np.float64]:
        """计算给定推力数组产生的实际力/力矩。"""
        thrusts = np.asarray(thrusts, dtype=np.float64).reshape(self.n,)
        return self._tam @ thrusts

    def total_power_kw(self, thrusts: NDArray[np.float64]) -> float:
        """估算总功率 (kW)。

        简化模型: P = sum(|u_i|^1.5 / max_thrust^0.5) * 效率因子
        更精确的模型需要推进器特性曲线。
        """
        total = 0.0
        for i, t in enumerate(self.config.thrusters):
            if t.faulty:
                continue
            u = abs(thrusts[i])
            if u > 1e-6:
                # 功率 ∝ 推力^1.5 (简化螺旋桨理论)
                total += (u ** 1.5) / max(t.max_thrust ** 0.5, 1.0)
        return total / 1000.0  # 转换为 kW

    def thrust_saturation_ratio(self, thrusts: NDArray[np.float64]) -> float:
        """计算推力饱和比例。"""
        n_active = 0
        n_saturated = 0
        for i, t in enumerate(self.config.thrusters):
            if t.faulty:
                continue
            n_active += 1
            if abs(thrusts[i]) > 0.95 * t.max_thrust * t.degraded:
                n_saturated += 1
        return n_saturated / max(n_active, 1)

    def degrade_thruster(self, index: int, factor: float) -> None:
        """设置推进器退化因子。

        Args:
            index: 推进器索引
            factor: 退化因子 [0, 1], 1.0=正常, 0.0=完全退化
        """
        idx = int(index)
        if 0 <= idx < self.n:
            self.config.thrusters[idx].degraded = float(np.clip(factor, 0.0, 1.0))

    def fault_thruster(self, index: int, faulty: bool = True) -> None:
        """设置推进器故障状态。"""
        if 0 <= index < self.n:
            self.config.thrusters[index].faulty = faulty

    def reset(self) -> None:
        """重置所有推进器到正常状态。"""
        for t in self.config.thrusters:
            t.degraded = 1.0
            t.faulty = False
            t.azimuth_locked = False

    def get_diagnostics(self) -> Dict[str, float | int | bool | str]:
        """返回推进器状态诊断。"""
        n_faulty = sum(1 for t in self.config.thrusters if t.faulty)
        n_degraded = sum(1 for t in self.config.thrusters if t.degraded < 0.99 and not t.faulty)
        avg_degradation = np.mean([t.degraded for t in self.config.thrusters])
        return {
            "n_thrusters": self.n,
            "n_faulty": n_faulty,
            "n_degraded": n_degraded,
            "avg_degradation": float(avg_degradation),
            "config_name": self.config.name,
        }

    def get_actuator_trace(
        self,
        thrusts: NDArray[np.float64],
        tau_desired: NDArray[np.float64],
        tau_actual: NDArray[np.float64],
    ) -> Dict[str, float]:
        """返回 per-timestep 执行器证据字段, 用于 trace 审计。

        包含:
        - 各推进器 commanded/actual 推力、角度、锁定/饱和/速率限制状态
        - 总功率 (cap 前/后)、功率上限是否激活、缩放因子
        - 分配残差
        """
        thrusts = np.asarray(thrusts, dtype=np.float64).reshape(self.n,)
        trace: Dict[str, float] = {}

        for i, t in enumerate(self.config.thrusters):
            prefix = f"thruster_{i}"
            trace[f"{prefix}_commanded_thrust"] = float(thrusts[i])
            trace[f"{prefix}_actual_thrust"] = float(np.clip(thrusts[i],
                                                              t.min_thrust,
                                                              t.max_thrust * t.degraded))
            trace[f"{prefix}_azimuth_locked"] = 1.0 if t.azimuth_locked else 0.0
            trace[f"{prefix}_saturated"] = 1.0 if abs(thrusts[i]) > 0.95 * t.max_thrust * t.degraded else 0.0
            trace[f"{prefix}_faulty"] = 1.0 if t.faulty else 0.0
            trace[f"{prefix}_degraded"] = float(t.degraded)
            # 方位角证据
            commanded_az = getattr(self, '_last_commanded_azimuths', self._prev_azimuths)
            actual_angle = math.degrees(self._prev_azimuths[i]) if i < len(self._prev_azimuths) else 0.0
            commanded_angle = math.degrees(commanded_az[i]) if i < len(commanded_az) else 0.0
            trace[f"{prefix}_actual_angle_deg"] = float(actual_angle)
            trace[f"{prefix}_commanded_angle_deg"] = float(commanded_angle)
            trace[f"{prefix}_rate_limited"] = 0.0  # 由 sim_loop 的 _apply_rate_limits 设置

        # 功率证据: 使用 allocate() 中保存的 cap 前/后值
        total_kw = self.total_power_kw(thrusts)
        pre_cap_kw = getattr(self, '_last_power_kw_before_cap', total_kw)
        scale_factor = getattr(self, '_last_power_scale_factor', 1.0)
        cap_kw = self.config.max_total_power_kw
        power_cap_active = scale_factor < 1.0 - 1e-6

        trace["power_kw_before_cap"] = float(pre_cap_kw)
        trace["total_power_kw"] = float(total_kw)
        trace["power_cap_kw"] = float(cap_kw)
        trace["power_cap_active"] = 1.0 if power_cap_active else 0.0
        trace["power_scale_factor"] = float(scale_factor)

        tau_err = float(np.linalg.norm(np.asarray(tau_desired) - np.asarray(tau_actual)))
        trace["allocation_residual_norm"] = tau_err
        trace["thrust_saturation_ratio"] = self.thrust_saturation_ratio(thrusts)

        return trace


@dataclass
class ThrusterDegradationProfile:
    """推进器退化场景配置。"""
    name: str
    description: str
    degradations: Dict[str, float]  # thruster_name -> degradation_factor
    faults: List[str] = field(default_factory=list)  # faulty thruster names
    azimuth_locked_flag: bool = False    # 是否锁定方位角
    locked_angle_deg: float = 0.0        # 锁定角度 (度)

    @classmethod
    def no_fault(cls) -> "ThrusterDegradationProfile":
        return cls(name="no_fault", description="All thrusters nominal", degradations={})

    @classmethod
    def single_thruster_loss(cls, thruster_name: str = "stern_port") -> "ThrusterDegradationProfile":
        return cls(
            name=f"loss_{thruster_name}",
            description=f"Complete loss of {thruster_name}",
            degradations={},
            faults=[thruster_name],
        )

    @classmethod
    def bow_degradation(cls, factor: float = 0.5) -> "ThrusterDegradationProfile":
        return cls(
            name=f"bow_degraded_{factor}",
            description=f"Bow thrusters degraded to {factor*100:.0f}%",
            degradations={"bow_port": factor, "bow_stbd": factor, "fwd_port": factor, "fwd_stbd": factor},
        )

    @classmethod
    def severe_degradation(cls) -> "ThrusterDegradationProfile":
        return cls(
            name="severe",
            description="All thrusters at 50% + one complete loss",
            degradations={"bow_port": 0.5, "bow_stbd": 0.5, "stern_port": 0.5, "stern_stbd": 0.5,
                          "fwd_port": 0.5, "fwd_stbd": 0.5, "aft_port": 0.5, "aft_stbd": 0.5},
            faults=["stern_port"],
        )

    @classmethod
    def azimuth_locked_profile(cls, angle_deg: float = 0.0) -> "ThrusterDegradationProfile":
        """方位角锁定: 所有可旋转推进器锁定在指定角度 (默认向前)。"""
        return cls(
            name="azimuth_locked",
            description=f"Azimuthing thrusters locked at {angle_deg:.0f}°",
            degradations={},
            faults=[],
            azimuth_locked_flag=True,
            locked_angle_deg=angle_deg,
        )

    def apply(self, allocator: ThrusterAllocator) -> None:
        """将退化配置应用到分配器。"""
        allocator.reset()
        for t in allocator.config.thrusters:
            if t.name in self.degradations:
                t.degraded = self.degradations[t.name]
            if t.name in self.faults:
                t.faulty = True
            if self.azimuth_locked_flag and t.azimuthable:
                t.azimuth_locked = True
                t.azimuth = math.radians(self.locked_angle_deg)
