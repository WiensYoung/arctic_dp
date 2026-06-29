"""实验构建器 — 构建仿真器实例。"""

from __future__ import annotations

from typing import Optional

from ..simulation.simulator import Simulator, SimulationConfig


class ExperimentBuilder:
    """构建实验用仿真器。"""

    def build_simulator(self, safe_region_radius: float = 10.0) -> Simulator:
        return Simulator(safe_region_radius=safe_region_radius)
