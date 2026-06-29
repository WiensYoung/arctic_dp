"""Actuator-aware feasible-set approximations for safety-filter QPs.

The main implementation intentionally separates two levels:

* :class:`BoxFeasibleSet` is the conservative baseline used by the first
  Soft-HOCBF-QP implementation.
* :class:`ActuatorAwareFeasibleSet` provides a fixed-size, mode-cached,
  conservative inner approximation of the allocatable generalized-force set.

Important limitation
--------------------
This module builds a proxy-scale inner polygon from conservative radial feasible vertices.
When requested, each vertex is additionally checked against the project
ThrusterAllocator and shrunk until the allocator residual is acceptable.  It is
intended for the simplified SCI1 benchmark and must not be advertised as a
full-scale thrust-allocation proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import copy

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FeasibleSetConstraints:
    """Fixed-shape linear constraints ``l <= A @ tau <= u``."""

    A: NDArray[np.float64]
    l: NDArray[np.float64]
    u: NDArray[np.float64]
    mode: str
    feasible_set_type: str
    active_rows: int


class BoxFeasibleSet:
    """Box constraints on generalized force.

    ``|tau_x| <= max_force_x``, ``|tau_y| <= max_force_y``,
    ``|tau_n| <= max_moment_n``.
    """

    def __init__(
        self,
        max_force_x: float = 3000.0,
        max_force_y: float = 3000.0,
        max_moment_n: float = 100000.0,
        use_allocator_feasibility_check: bool = True,
        allocator_residual_tol: float = 0.25,
    ):
        self.max_force_x = float(max_force_x)
        self.max_force_y = float(max_force_y)
        self.max_moment_n = float(max_moment_n)

    def get_bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(tau_min, tau_max)`` box bounds."""
        tau_min = np.array([-self.max_force_x, -self.max_force_y, -self.max_moment_n], dtype=np.float64)
        tau_max = np.array([self.max_force_x, self.max_force_y, self.max_moment_n], dtype=np.float64)
        return tau_min, tau_max

    def get_constraints(self, *_args: Any, **_kwargs: Any) -> FeasibleSetConstraints:
        """Return box constraints as fixed linear inequalities."""
        A = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        l = np.array([-self.max_force_x, -self.max_force_y, -self.max_moment_n], dtype=np.float64)
        u = np.array([self.max_force_x, self.max_force_y, self.max_moment_n], dtype=np.float64)
        return FeasibleSetConstraints(A=A, l=l, u=u, mode="box", feasible_set_type="box", active_rows=3)


class ActuatorAwareFeasibleSet:
    """Mode-cached conservative inner polygon for generalized-force constraints.

    This class does **not** create a finite support-function outer
    approximation.  Instead it constructs conservative radial vertices that are
    treated as allocatable proxy vertices, takes their convex hull (ordered
    radial polygon), and converts that inner polygon into a fixed-size
    half-space representation.  The output shape is constant so that an OSQP
    implementation can keep the sparsity pattern stable.
    """

    def __init__(
        self,
        n_vertices: int = 32,
        safety_factor: float = 0.90,
        max_facets: int | None = None,
        max_force_x: float = 3000.0,
        max_force_y: float = 3000.0,
        max_moment_n: float = 100000.0,
        use_allocator_feasibility_check: bool = True,
        allocator_residual_tol: float = 0.25,
    ):
        if n_vertices < 8:
            raise ValueError("n_vertices must be at least 8 for a useful polygon")
        self.n_vertices = int(n_vertices)
        self.safety_factor = float(np.clip(safety_factor, 0.05, 1.0))
        # max_facets=0 is treated as "use all vertices" (default); otherwise clamp to [4, n_vertices]
        self.max_facets = int(max_facets) if max_facets and max_facets > 0 else int(n_vertices)
        self.max_facets = max(4, min(self.max_facets, int(n_vertices)))
        self.max_force_x = float(max_force_x)
        self.max_force_y = float(max_force_y)
        self.max_moment_n = float(max_moment_n)
        self.use_allocator_feasibility_check = bool(use_allocator_feasibility_check)
        self.allocator_residual_tol = float(np.clip(allocator_residual_tol, 0.05, 0.60))
        self._cache: Dict[str, tuple[NDArray[np.float64], NDArray[np.float64], int]] = {}
        theta = np.linspace(0.0, 2.0 * np.pi, self.n_vertices, endpoint=False)
        self._directions = np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float64)

    @property
    def n_rows(self) -> int:
        """Number of fixed constraint rows, including two yaw-box rows."""
        return self.max_facets + 2

    def _mode_shape_radius(self, mode: str, direction_xy: NDArray[np.float64]) -> float:
        """Return a conservative radial force bound for a discrete actuator mode.

        The values are intentionally conservative proxy-scale bounds.  They are
        not full-scale allocation certificates, but they provide a tested inner
        approximation for method experiments.
        """
        mode = (mode or "nominal").lower()
        dx, dy = float(direction_xy[0]), float(direction_xy[1])
        # Base elliptical authority: use the smaller axis as a safe radial bound.
        base = min(self.max_force_x, self.max_force_y)

        if mode in ("no_fault", "nominal", "generic_dp"):
            factor = 1.00
        elif "bow_degraded_0.7" in mode:
            # Bow thruster degraded by 30%; lateral/forward authority shrinks
            # non-uniformly.  Keep a conservative direction-dependent factor.
            factor = 0.72 - 0.08 * max(0.0, dx)
        elif "bow_degraded_0.5" in mode:
            factor = 0.55 - 0.10 * max(0.0, dx)
        elif "azimuth_locked" in mode or "locked" in mode:
            # Locked azimuth changes feasible-set shape, not just scale.
            # The proxy keeps stronger longitudinal authority and reduced sway.
            factor = 0.55 + 0.20 * abs(dx) - 0.18 * abs(dy)
        elif "power_limited" in mode or "power" in mode:
            factor = 0.45
        elif "rate_limited" in mode or "rate" in mode:
            factor = 0.60
        else:
            factor = 0.80
        return max(1.0, base * float(np.clip(factor, 0.20, 1.0)))

    def _make_allocator_for_mode(self, mode: str):
        """Build a fresh project allocator configured for a discrete fault mode.

        This is used only for conservative vertex certification.  A fresh
        allocator avoids hidden dependence on previous azimuth/rate states.
        """
        from ..thruster import ThrusterAllocator, ThrusterConfig, ThrusterDegradationProfile

        mode_l = (mode or "nominal").lower()
        config = copy.deepcopy(ThrusterConfig.generic_dp_vessel())
        if "power" in mode_l:
            # Match the proxy C7 power-limited profile used by runner.py.
            config.max_total_power_kw = 0.015
        allocator = ThrusterAllocator(config)
        if "bow_degraded_0.7" in mode_l:
            ThrusterDegradationProfile.bow_degradation(0.7).apply(allocator)
        elif "bow_degraded_0.5" in mode_l:
            ThrusterDegradationProfile.bow_degradation(0.5).apply(allocator)
        elif "azimuth_locked" in mode_l or "locked" in mode_l:
            ThrusterDegradationProfile.azimuth_locked_profile().apply(allocator)
        return allocator

    def _allocator_accepts_tau(self, mode: str, tau_xy: NDArray[np.float64]) -> bool:
        """Return whether the project allocator can realize a candidate XY force.

        The check is intentionally conservative: the allocator must report
        feasibility and the resulting generalized force must remain within a
        relative residual tolerance.  Only the XY part is certified here; yaw is
        constrained separately by a conservative box.
        """
        tau = np.array([float(tau_xy[0]), float(tau_xy[1]), 0.0], dtype=np.float64)
        norm = max(float(np.linalg.norm(tau[:2])), 1.0)
        allocator = self._make_allocator_for_mode(mode)
        thrusts, feasible = allocator.allocate(tau, dt=0.1)
        tau_actual = allocator.resulting_tau(thrusts)
        rel_resid = float(np.linalg.norm(tau_actual[:2] - tau[:2]) / norm)
        return bool(feasible and rel_resid <= self.allocator_residual_tol)

    def _certified_radius(self, mode: str, direction_xy: NDArray[np.float64], r_initial: float) -> float:
        """Shrink a radial vertex until the allocator certifies it."""
        if not self.use_allocator_feasibility_check:
            return float(r_initial)
        lo = 0.0
        hi = max(1.0, float(r_initial))
        # If even the initial radius is feasible, keep it.  Otherwise binary
        # search for a conservative feasible radius.
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            candidate = mid * np.asarray(direction_xy, dtype=np.float64)
            if self._allocator_accepts_tau(mode, candidate):
                lo = mid
            else:
                hi = mid
        # 返回通过可行性检查的最大半径 (保守: lo 可小于 1.0)
        # 注意: lo=0.0 表示该方向无法分配任何力，顶点在原点使可行集收缩
        return lo

    def _build_vertices(self, mode: str) -> NDArray[np.float64]:
        vertices = []
        for d in self._directions:
            r_shape = self._mode_shape_radius(mode, d)
            r_cert = self._certified_radius(mode, d, r_shape)
            vertices.append(self.safety_factor * r_cert * d)
        return np.asarray(vertices, dtype=np.float64)

    @staticmethod
    def _vertices_to_halfspaces(vertices_xy: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert CCW polygon vertices to ``A_xy x <= b``."""
        vertices_xy = np.asarray(vertices_xy, dtype=np.float64)
        n = len(vertices_xy)
        A_rows = []
        b_rows = []
        for i in range(n):
            p0 = vertices_xy[i]
            p1 = vertices_xy[(i + 1) % n]
            edge = p1 - p0
            normal = np.array([edge[1], -edge[0]], dtype=np.float64)  # outward for CCW polygon
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                continue
            normal /= norm
            A_rows.append([normal[0], normal[1], 0.0])
            b_rows.append(float(normal @ p0))
        return np.asarray(A_rows, dtype=np.float64), np.asarray(b_rows, dtype=np.float64)

    def _cached_halfspaces(self, mode: str) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        key = (mode or "nominal").lower()
        if key in self._cache:
            return self._cache[key]
        vertices = self._build_vertices(key)
        A_xy, b_xy = self._vertices_to_halfspaces(vertices)
        active = min(len(b_xy), self.max_facets)
        A = np.zeros((self.n_rows, 3), dtype=np.float64)
        b = np.full(self.n_rows, np.inf, dtype=np.float64)
        if active:
            A[:active, :] = A_xy[:active, :]
            b[:active] = b_xy[:active]
        # Fixed yaw box rows are always active.
        A[self.max_facets, 2] = 1.0
        b[self.max_facets] = self.max_moment_n * self.safety_factor
        A[self.max_facets + 1, 2] = -1.0
        b[self.max_facets + 1] = self.max_moment_n * self.safety_factor
        active_total = active + 2
        self._cache[key] = (A, b, active_total)
        return self._cache[key]

    def get_constraints(
        self,
        actuator_state: Any = None,
        mode: str = "nominal",
        power_scale_factor: float = 1.0,
    ) -> FeasibleSetConstraints:
        """Return fixed-shape linear constraints for the requested mode.

        Invalid/padded rows use ``l=-inf`` and ``u=inf`` to preserve the row
        count without affecting the QP.
        """
        # Allow caller to pass a dict-like state with mode/power scale.
        if isinstance(actuator_state, dict):
            mode = str(actuator_state.get("mode", mode))
            power_scale_factor = float(actuator_state.get("power_scale_factor", power_scale_factor))

        A, b_nominal, active_total = self._cached_halfspaces(mode)
        power_scale = float(np.clip(power_scale_factor, 0.0, 1.0))
        # If the project power model is P ~ |T|^1.5, thrust scales as P^(2/3).
        thrust_scale = max(0.0, power_scale) ** (2.0 / 3.0) if power_scale < 1.0 else 1.0
        u = b_nominal.copy()
        finite = np.isfinite(u)
        u[finite] = u[finite] * thrust_scale
        l = np.full_like(u, -np.inf, dtype=np.float64)
        return FeasibleSetConstraints(
            A=A.copy(), l=l, u=u, mode=mode, feasible_set_type="inner_polygon_proxy", active_rows=active_total
        )

    def fallback_box(self) -> BoxFeasibleSet:
        """Return a conservative box fallback."""
        return BoxFeasibleSet(
            max_force_x=self.max_force_x * self.safety_factor,
            max_force_y=self.max_force_y * self.safety_factor,
            max_moment_n=self.max_moment_n * self.safety_factor,
        )
