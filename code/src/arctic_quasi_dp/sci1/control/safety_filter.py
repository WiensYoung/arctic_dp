"""Soft-HOCBF-QP Safety Filter.

This is a proxy-scale relative-degree-2 Soft-HOCBF implementation for the
simplified 3-DOF benchmark.  It is not a full-scale ship safety proof.

The filter solves a small QP at each timestep:

    min 0.5 * ||tau - tau_des||_W^2 + 0.5 * p_delta * delta^2
    s.t. a_hocbf @ tau + delta >= b_hocbf
         delta >= 0
         feasible-set constraints on tau

Primary production backend: OSQP when installed.  The SciPy fallback is kept for
base-environment tests and is always labelled as a fallback in trace output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
from numpy.typing import NDArray

from .actuator_feasible_set import ActuatorAwareFeasibleSet, BoxFeasibleSet, FeasibleSetConstraints
from .hocbf import HOCBFParams, compute_hocbf_constraint


@dataclass
class SafetyFilterResult:
    """Safety filter output — fully auditable."""

    tau_des: NDArray[np.float64]
    tau_safe: NDArray[np.float64]

    active: bool = False
    qp_success: bool = True
    infeasible: bool = False
    status: str = "disabled"
    solve_time_ms: float = 0.0
    iterations: int = 0
    solver_backend: str = "disabled"

    slack: float = 0.0
    # safety_set_h is h(x)=R^2-||p-p_ref||^2.
    safety_set_h: float = 0.0
    safety_set_h_dot: float = 0.0
    # hocbf_margin is the actual QP inequality margin:
    # a_hocbf @ tau_safe + slack - b_hocbf.
    hocbf_margin: float = 0.0
    hocbf_nominal_margin: float = 0.0
    hocbf_robust_disturbance_margin: float = 0.0
    hocbf_disturbance_accel_bound_mps2: float = 0.0
    hocbf_a_norm: float = 0.0
    hocbf_b: float = 0.0
    hocbf_b_nominal: float = 0.0
    hocbf_soft_certificate: bool = False
    correction_norm: float = 0.0

    solver_setup_count: int = 0
    solver_update_count: int = 0
    osqp_reused_factorization: bool = False

    risk_level: float = 0.0
    risk_scale: float = 1.0
    alpha1: float = 0.0
    alpha2: float = 0.0

    constraint_mode: str = "box"
    feasible_set_type: str = "none"
    actuator_mode: str = "nominal"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dict for trace logging."""
        return {
            "tau_des_x": float(self.tau_des[0]),
            "tau_des_y": float(self.tau_des[1]),
            "tau_des_n": float(self.tau_des[2]),
            "tau_safe_x": float(self.tau_safe[0]),
            "tau_safe_y": float(self.tau_safe[1]),
            "tau_safe_n": float(self.tau_safe[2]),
            "safety_filter_active": 1.0 if self.active else 0.0,
            "safety_filter_qp_success": 1.0 if self.qp_success else 0.0,
            "safety_filter_infeasible": 1.0 if self.infeasible else 0.0,
            "safety_filter_status": self.status,
            "safety_filter_solver_backend": self.solver_backend,
            "safety_filter_solve_time_ms": float(self.solve_time_ms),
            "safety_filter_iter": float(self.iterations),
            "safety_filter_slack": float(self.slack),
            "safety_filter_slack_active": 1.0 if self.slack > 1e-9 else 0.0,
            "safety_set_h": float(self.safety_set_h),
            "safety_set_h_dot": float(self.safety_set_h_dot),
            "safety_filter_hocbf_margin": float(self.hocbf_margin),
            "hocbf_constraint_margin": float(self.hocbf_margin),
            "hocbf_nominal_constraint_margin": float(self.hocbf_nominal_margin),
            "hocbf_robust_disturbance_margin": float(self.hocbf_robust_disturbance_margin),
            "hocbf_disturbance_accel_bound_mps2": float(self.hocbf_disturbance_accel_bound_mps2),
            "hocbf_a_norm": float(self.hocbf_a_norm),
            "hocbf_b": float(self.hocbf_b),
            "hocbf_b_nominal": float(self.hocbf_b_nominal),
            "hocbf_soft_certificate": 1.0 if self.hocbf_soft_certificate else 0.0,
            "safety_filter_correction_norm": float(self.correction_norm),
            "safety_filter_risk_level": float(self.risk_level),
            "safety_filter_risk_scale": float(self.risk_scale),
            "safety_filter_alpha1": float(self.alpha1),
            "safety_filter_alpha2": float(self.alpha2),
            "safety_filter_constraint_mode": self.constraint_mode,
            "safety_filter_feasible_set_type": self.feasible_set_type,
            "actuator_feasible_mode": self.actuator_mode,
            "safety_filter_solver_setup_count": float(self.solver_setup_count),
            "safety_filter_solver_update_count": float(self.solver_update_count),
            "safety_filter_osqp_reused_factorization": 1.0 if self.osqp_reused_factorization else 0.0,
        }


class SoftHOCBFSafetyFilter:
    """Soft-HOCBF-QP safety filter with box or proxy polygon constraints."""

    def __init__(
        self,
        hocbf_params: Optional[HOCBFParams] = None,
        max_force_x: float = 3000.0,
        max_force_y: float = 3000.0,
        max_moment_n: float = 100000.0,
        tau_weight: Optional[NDArray[np.float64]] = None,
        slack_weight: float = 10000.0,
        risk_gain: float = 0.0,
        constraint_mode: str = "box",
        require_osqp: bool = False,
    ):
        self.hocbf_params = hocbf_params or HOCBFParams()
        self.max_force_x = float(max_force_x)
        self.max_force_y = float(max_force_y)
        self.max_moment_n = float(max_moment_n)
        self.tau_weight = np.asarray(
            tau_weight if tau_weight is not None else [1.0, 1.0, 1.0], dtype=np.float64
        )
        self.slack_weight = float(slack_weight)
        self.risk_gain = float(risk_gain)
        self.constraint_mode = str(constraint_mode or "box")
        self.require_osqp = bool(require_osqp)
        self.actuator_mode = "nominal"
        self.power_scale_factor = 1.0
        self._box = BoxFeasibleSet(self.max_force_x, self.max_force_y, self.max_moment_n)
        self._polygon = ActuatorAwareFeasibleSet(
            n_vertices=32,
            safety_factor=0.90,
            max_force_x=self.max_force_x,
            max_force_y=self.max_force_y,
            max_moment_n=self.max_moment_n,
        )
        self._osqp_available = self._check_osqp_available()
        self._osqp_solver = None
        self._osqp_shape_key = None
        self._osqp_setup_count = 0
        self._osqp_update_count = 0
        if self.require_osqp and not self._osqp_available:
            # Do not fail construction.  Mark the per-step result as unavailable
            # so configs/tests can still report the issue honestly.
            pass

    @staticmethod
    def _check_osqp_available() -> bool:
        from .qp_solver import check_osqp_available
        return check_osqp_available()

    def set_actuator_mode(self, mode: str, power_scale_factor: float = 1.0) -> None:
        """Set the discrete/continuous feasible-set mode used by polygon constraints."""
        self.actuator_mode = str(mode or "nominal")
        self.power_scale_factor = float(np.clip(power_scale_factor, 0.0, 1.0))

    def _feasible_constraints(self) -> FeasibleSetConstraints:
        if self.constraint_mode in ("polygon", "inner_polygon", "actuator_polygon"):
            return self._polygon.get_constraints(
                mode=self.actuator_mode, power_scale_factor=self.power_scale_factor
            )
        return self._box.get_constraints()

    def _solve_qp_scipy(
        self,
        tau_des: NDArray[np.float64],
        a_hocbf: NDArray[np.float64],
        b_hocbf: float,
        feasible: FeasibleSetConstraints,
    ) -> tuple[NDArray[np.float64], float, str, int, str]:
        """Fallback solver with analytical slack and linear feasible constraints."""
        from scipy.optimize import minimize

        tau_des = np.asarray(tau_des, dtype=np.float64).reshape(3,)
        A = np.asarray(feasible.A, dtype=np.float64)
        l = np.asarray(feasible.l, dtype=np.float64)
        u = np.asarray(feasible.u, dtype=np.float64)
        finite_l = np.isfinite(l)
        finite_u = np.isfinite(u)
        weights = self.tau_weight.astype(np.float64)
        slack_weight = float(self.slack_weight)
        a = np.asarray(a_hocbf, dtype=np.float64).reshape(3,)
        b = float(b_hocbf)

        # Initial point: clipped to simple box, then scaled until it satisfies
        # finite upper halfspaces.  This keeps SLSQP stable for proxy polygons.
        tau_min = np.array([-self.max_force_x, -self.max_force_y, -self.max_moment_n], dtype=np.float64)
        tau_max = np.array([self.max_force_x, self.max_force_y, self.max_moment_n], dtype=np.float64)
        tau0 = np.clip(tau_des, tau_min, tau_max)
        for _ in range(20):
            vals = A @ tau0
            ok_u = np.all(vals[finite_u] <= u[finite_u] + 1e-7) if np.any(finite_u) else True
            ok_l = np.all(vals[finite_l] >= l[finite_l] - 1e-7) if np.any(finite_l) else True
            if ok_u and ok_l:
                break
            tau0 *= 0.75

        def slack_for(tau: NDArray[np.float64]) -> float:
            return max(0.0, b - float(a @ tau))

        def objective(tau: NDArray[np.float64]) -> float:
            d_tau = tau - tau_des
            slack = slack_for(tau)
            return float(0.5 * np.sum(weights * d_tau * d_tau) + 0.5 * slack_weight * slack * slack)

        constraints = []
        for row, lo, hi in zip(A, l, u):
            if np.isfinite(hi):
                constraints.append({"type": "ineq", "fun": lambda x, row=row, hi=hi: float(hi - row @ x)})
            if np.isfinite(lo):
                constraints.append({"type": "ineq", "fun": lambda x, row=row, lo=lo: float(row @ x - lo)})

        result = minimize(
            objective,
            tau0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-9, "disp": False},
        )
        if result.success and np.all(np.isfinite(result.x)):
            tau_safe = np.asarray(result.x, dtype=np.float64)
            status = "solved_fallback_scipy"
        else:
            # 求解失败: 标记为不可行, 让外层 except 处理
            raise RuntimeError(f"scipy_slsqp_failed: {result.message}")
        return tau_safe, float(slack_for(tau_safe)), status, int(getattr(result, "nit", 0) or 0), "scipy_fallback"

    def _build_osqp_matrices(
        self,
        tau_des: NDArray[np.float64],
        a_hocbf: NDArray[np.float64],
        b_hocbf: float,
        feasible: FeasibleSetConstraints,
    ):
        """Build fixed-pattern OSQP matrices for z=[tau_x,tau_y,tau_n,delta].

        The constraint matrix is stored with all entries present through a tiny
        epsilon template. Later updates may set entries to exact zero while
        preserving the sparse pattern required for OSQP factorization reuse.
        """
        import scipy.sparse as sparse

        tau_des = np.asarray(tau_des, dtype=np.float64).reshape(3,)
        q = np.array([
            -self.tau_weight[0] * tau_des[0],
            -self.tau_weight[1] * tau_des[1],
            -self.tau_weight[2] * tau_des[2],
            0.0,
        ], dtype=np.float64)
        P = sparse.diags(
            [self.tau_weight[0], self.tau_weight[1], self.tau_weight[2], self.slack_weight],
            format="csc",
        )
        rows = []
        l_rows = []
        u_rows = []
        rows.append([float(a_hocbf[0]), float(a_hocbf[1]), float(a_hocbf[2]), 1.0])
        l_rows.append(float(b_hocbf))
        u_rows.append(np.inf)
        rows.append([0.0, 0.0, 0.0, 1.0])
        l_rows.append(0.0)
        u_rows.append(np.inf)
        for row, lo, hi in zip(feasible.A, feasible.l, feasible.u):
            rows.append([float(row[0]), float(row[1]), float(row[2]), 0.0])
            l_rows.append(float(lo) if np.isfinite(lo) else -np.inf)
            u_rows.append(float(hi) if np.isfinite(hi) else np.inf)
        dense = np.asarray(rows, dtype=np.float64)
        l = np.asarray(l_rows, dtype=np.float64)
        u = np.asarray(u_rows, dtype=np.float64)
        # Full dense sparsity template: every entry is stored once at setup.
        template = dense.copy()
        template[np.abs(template) < 1e-300] = 1e-12
        A_template = sparse.csc_matrix(template)
        return P, q, A_template, dense, l, u

    def _solve_qp_osqp(
        self,
        tau_des: NDArray[np.float64],
        a_hocbf: NDArray[np.float64],
        b_hocbf: float,
        feasible: FeasibleSetConstraints,
    ) -> tuple[NDArray[np.float64], float, str, int, str]:
        """Solve the QP with a cached OSQP direct-API solver when available."""
        if not self._osqp_available:
            raise RuntimeError("osqp_unavailable")
        import osqp

        P, q, A_template, dense_A, l, u = self._build_osqp_matrices(tau_des, a_hocbf, b_hocbf, feasible)
        key = (A_template.shape, P.shape, tuple(np.round(np.asarray(P.diagonal()), 12)))
        reused = self._osqp_solver is not None and self._osqp_shape_key == key
        if not reused:
            solver = osqp.OSQP()
            solver.setup(
                P=P, q=q, A=A_template, l=l, u=u,
                verbose=False, warm_starting=True, polishing=False,
                max_iter=2000, eps_abs=1e-5, eps_rel=1e-5,
                time_limit=0.5,  # 每次 QP 求解最多 0.5 秒
            )
            self._osqp_solver = solver
            self._osqp_shape_key = key
            self._osqp_setup_count += 1
        else:
            # A_template has full stored pattern. dense_A.T.ravel(order="C")
            # matches scipy CSC data order for a dense full-pattern matrix.
            Ax = np.asarray(dense_A.T, dtype=np.float64).ravel(order="C")
            self._osqp_solver.update(q=q, l=l, u=u, Ax=Ax)
            self._osqp_update_count += 1
        result = self._osqp_solver.solve()
        if result.info.status_val not in (1, 2):
            raise RuntimeError(str(result.info.status))
        z = np.asarray(result.x, dtype=np.float64)
        status = "solved_osqp_cached" if reused else "solved_osqp_setup"
        return z[:3], float(max(0.0, z[3])), status, int(result.info.iter), "osqp"

    def filter(
        self,
        state: NDArray[np.float64],
        tau_des: NDArray[np.float64],
        target_pos: tuple,
        target_psi: float,
        ice_state: Optional[Dict[str, float]] = None,
        risk_level: float = 0.0,
        dt: float = 0.1,
    ) -> SafetyFilterResult:
        """Apply safety filter to desired control input."""
        tau_des = np.asarray(tau_des, dtype=np.float64).reshape(3,)
        risk_scale = 1.0 + self.risk_gain * float(np.clip(risk_level, 0.0, 1.0))
        alpha1 = self.hocbf_params.alpha1_base * risk_scale
        alpha2 = self.hocbf_params.alpha2_base * risk_scale
        hocbf = compute_hocbf_constraint(
            state, tau_des, target_pos, target_psi, self.hocbf_params, alpha1, alpha2, dt
        )
        a_hocbf = np.asarray(hocbf["a_hocbf"], dtype=np.float64)
        b_hocbf = float(hocbf["b_hocbf"])
        h_val = float(hocbf["h_val"])
        h_dot = float(hocbf.get("h_dot", 0.0))
        b_hocbf_nominal = float(hocbf.get("b_hocbf_nominal", b_hocbf))
        robust_margin = float(hocbf.get("robust_disturbance_margin", 0.0))
        disturbance_accel_bound = float(hocbf.get("disturbance_accel_bound_mps2", 0.0))
        a_hocbf_norm = float(np.linalg.norm(a_hocbf))
        feasible = self._feasible_constraints()

        import time
        tic = time.perf_counter()
        if self.require_osqp and not self._osqp_available:
            elapsed_ms = (time.perf_counter() - tic) * 1000.0
            return SafetyFilterResult(
                tau_des=tau_des.copy(), tau_safe=tau_des.copy(), active=False,
                qp_success=False, infeasible=True, status="osqp_required_unavailable",
                solver_backend="osqp_unavailable", solve_time_ms=elapsed_ms,
                safety_set_h=h_val, safety_set_h_dot=h_dot, hocbf_margin=float("nan"),
                hocbf_nominal_margin=float("nan"), hocbf_robust_disturbance_margin=robust_margin,
                hocbf_disturbance_accel_bound_mps2=disturbance_accel_bound, hocbf_a_norm=a_hocbf_norm,
                hocbf_b=b_hocbf, hocbf_b_nominal=b_hocbf_nominal, hocbf_soft_certificate=False,
                risk_level=risk_level, risk_scale=risk_scale,
                alpha1=alpha1, alpha2=alpha2, constraint_mode=self.constraint_mode,
                feasible_set_type=feasible.feasible_set_type, actuator_mode=feasible.mode,
            )
        try:
            if self._osqp_available:
                try:
                    tau_safe, slack, status, iters, backend = self._solve_qp_osqp(tau_des, a_hocbf, b_hocbf, feasible)
                except RuntimeError:
                    # OSQP failed — fall back to SciPy before giving up
                    tau_safe, slack, status, iters, backend = self._solve_qp_scipy(tau_des, a_hocbf, b_hocbf, feasible)
            else:
                tau_safe, slack, status, iters, backend = self._solve_qp_scipy(tau_des, a_hocbf, b_hocbf, feasible)
            elapsed_ms = (time.perf_counter() - tic) * 1000.0
            correction_norm = float(np.linalg.norm(tau_safe - tau_des))
            tau_safe_arr = np.asarray(tau_safe, dtype=np.float64).reshape(3,)
            hocbf_margin = float(a_hocbf @ tau_safe_arr + float(slack) - b_hocbf)
            nominal_margin = float(a_hocbf @ tau_safe_arr + float(slack) - b_hocbf_nominal)
            soft_certificate = bool(
                hocbf_margin >= float(getattr(self.hocbf_params, "min_certificate_margin", -1e-7))
                and float(slack) <= float(getattr(self.hocbf_params, "max_certificate_slack", 1e-7))
                and not np.isnan(hocbf_margin)
            )
            return SafetyFilterResult(
                tau_des=tau_des.copy(), tau_safe=tau_safe_arr,
                active=correction_norm > 1e-9, qp_success=True, infeasible=False,
                status=status, solver_backend=backend, solve_time_ms=elapsed_ms, iterations=iters,
                slack=float(slack), safety_set_h=h_val, safety_set_h_dot=h_dot, hocbf_margin=hocbf_margin,
                hocbf_nominal_margin=nominal_margin, hocbf_robust_disturbance_margin=robust_margin,
                hocbf_disturbance_accel_bound_mps2=disturbance_accel_bound, hocbf_a_norm=a_hocbf_norm,
                hocbf_b=b_hocbf, hocbf_b_nominal=b_hocbf_nominal, hocbf_soft_certificate=soft_certificate,
                correction_norm=correction_norm,
                risk_level=float(risk_level), risk_scale=risk_scale, alpha1=alpha1, alpha2=alpha2,
                constraint_mode=self.constraint_mode, feasible_set_type=feasible.feasible_set_type,
                actuator_mode=feasible.mode,
                solver_setup_count=self._osqp_setup_count, solver_update_count=self._osqp_update_count,
                osqp_reused_factorization=(backend == "osqp" and status == "solved_osqp_cached"),
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - tic) * 1000.0
            # C1 fix: on QP infeasibility, return tau_des clamped to box constraints
            # to maintain stationkeeping ability instead of zero torque.
            tau_limit = np.array([self.max_force_x, self.max_force_y, self.max_moment_n], dtype=np.float64)
            tau_fallback = np.clip(tau_des, -tau_limit, tau_limit).astype(np.float64)
            correction_norm = float(np.linalg.norm(tau_fallback - tau_des))
            return SafetyFilterResult(
                tau_des=tau_des.copy(), tau_safe=tau_fallback, active=True,
                qp_success=False, infeasible=True, status=f"error:{e}", solver_backend="error",
                solve_time_ms=elapsed_ms, iterations=0, slack=0.0, safety_set_h=h_val, safety_set_h_dot=h_dot, hocbf_margin=float("nan"),
                hocbf_nominal_margin=float("nan"), hocbf_robust_disturbance_margin=robust_margin,
                hocbf_disturbance_accel_bound_mps2=disturbance_accel_bound, hocbf_a_norm=a_hocbf_norm,
                hocbf_b=b_hocbf, hocbf_b_nominal=b_hocbf_nominal, hocbf_soft_certificate=False,
                correction_norm=correction_norm, risk_level=float(risk_level), risk_scale=risk_scale,
                alpha1=alpha1, alpha2=alpha2, constraint_mode=self.constraint_mode,
                feasible_set_type=feasible.feasible_set_type, actuator_mode=feasible.mode,
            )


class DisabledSafetyFilter:
    """Pass-through filter — returns tau_des unchanged."""

    def filter(
        self,
        state: NDArray[np.float64],
        tau_des: NDArray[np.float64],
        target_pos: tuple,
        target_psi: float,
        ice_state: Optional[Dict[str, float]] = None,
        risk_level: float = 0.0,
        dt: float = 0.1,
    ) -> SafetyFilterResult:
        tau_des = np.asarray(tau_des, dtype=np.float64).reshape(3,)
        return SafetyFilterResult(
            tau_des=tau_des.copy(), tau_safe=tau_des.copy(), active=False,
            qp_success=True, infeasible=False, status="disabled", solver_backend="disabled",
            solve_time_ms=0.0, iterations=0, slack=0.0, safety_set_h=0.0, safety_set_h_dot=0.0, hocbf_margin=0.0,
            correction_norm=0.0, risk_level=0.0, risk_scale=1.0, alpha1=0.0, alpha2=0.0,
            constraint_mode="none", feasible_set_type="none", actuator_mode="disabled",
        )
