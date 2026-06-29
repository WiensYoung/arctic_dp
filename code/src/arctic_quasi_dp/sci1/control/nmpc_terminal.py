"""NMPC terminal cost and terminal set from LQR Riccati equation.

Implements:
1. Linearization of 3-DOF DP dynamics at equilibrium
2. Discrete-time Algebraic Riccati Equation (DARE) solution
3. Terminal cost V_f(x) = (x - x_ref)^T P (x - x_ref)
4. Terminal feedback gain K for local stabilization
5. Terminal ellipsoid set estimation

This provides the theoretical foundation for:
- Recursive feasibility of the NMPC
- Local stability of the closed-loop system
- Practical safety under ice disturbances

Reference:
- Mayne et al. (2000) "Constrained MPC: Stability and Optimality"
- Rawlings et al. (2017) "Model Predictive Control: Theory, Computation, and Design"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class TerminalCostResult:
    """Result of terminal cost computation."""
    P: NDArray[np.float64]          # Terminal cost matrix (6x6, SPD)
    K: NDArray[np.float64]          # Terminal feedback gain (3x6)
    A_cl: NDArray[np.float64]       # Closed-loop A matrix (A + BK)
    spectral_radius: float          # Spectral radius of A_cl
    is_stable: bool                 # True if spectral_radius < 1
    terminal_cost: float            # V_f(x) for given state
    alpha: float                    # Terminal set radius estimate
    method: str                     # "dare" or "pd_fallback"


def linearize_discrete_dynamics(
    mass: float,
    Izz: float,
    Xu: float,
    Yv: float,
    Nr: float,
    dt: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Linearize 3-DOF DP dynamics at equilibrium (zero velocity, zero heading).

    The continuous-time model is:
        x_dot = A_c x + B_c u
    where x = [x_ned, y_ned, psi, u_body, v_body, r_body]

    Discretized using forward Euler: A = I + A_c * dt, B = B_c * dt

    Args:
        mass: Vessel mass (kg)
        Izz: Yaw moment of inertia (kg*m^2)
        Xu, Yv, Nr: Linear damping coefficients
        dt: Time step (s)

    Returns:
        A_d: Discrete-time state matrix (6x6)
        B_d: Discrete-time input matrix (6x3)
    """
    # Continuous-time state matrix at equilibrium (psi=0, u=v=r=0)
    A_c = np.zeros((6, 6), dtype=np.float64)
    # Position kinematics: xdot = u, ydot = v, psidot = r
    A_c[0, 3] = 1.0
    A_c[1, 4] = 1.0
    A_c[2, 5] = 1.0
    # Velocity dynamics (linearized at zero velocity)
    A_c[3, 3] = -Xu / mass
    A_c[4, 4] = -Yv / mass
    A_c[5, 5] = -Nr / Izz

    # Continuous-time input matrix
    B_c = np.zeros((6, 3), dtype=np.float64)
    B_c[3, 0] = 1.0 / mass    # Fx -> u_dot
    B_c[4, 1] = 1.0 / mass    # Fy -> v_dot
    B_c[5, 2] = 1.0 / Izz     # Mz -> r_dot

    # Forward Euler discretization
    A_d = np.eye(6) + A_c * dt
    B_d = B_c * dt

    return A_d, B_d


def solve_dare(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
    max_iter: int = 200,
    tol: float = 1e-10,
) -> NDArray[np.float64]:
    """Solve Discrete-time Algebraic Riccati Equation (DARE).

    P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q

    Uses iterative fixed-point method:
    P_{k+1} = A^T P_k A - A^T P_k B (R + B^T P_k B)^{-1} B^T P_k A + Q

    Args:
        A: State matrix (n x n)
        B: Input matrix (n x m)
        Q: State cost weight (n x n, PSD)
        R: Input cost weight (m x m, PD)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        P: Solution to DARE (n x n, SPD)
    """
    import warnings
    n = A.shape[0]
    P = Q.copy()  # Initialize with Q

    for i in range(max_iter):
        BtPB = B.T @ P @ B
        K = np.linalg.solve(R + BtPB, B.T @ P @ A)
        P_new = A.T @ P @ A - A.T @ P @ B @ K + Q

        # Check convergence
        delta = np.max(np.abs(P_new - P))
        P = P_new
        if delta < tol:
            return P, True  # converged

    # 未收敛: 发出警告, 返回当前近似解并标记 unconverged
    warnings.warn(
        f"DARE solver did not converge after {max_iter} iterations "
        f"(final delta={delta:.2e}, tol={tol:.2e}). Terminal cost matrix may be inaccurate.",
        stacklevel=2,
    )
    return P, False  # (P_matrix, converged_flag)


def compute_terminal_cost(
    mass: float = 500000.0,
    Izz: float = 5e8,
    Xu: float = 500.0,
    Yv: float = 800.0,
    Nr: float = 2e5,
    dt: float = 0.1,
    Q_pos: float = 100.0,
    Q_heading: float = 50.0,
    Q_vel: float = 1.0,
    R_force: float = 0.001,
    R_moment: float = 0.0001,
) -> TerminalCostResult:
    """Compute terminal cost matrix P and feedback gain K.

    Solves the DARE for the linearized 3-DOF DP dynamics to obtain:
    - P: terminal cost matrix V_f(x) = x^T P x
    - K: terminal feedback gain u = -K x
    - A_cl = A + B K: closed-loop dynamics (should be stable)

    Args:
        mass, Izz: Vessel parameters
        Xu, Yv, Nr: Damping coefficients
        dt: Time step
        Q_pos, Q_heading, Q_vel: State cost weights
        R_force, R_moment: Input cost weights

    Returns:
        TerminalCostResult with P, K, stability analysis
    """
    # 1. Linearize dynamics
    A, B = linearize_discrete_dynamics(mass, Izz, Xu, Yv, Nr, dt)

    # 2. Weight matrices
    Q = np.diag([Q_pos, Q_pos, Q_heading, Q_vel, Q_vel, Q_vel])
    R = np.diag([R_force, R_force, R_moment])

    # 3. Solve DARE
    converged = False
    try:
        P, converged = solve_dare(A, B, Q, R)
        method = "dare" if converged else "dare_unconverged"
    except np.linalg.LinAlgError:
        # Fallback: use Q as terminal cost (conservative)
        P = Q.copy() * 10.0
        method = "pd_fallback"

    # 4. Compute terminal feedback gain (use P even if unconverged, with warning logged)
    BtPB = B.T @ P @ B
    K = np.linalg.solve(R + BtPB, B.T @ P @ A)

    # 5. Closed-loop dynamics
    A_cl = A - B @ K  # Note: K is positive, so u = -Kx means A_cl = A - BK

    # 6. Stability check
    eigenvalues = np.linalg.eigvals(A_cl)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    is_stable = spectral_radius < 1.0

    # 7. Terminal set estimate (conservative ellipsoid)
    # alpha = min over constraints of the maximum x^T P x such that constraints are satisfied
    # For simplicity, use a heuristic based on force limits
    max_force = 3000.0
    max_moment = 100000.0
    # The terminal set should be small enough that the linear feedback K
    # keeps the system within actuator limits
    K_norm = np.max(np.abs(K))
    if K_norm > 1e-12:
        alpha = min(max_force, max_moment) / K_norm
    else:
        alpha = 1e6

    return TerminalCostResult(
        P=P,
        K=K,
        A_cl=A_cl,
        spectral_radius=spectral_radius,
        is_stable=is_stable,
        terminal_cost=0.0,  # Will be computed for specific state
        alpha=alpha,
        method=method,
    )


def terminal_value(
    state_error: NDArray[np.float64],
    P: NDArray[np.float64],
) -> float:
    """Compute terminal cost V_f(x) = (x - x_ref)^T P (x - x_ref).

    Args:
        state_error: State error vector (6,)
        P: Terminal cost matrix (6x6)

    Returns:
        Terminal cost value (scalar)
    """
    state_error = np.asarray(state_error, dtype=np.float64).reshape(6,)
    return float(state_error @ P @ state_error)


def is_in_terminal_set(
    state_error: NDArray[np.float64],
    P: NDArray[np.float64],
    alpha: float,
) -> bool:
    """Check if state is in terminal set X_f = {x : x^T P x <= alpha}.

    Args:
        state_error: State error vector (6,)
        P: Terminal cost matrix (6x6)
        alpha: Terminal set radius

    Returns:
        True if state is in terminal set
    """
    return terminal_value(state_error, P) <= alpha
