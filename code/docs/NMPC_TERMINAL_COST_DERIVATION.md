# NMPC Terminal Cost Derivation

## 1. Problem Formulation

The NMPC solves at each timestep $k$:

$$\min_{u_0, \ldots, u_{N-1}} \sum_{i=0}^{N-1} \ell(x_i, u_i) + V_f(x_N)$$

subject to:
- $x_{i+1} = f(x_i, u_i)$ (dynamics)
- $u_i \in \mathcal{U}$ (input constraints)
- $x_N \in \mathcal{X}_f$ (terminal set)

where $V_f(x_N)$ is the terminal cost and $\mathcal{X}_f$ is the terminal set.

## 2. Linearization at Equilibrium

The 3-DOF DP dynamics are linearized at the equilibrium point:
- $x_{\text{ref}} = [x_0, y_0, \psi_0, 0, 0, 0]^T$ (zero velocity)
- $u_{\text{ref}} = [0, 0, 0]^T$ (zero control)

The continuous-time model is:

$$\dot{x} = A_c x + B_c u$$

where:

$$A_c = \begin{bmatrix} 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & -X_u/m & 0 & 0 \\ 0 & 0 & 0 & 0 & -Y_v/m & 0 \\ 0 & 0 & 0 & 0 & 0 & -N_r/I_z \end{bmatrix}$$

$$B_c = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \\ 1/m & 0 & 0 \\ 0 & 1/m & 0 \\ 0 & 0 & 1/I_z \end{bmatrix}$$

Discretized using forward Euler ($\Delta t = 0.1$s):

$$A_d = I + A_c \Delta t, \quad B_d = B_c \Delta t$$

## 3. Discrete-time Algebraic Riccati Equation (DARE)

The terminal cost matrix $P$ is obtained by solving the DARE:

$$P = A_d^T P A_d - A_d^T P B_d (R + B_d^T P B_d)^{-1} B_d^T P A_d + Q$$

where:
- $Q = \text{diag}(Q_{\text{pos}}, Q_{\text{pos}}, Q_{\text{heading}}, Q_{\text{vel}}, Q_{\text{vel}}, Q_{\text{vel}})$
- $R = \text{diag}(R_{\text{force}}, R_{\text{force}}, R_{\text{moment}})$

The DARE is solved iteratively until convergence:

$$P_{k+1} = A_d^T P_k A_d - A_d^T P_k B_d (R + B_d^T P_k B_d)^{-1} B_d^T P_k A_d + Q$$

## 4. Terminal Feedback Gain

The terminal feedback gain is:

$$K = (R + B_d^T P B_d)^{-1} B_d^T P A_d$$

The terminal control law is $u = -Kx$, which locally stabilizes the system.

## 5. Closed-loop Stability

The closed-loop dynamics are:

$$A_{\text{cl}} = A_d - B_d K$$

The eigenvalues of $A_{\text{cl}}$ must satisfy $|\lambda_i| < 1$ for all $i$.

The spectral radius $\rho(A_{\text{cl}}) = \max_i |\lambda_i|$ is reported as a diagnostic.

## 6. Terminal Set Estimation

The terminal set is defined as:

$$\mathcal{X}_f = \{ x : x^T P x \leq \alpha \}$$

where $\alpha$ is chosen such that for all $x \in \mathcal{X}_f$:
1. The terminal control $u = -Kx$ satisfies input constraints $u \in \mathcal{U}$
2. The closed-loop trajectory remains in $\mathcal{X}_f$

A conservative estimate is:

$$\alpha = \min\left(\frac{F_{\max}}{\|K\|_{\infty}}, \frac{M_{\max}}{\|K\|_{\infty}}\right)$$

## 7. Limitations

This terminal cost is derived from the **linearized** dynamics. Under ice disturbances:
- The actual dynamics are nonlinear
- The linearization is valid only near the equilibrium
- The terminal cost provides **local** stability guarantees
- For large deviations, the NMPC must rely on the stage cost and constraints

The terminal set $\mathcal{X}_f$ is conservative — states outside this set may still be stabilizable by the NMPC, but without formal guarantees.

## 8. Implementation

The implementation is in `src/arctic_quasi_dp/sci1/control/nmpc_terminal.py`.

Key functions:
- `linearize_discrete_dynamics()`: Returns $(A_d, B_d)$
- `solve_dare()`: Solves DARE for $P$
- `compute_terminal_cost()`: Returns $P$, $K$, stability analysis
- `terminal_value()`: Computes $V_f(x) = x^T P x$
- `is_in_terminal_set()`: Checks if $x \in \mathcal{X}_f$

## 9. References

1. Mayne, D. Q., Rawlings, J. B., Rao, C. V., & Scokaert, P. O. (2000). Constrained model predictive control: Stability and optimality. *Automatica*, 36(6), 789-814.
2. Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing.
3. Fossen, T. I. (2011). *Handbook of Marine Craft Hydrodynamics and Motion Control*. Wiley.
