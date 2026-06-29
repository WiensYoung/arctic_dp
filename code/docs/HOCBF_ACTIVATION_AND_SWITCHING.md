# HOCBF Activation and Mode Switching

## 1. Problem Statement

The HOCBF safety filter must be activated/deactivated based on the vessel's proximity to the safety boundary. Naive threshold-based activation causes:

- **Chattering**: Rapid on/off switching when the state is near the threshold
- **Control discontinuities**: Abrupt changes in the control signal
- **Zeno behavior**: Infinite switches in finite time

## 2. Hysteresis Activation

### 2.1 Dual-Threshold Design

Instead of a single threshold, we use two thresholds:

**Activation condition** (enter safety filter mode):
$$h(x) \leq h_{\text{activate}} \quad \text{OR} \quad r_{\text{filtered}} \geq r_{\text{activate}}$$

**Deactivation condition** (exit safety filter mode):
$$h(x) \geq h_{\text{deactivate}} \quad \text{AND} \quad r_{\text{filtered}} \leq r_{\text{deactivate}}$$

where:
- $h(x) = R^2 - \|p - p_{\text{ref}}\|^2$ is the HOCBF safety function
- $r_{\text{filtered}}$ is the first-order filtered risk estimate
- $h_{\text{deactivate}} > h_{\text{activate}}$ (hysteresis gap)
- $r_{\text{deactivate}} < r_{\text{activate}}$ (hysteresis gap)

### 2.2 Why Hysteresis Prevents Chattering

With a single threshold $\theta$:
- If $h(x) \approx \theta$, small perturbations cause repeated switching
- The system oscillates between modes

With dual thresholds $[\theta_{\text{low}}, \theta_{\text{high}}]$:
- Once activated at $\theta_{\text{low}}$, the state must reach $\theta_{\text{high}}$ to deactivate
- This creates a "dead zone" that absorbs perturbations

## 3. Dwell-Time Enforcement

### 3.1 Minimum Dwell Time

Each mode transition requires a minimum time $\tau_{\text{dwell}}$ in the current mode before another transition is allowed.

**Justification**: Prevents Zeno behavior and ensures the system has time to respond to the current control law before switching.

**Implementation**: If a transition is requested before $\tau_{\text{dwell}}$ has elapsed, the transition is delayed.

### 3.2 Ordered Transitions

Mode transitions are restricted to be **ordered** (one step at a time):

$$\text{NORMAL} \leftrightarrow \text{CAUTION} \leftrightarrow \text{SAFETY\_FILTER\_ACTIVE} \leftrightarrow \text{EMERGENCY\_BACKUP}$$

This prevents abrupt jumps from NORMAL to EMERGENCY, which would cause large control transients.

## 4. Smooth Risk Scheduling

The raw risk estimate $r_{\text{raw}}$ is filtered using a first-order low-pass filter:

$$r_{\text{filtered}}[k+1] = (1 - \beta) \cdot r_{\text{filtered}}[k] + \beta \cdot r_{\text{raw}}[k]$$

where:

$$\beta = \frac{\Delta t}{\tau_{\text{risk}} + \Delta t}$$

This prevents sudden changes in risk from triggering abrupt mode switches.

## 5. Practical Safety Under Slack

The soft-HOCBF QP allows slack $\delta \geq 0$:

$$\min_{\tau, \delta} \frac{1}{2} \|\tau - \tau_{\text{des}}\|^2 + \frac{\lambda}{2} \delta^2$$
$$\text{s.t.} \quad a_{\text{hocbf}}^T \tau + \delta \geq b_{\text{hocbf}}$$

**Practical safety**: When $\delta > 0$, the HOCBF constraint is violated. The system is **practically safe** if:
- $\delta$ is small relative to the safety margin
- The violation is temporary
- The system returns to the safe set

**Certificate diagnostics**: The `hocbf_margin` and `slack` fields in `SafetyFilterResult` provide real-time monitoring of practical safety.

## 6. ISS-style Switched System Discussion

### 6.1 Switched System Model

The closed-loop system can be modeled as a switched system:

$$\dot{x} = f_{\sigma(t)}(x)$$

where $\sigma(t) \in \{\text{NORMAL}, \text{CAUTION}, \text{FILTER}, \text{EMERGENCY}\}$ is the mode.

### 6.2 Dwell-Time Stability

Under the dwell-time assumption ($\tau_{\text{dwell}} > 0$), if each mode is Input-to-State Stable (ISS), then the switched system is ISS.

**ISS Lyapunov function**: $V(x) = x^T P x$ where $P$ is the terminal cost matrix.

**ISS condition**: For each mode $i$:

$$\dot{V} \leq -\alpha_i V + \gamma_i \|d\|^2$$

where $d$ is the disturbance (ice force).

### 6.3 Limitations

- The ISS analysis assumes each mode is ISS, which requires verification for each controller
- The dwell-time assumption may be violated in extreme scenarios
- The practical safety guarantee is **soft** (allows temporary violations)

## 7. Diagnostic Metrics

The following metrics are recorded for paper reporting:

| Metric | Description |
|--------|-------------|
| `mode_switch_count` | Total number of mode transitions |
| `min_mode_dwell_time_s` | Minimum time spent in any mode |
| `dwell_time_violation_count` | Number of delayed transitions |
| `hocbf_active_rate` | Fraction of time HOCBF is active |
| `chattering_index` | Mode transitions per unit time |
| `risk_filtered_mean` | Mean filtered risk over simulation |

## 8. Configuration

```yaml
safety_filter:
  activation:
    mode: hysteresis
    h_activate: 4.0        # activate when h(x) <= this
    h_deactivate: 9.0      # deactivate when h(x) >= this
    risk_activate: 0.65    # activate when risk >= this
    risk_deactivate: 0.45  # deactivate when risk <= this
    min_dwell_time_s: 5.0  # minimum time in each mode
    risk_time_constant_s: 1.0  # risk filter time constant
```

## 9. References

1. Hespanha, J. P., & Morse, A. S. (1999). Stability of switched systems with average dwell-time. *Proc. IEEE CDC*, 2655-2660.
2. Ames, A. D., Grizzle, J. W., & Tabuada, P. (2019). Control barrier function based quadratic programs with application to adaptive cruise control. *Proc. IEEE CDC*, 6271-6278.
3. Liberzon, D. (2003). *Switching in Systems and Control*. Birkhäuser.
