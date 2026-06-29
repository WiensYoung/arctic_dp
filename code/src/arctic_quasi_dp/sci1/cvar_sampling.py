"""Profile-based CVaR (Conditional Value-at-Risk) sampling.

Implements:
1. Profile-based sample counts (smoke=32, paper=256+)
2. Adaptive early stopping based on relative standard error
3. Vectorized sampling for performance
4. Reproducible sampling with fixed seeds

Reference:
- Rockafellar & Uryasev (2000) "Optimization of Conditional Value-at-Risk"
- Acerbi & Tasche (2002) "On the coherence of expected shortfall"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray


# Profile-based sample counts
CVAR_PROFILE_SAMPLES = {
    "smoke": 32,
    "method_smoke": 64,
    "method_fast": 128,
    "paper_small": 256,
    "paper_full": 512,
    "submission": 512,
}


@dataclass
class CVaRResult:
    """Result of CVaR computation with diagnostics."""
    cvar: float                     # CVaR estimate
    var: float                      # VaR (quantile) estimate
    mean: float                     # Mean of all samples
    std: float                      # Standard deviation of all samples
    n_samples: int                  # Total number of samples
    n_tail: int                     # Number of tail samples
    cvar_standard_error: float      # Standard error of CVaR estimate
    cvar_relative_se: float         # Relative standard error (SE/mean)
    cvar_effective_tail_samples: int  # Effective tail samples
    cvar_converged: bool            # True if relative SE < threshold
    alpha: float                    # CVaR confidence level


def compute_cvar_vectorized(
    samples: NDArray[np.float64],
    alpha: float = 0.90,
) -> CVaRResult:
    """Compute CVaR from pre-computed samples.

    CVaR_alpha = E[X | X >= VaR_alpha]
    where VaR_alpha = quantile(X, alpha)

    Args:
        samples: Array of loss/risk samples (n,)
        alpha: Confidence level (default 0.90)

    Returns:
        CVaRResult with estimate and diagnostics
    """
    samples = np.asarray(samples, dtype=np.float64).ravel()
    n = len(samples)

    if n == 0:
        return CVaRResult(
            cvar=0.0, var=0.0, mean=0.0, std=0.0,
            n_samples=0, n_tail=0, cvar_standard_error=0.0,
            cvar_relative_se=0.0, cvar_effective_tail_samples=0,
            cvar_converged=True, alpha=alpha,
        )

    # Sort samples for quantile computation
    sorted_samples = np.sort(samples)

    # VaR: alpha-quantile
    var_idx = int(np.ceil(alpha * n)) - 1
    var_idx = min(var_idx, n - 1)
    var = float(sorted_samples[var_idx])

    # CVaR: mean of samples >= VaR
    tail_mask = samples >= var
    tail = samples[tail_mask]
    n_tail = len(tail)

    if n_tail == 0:
        # Edge case: all samples are identical
        return CVaRResult(
            cvar=float(np.mean(samples)),
            var=float(var),
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            n_samples=n, n_tail=0,
            cvar_standard_error=0.0,
            cvar_relative_se=0.0,
            cvar_effective_tail_samples=0,
            cvar_converged=True, alpha=alpha,
        )

    cvar = float(np.mean(tail))

    # Standard error of CVaR estimate
    # SE(CVaR) ≈ std(tail) / sqrt(n_tail)
    tail_std = float(np.std(tail, ddof=1)) if n_tail > 1 else 0.0
    cvar_se = tail_std / np.sqrt(n_tail) if n_tail > 0 else 0.0

    # Relative standard error
    cvar_rel_se = cvar_se / max(abs(cvar), 1e-12)

    return CVaRResult(
        cvar=cvar,
        var=float(var),
        mean=float(np.mean(samples)),
        std=float(np.std(samples)),
        n_samples=n,
        n_tail=n_tail,
        cvar_standard_error=float(cvar_se),
        cvar_relative_se=float(cvar_rel_se),
        cvar_effective_tail_samples=n_tail,
        cvar_converged=cvar_rel_se < 0.05,  # 5% relative SE threshold
        alpha=alpha,
    )


def compute_cvar_adaptive(
    loss_fn,
    rng: np.random.Generator,
    alpha: float = 0.90,
    min_samples: int = 64,
    max_samples: int = 512,
    relative_se_threshold: float = 0.05,
    batch_size: int = 64,
) -> CVaRResult:
    """Compute CVaR with adaptive sample count.

    Starts with min_samples and increases in batches until:
    - Relative SE < threshold, OR
    - max_samples reached

    Args:
        loss_fn: Function that takes rng and returns array of loss samples
        rng: Random number generator
        alpha: CVaR confidence level
        min_samples: Minimum number of samples
        max_samples: Maximum number of samples
        relative_se_threshold: Convergence threshold
        batch_size: Samples per batch

    Returns:
        CVaRResult with estimate and diagnostics
    """
    all_samples = []

    for n_batch in range(0, max_samples, batch_size):
        # Generate batch
        batch = loss_fn(rng, batch_size)
        all_samples.append(np.asarray(batch, dtype=np.float64).ravel())

        # Check convergence if we have enough samples
        combined = np.concatenate(all_samples)
        if len(combined) >= min_samples:
            result = compute_cvar_vectorized(combined, alpha)
            if result.cvar_converged:
                return result

    # Max samples reached
    combined = np.concatenate(all_samples)
    return compute_cvar_vectorized(combined, alpha)


def get_cvar_samples_for_profile(profile: str) -> int:
    """Get recommended CVaR sample count for experiment profile.

    Args:
        profile: Experiment profile name

    Returns:
        Recommended number of CVaR samples
    """
    return CVAR_PROFILE_SAMPLES.get(profile, 128)
