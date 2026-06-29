"""QP solver wrapper for safety filter.

Primary: OSQP direct API (cached factorization, warm-started).
Optional: CVXPY as oracle for verification only.

Note: Do not use CVXPY in the real-time loop.
"""

from __future__ import annotations

import numpy as np


def check_osqp_available() -> bool:
    """Check if OSQP is available."""
    try:
        import osqp  # noqa: F401
        return True
    except ImportError:
        return False
