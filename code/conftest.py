"""Pytest configuration — adds src/ to sys.path for package discovery."""
import sys
from pathlib import Path

import numpy as np
import pytest

_src = Path(__file__).parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


# ============================================================
# Shared test fixtures
# ============================================================

@pytest.fixture
def default_vessel_params():
    """Default 500t proxy vessel parameters."""
    from arctic_quasi_dp.sci1.sim_loop import VesselParams
    return VesselParams()


@pytest.fixture
def default_ice_conditions():
    """Default moderate ice conditions."""
    return {
        "concentration": 0.5,
        "thickness": 0.8,
        "drift_speed": 0.3,
        "drift_direction": 140.0,
    }


@pytest.fixture
def zero_state():
    """Zero 6-DOF state vector at origin."""
    return np.zeros(6, dtype=np.float64)


@pytest.fixture
def make_state():
    """Factory fixture for creating state vectors with custom values."""
    def _make(x=0.0, y=0.0, psi=0.0, u=0.0, v=0.0, r=0.0):
        return np.array([x, y, psi, u, v, r], dtype=np.float64)
    return _make


@pytest.fixture
def default_thruster_config():
    """Default thruster configuration."""
    from arctic_quasi_dp.sci1.thruster import ThrusterConfig
    return ThrusterConfig.generic_dp_vessel()


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Temporary directory for test output files."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir
