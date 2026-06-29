"""SI unit constants and conversion utilities.

All internal computations use SI units:
  force: N, moment: N*m, mass: kg, length: m, time: s, ice_strength: Pa, power: W
"""

from __future__ import annotations

MPA_TO_PA = 1e6
KN_TO_N = 1e3
MN_TO_N = 1e6

# Default actuator limits for simplified 500t model (NOT XueLong2)
SIMPLIFIED_500T_MAX_FORCE_N = 1500.0
SIMPLIFIED_500T_MAX_MOMENT_NM = 20000.0

# Engineering-estimated actuator limits for XueLong2-like scale
# ~14000t displacement, multiple thrusters ~200-500 kN each
XUELONG2_LIKE_MAX_FORCE_N = 1.0e6    # ~1 MN total
XUELONG2_LIKE_MAX_MOMENT_NM = 8.0e7  # ~80 MN*m


def mpa_to_pa(x: float) -> float:
    """Convert MPa to Pa."""
    return x * MPA_TO_PA


def kn_to_n(x: float) -> float:
    """Convert kN to N."""
    return x * KN_TO_N


def mn_to_n(x: float) -> float:
    """Convert MN to N."""
    return x * MN_TO_N


UNITS_MANIFEST = {
    "force": "N",
    "moment": "N_m",
    "mass": "kg",
    "length": "m",
    "time": "s",
    "ice_strength": "Pa",
    "power": "W",
    "angle": "rad",
    "notes": "All internal computations use SI units. MPa converted via * 1e6.",
}
