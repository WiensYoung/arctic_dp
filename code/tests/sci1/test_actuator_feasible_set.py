"""Actuator-aware feasible-set tests."""

import numpy as np

from arctic_quasi_dp.sci1.control.actuator_feasible_set import ActuatorAwareFeasibleSet, BoxFeasibleSet


def test_box_feasible_set_bounds():
    fs = BoxFeasibleSet(max_force_x=10, max_force_y=20, max_moment_n=30)
    lo, hi = fs.get_bounds()
    assert np.allclose(lo, [-10, -20, -30])
    assert np.allclose(hi, [10, 20, 30])


def test_polygon_constraints_shape_fixed():
    fs = ActuatorAwareFeasibleSet(n_vertices=16, max_facets=16)
    c1 = fs.get_constraints(mode="nominal")
    c2 = fs.get_constraints(mode="bow_degraded_0.5")
    assert c1.A.shape == c2.A.shape == (18, 3)
    assert c1.l.shape == c1.u.shape == c2.l.shape == c2.u.shape == (18,)


def test_degraded_polygon_is_more_conservative_than_nominal():
    fs = ActuatorAwareFeasibleSet(n_vertices=16, max_facets=16)
    nominal = fs.get_constraints(mode="nominal")
    degraded = fs.get_constraints(mode="bow_degraded_0.5")
    finite = np.isfinite(nominal.u) & np.isfinite(degraded.u)
    assert np.nanmean(degraded.u[finite]) < np.nanmean(nominal.u[finite])


def test_power_limited_scalar_shrink():
    fs = ActuatorAwareFeasibleSet(n_vertices=16, max_facets=16)
    full = fs.get_constraints(mode="power_limited", power_scale_factor=1.0)
    limited = fs.get_constraints(mode="power_limited", power_scale_factor=0.25)
    finite = np.isfinite(full.u) & np.isfinite(limited.u)
    assert np.all(limited.u[finite] <= full.u[finite] + 1e-9)
    assert np.any(limited.u[finite] < full.u[finite] - 1e-9)


def test_azimuth_locked_has_distinct_cached_shape():
    fs = ActuatorAwareFeasibleSet(n_vertices=16, max_facets=16)
    nominal = fs.get_constraints(mode="nominal")
    locked = fs.get_constraints(mode="azimuth_locked")
    finite = np.isfinite(nominal.u) & np.isfinite(locked.u)
    assert not np.allclose(nominal.u[finite], locked.u[finite])
    assert locked.feasible_set_type == "inner_polygon_proxy"
