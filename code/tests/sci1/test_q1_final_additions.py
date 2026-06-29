from pathlib import Path

import numpy as np

from arctic_quasi_dp.sci1.runner import build_controller
from arctic_quasi_dp.sci1.control.actuator_feasible_set import ActuatorAwareFeasibleSet
from arctic_quasi_dp.sci1.data_bridge import DataDrivenIceSchedule


def test_new_proxy_baselines_are_runnable():
    state = np.zeros(6, dtype=float)
    for name in ["adrc", "robust_mpc", "tube_mpc"]:
        c = build_controller(name)
        c.set_target(1.0, -1.0, 5.0)
        c.set_ice_conditions(0.5, 0.8, 0.2, 180.0)
        r = c.compute_control(state, dt=0.2)
        assert r.tau.shape == (3,)
        assert np.all(np.isfinite(r.tau))
        assert r.mode in {name, "tube_mpc"}


def test_actuator_polygon_is_allocator_checked_and_fixed_shape():
    fs = ActuatorAwareFeasibleSet(n_vertices=16, max_facets=16, use_allocator_feasibility_check=True)
    nominal = fs.get_constraints(mode="nominal")
    degraded = fs.get_constraints(mode="bow_degraded_0.5")
    locked = fs.get_constraints(mode="azimuth_locked")
    assert nominal.A.shape == degraded.A.shape == locked.A.shape
    assert nominal.feasible_set_type == "inner_polygon_proxy"
    assert np.nanmean(degraded.u[np.isfinite(degraded.u)]) < np.nanmean(nominal.u[np.isfinite(nominal.u)])
    assert locked.active_rows == nominal.active_rows


def test_packaged_h1_mock_replay_loads():
    nc = Path("data/sci1_sources/copernicus/arctic_ice_2020_jan1_7.nc")
    assert nc.exists()
    sched = DataDrivenIceSchedule(nc, lat=80.0, lon=0.0, duration=20.0)
    s0 = sched.at(0.0)
    s1 = sched.at(10.0)
    assert 0.0 <= s0.concentration <= 1.0
    assert s0.thickness > 0.0
    assert s1.drift_speed >= 0.0
    assert 0.0 <= s1.drift_direction <= 360.0
