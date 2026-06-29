"""Test ice force outputs correct SI units and reasonable magnitude."""

import numpy as np
import pytest

from arctic_quasi_dp.sci1.ice_models import EmpiricalIceLoadModel, StochasticIceLoadModel
from arctic_quasi_dp.sci1.sim_loop import _ice_force_body, VesselParams


class TestIceForceMagnitude:

    def test_empirical_outputs_si_units(self):
        model = EmpiricalIceLoadModel()
        result = model.compute(
            psi=0.0, concentration=0.5, thickness=0.8,
            drift_speed=0.3, drift_direction=0.0,
            vessel_length=122.5, vessel_beam=22.0,
        )
        assert result.force_body.shape == (3,)
        assert np.all(np.isfinite(result.force_body))

    def test_empirical_force_reasonable_magnitude(self):
        """For moderate ice (c=0.5, h=0.8m, v=0.3m/s), proxy model force should be ~100N-100kN."""
        model = EmpiricalIceLoadModel()
        result = model.compute(
            psi=0.0, concentration=0.5, thickness=0.8,
            drift_speed=0.3, drift_direction=0.0,
            vessel_length=122.5, vessel_beam=22.0,
        )
        force_mag = np.linalg.norm(result.force_body[:2])
        assert 1e2 <= force_mag <= 1e5, f"Force magnitude {force_mag:.2e} N out of range"

    def test_sim_loop_ice_force_si(self):
        params = VesselParams()
        ice = {"concentration": 0.5, "thickness": 0.8, "drift_speed": 0.3, "drift_direction": 0.0}
        force = _ice_force_body(ice, 0.0, params)
        assert force.shape == (3,)
        assert np.all(np.isfinite(force))
        force_mag = np.linalg.norm(force[:2])
        assert 1e2 <= force_mag <= 1e5, f"Force magnitude {force_mag:.2e} N out of range"

    def test_ice_force_increases_with_thickness(self):
        model = EmpiricalIceLoadModel()
        r1 = model.compute(0.0, 0.5, 0.3, 0.2, 0.0, 122.5, 22.0)
        r2 = model.compute(0.0, 0.5, 1.5, 0.2, 0.0, 122.5, 22.0)
        assert r2.base_force_n > r1.base_force_n

    def test_ice_force_zero_when_no_ice(self):
        model = EmpiricalIceLoadModel()
        result = model.compute(0.0, 0.0, 0.0, 0.0, 0.0, 122.5, 22.0)
        np.testing.assert_allclose(result.force_body, 0.0, atol=1.0)

    def test_stochastic_reasonable_magnitude(self):
        model = StochasticIceLoadModel()
        result = model.compute(
            psi=0.0, concentration=0.5, thickness=0.8,
            drift_speed=0.3, drift_direction=0.0,
            vessel_length=122.5, vessel_beam=22.0,
            rng=np.random.default_rng(42),
        )
        force_mag = np.linalg.norm(result.force_body[:2])
        # Stochastic can burst, so wider range
        assert 1e1 <= force_mag <= 1e6, f"Stochastic force {force_mag:.2e} N out of range"
