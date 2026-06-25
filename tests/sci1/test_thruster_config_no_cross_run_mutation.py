"""Test that thruster configs are not mutated across runs."""

import copy
import numpy as np
import pytest

from arctic_quasi_dp.sci1.thruster import (
    ThrusterConfig, ThrusterAllocator, ThrusterDegradationProfile,
)


class TestNoCrossRunMutation:
    """Thruster config/degradation must not leak between runs."""

    def test_degradation_does_not_mutate_original_config(self):
        """Applying degradation to allocator must not affect the source config."""
        original = ThrusterConfig.generic_dp_vessel()
        original_degraded = [t.degraded for t in original.thrusters]

        allocator = ThrusterAllocator(copy.deepcopy(original))
        profile = ThrusterDegradationProfile.bow_degradation(0.3)
        profile.apply(allocator)

        # Original config must be unchanged
        for i, t in enumerate(original.thrusters):
            assert t.degraded == original_degraded[i], (
                f"Thruster {t.name} degraded from {original_degraded[i]} to {t.degraded}"
            )

    def test_two_allocators_independent(self):
        """Two allocators from same config must be independent."""
        config = ThrusterConfig.generic_dp_vessel()
        alloc1 = ThrusterAllocator(copy.deepcopy(config))
        alloc2 = ThrusterAllocator(copy.deepcopy(config))

        profile = ThrusterDegradationProfile.severe_degradation()
        profile.apply(alloc1)

        # alloc2 should be unaffected
        for t in alloc2.config.thrusters:
            assert t.degraded == 1.0, f"alloc2 thruster {t.name} was degraded"
            assert not t.faulty, f"alloc2 thruster {t.name} was faulted"

    def test_runner_deep_copies_config(self):
        """Runner's _get_thruster_config must return a fresh copy each time."""
        from arctic_quasi_dp.sci1.runner import _get_thruster_config
        from arctic_quasi_dp.sci1.scenarios import build_sci1_scenarios

        scenarios = build_sci1_scenarios("smoke")
        # Use B1 scenario which has generic_dp config (not None)
        scenario_b1 = [s for s in scenarios if s.scenario_id.startswith("B1")][0]
        cfg1 = _get_thruster_config(scenario_b1)
        cfg2 = _get_thruster_config(scenario_b1)

        # They should be equal but not the same object
        assert cfg1 is not cfg2, "Configs should be different objects"
        assert cfg1.name == cfg2.name

    def test_runner_deep_copies_degradation(self):
        """Runner's _get_degradation_profile must return a fresh copy each time."""
        from arctic_quasi_dp.sci1.runner import _get_degradation_profile
        from arctic_quasi_dp.sci1.scenarios import build_sci1_scenarios

        scenarios = build_sci1_scenarios("smoke")
        scenario_b1 = [s for s in scenarios if s.scenario_id.startswith("B1")][0]
        prof1 = _get_degradation_profile(scenario_b1)
        prof2 = _get_degradation_profile(scenario_b1)

        assert prof1 is not prof2, "Profiles should be different objects"
