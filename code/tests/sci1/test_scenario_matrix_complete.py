"""Test scenario matrix completeness and correctness."""

import pytest
from arctic_quasi_dp.sci1.scenarios import build_sci1_scenarios


class TestScenarioMatrixComplete:
    """Scenario matrix must cover all required groups and properties."""

    def test_all_groups_present(self):
        scenarios = build_sci1_scenarios("paper")
        groups = {s.group for s in scenarios}
        required = {"A_precision", "B_ice_enhancement", "C_fault_tolerance",
                    "D_safety_degradation", "E_sensor_degradation", "F_runtime",
                    "G_ice_sensitivity"}
        assert required.issubset(groups), f"Missing groups: {required - groups}"

    def test_minimum_scenario_count(self):
        scenarios = build_sci1_scenarios("paper")
        assert len(scenarios) >= 30, f"Expected >= 30 scenarios, got {len(scenarios)}"

    def test_unique_scenario_ids(self):
        scenarios = build_sci1_scenarios("paper")
        ids = [s.scenario_id for s in scenarios]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"

    def test_each_scenario_has_group(self):
        for s in build_sci1_scenarios("paper"):
            assert s.group, f"{s.scenario_id} missing group"

    def test_each_scenario_has_description(self):
        for s in build_sci1_scenarios("paper"):
            assert s.description, f"{s.scenario_id} missing description"

    def test_each_scenario_has_evidence_level(self):
        for s in build_sci1_scenarios("paper"):
            assert s.evidence_level, f"{s.scenario_id} missing evidence_level"

    def test_each_scenario_has_primary_claim(self):
        for s in build_sci1_scenarios("paper"):
            assert s.primary_claim, f"{s.scenario_id} missing primary_claim"

    def test_smoke_profile_short_duration(self):
        scenarios = build_sci1_scenarios("smoke")
        for s in scenarios:
            assert s.duration <= 30, f"{s.scenario_id} duration too long for smoke: {s.duration}"

    def test_paper_profile_reasonable_duration(self):
        scenarios = build_sci1_scenarios("paper")
        for s in scenarios:
            assert s.duration >= 100, f"{s.scenario_id} duration too short for paper: {s.duration}"
