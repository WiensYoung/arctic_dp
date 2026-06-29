"""Verify baseline controllers are honestly named in documentation and manifests."""

import pytest


class TestBaselineNamingClaims:
    """Ensure proxy baselines are not claimed as full implementations."""

    def test_tube_mpc_docstring_honest(self):
        """TubeMPCController docstring must state it is a proxy."""
        from arctic_quasi_dp.sci1.baseline_controllers import TubeMPCController
        doc = TubeMPCController.__doc__ or ""
        assert "proxy" in doc.lower() or "NOT" in doc, \
            "TubeMPCController docstring must state it is a proxy, not a complete tube MPC"

    def test_robust_mpc_docstring_honest(self):
        """RobustMPCController docstring must state it is a proxy."""
        from arctic_quasi_dp.sci1.baseline_controllers import RobustMPCController
        doc = RobustMPCController.__doc__ or ""
        assert "proxy" in doc.lower() or "tighten" in doc.lower() or "margin" in doc.lower(), \
            "RobustMPCController docstring must state it uses margin tightening"

    def test_adrc_docstring_honest(self):
        """ADRCController (adrc_proxy) docstring must state it is a proxy."""
        from arctic_quasi_dp.sci1.baseline_controllers import ADRCController
        doc = ADRCController.__doc__ or ""
        assert "proxy" in doc.lower() or "lightweight" in doc.lower() or "simplified" in doc.lower(), \
            "ADRCController docstring must state it is a simplified/proxy implementation"

    def test_leso_adrc_is_full_implementation(self):
        """LESOADRCController should describe a proper LESO implementation."""
        from arctic_quasi_dp.sci1.baseline_controllers import LESOADRCController
        doc = LESOADRCController.__doc__ or ""
        assert "LESO" in doc or "Linear Extended State Observer" in doc, \
            "LESOADRCController must describe LESO"

    def test_controller_capability_matrix_no_false_claims(self):
        """Controller capability matrix should not claim CBF/CVaR for baselines that lack them."""
        from arctic_quasi_dp.sci1.tables import _CONTROLLER_CAPABILITIES
        # ADRC/LESO should not claim CBF
        for name in ["adrc", "leso_adrc"]:
            if name in _CONTROLLER_CAPABILITIES:
                caps = _CONTROLLER_CAPABILITIES[name]
                assert not caps.get("cbf", False), f"{name} should not claim CBF capability"
                assert not caps.get("cvar", False), f"{name} should not claim CVaR capability"
