"""Test effect size direction: positive d means proposed is better."""

import numpy as np
import pytest

from arctic_quasi_dp.sci1.metrics import _cohens_d


class TestEffectSizeDirection:
    """Cohen's d must be positive when proposed is better."""

    def test_positive_when_proposed_lower(self):
        """For lower-is-better metrics, proposed < baseline => d > 0."""
        baseline = np.array([10.0, 12.0, 11.0, 13.0, 10.5])
        proposed = np.array([5.0, 6.0, 5.5, 6.5, 5.2])
        d = _cohens_d(baseline, proposed)
        assert d > 0, f"Expected positive d, got {d}"

    def test_negative_when_proposed_higher(self):
        """For lower-is-better metrics, proposed > baseline => d < 0."""
        baseline = np.array([5.0, 6.0, 5.5, 6.5, 5.2])
        proposed = np.array([10.0, 12.0, 11.0, 13.0, 10.5])
        d = _cohens_d(baseline, proposed)
        assert d < 0, f"Expected negative d, got {d}"

    def test_zero_when_equal(self):
        """When both are equal, d ≈ 0."""
        baseline = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        proposed = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        d = _cohens_d(baseline, proposed)
        assert abs(d) < 0.01

    def test_few_samples_returns_nan(self):
        """With < 2 samples, should return NaN."""
        d = _cohens_d(np.array([1.0]), np.array([2.0]))
        assert np.isnan(d)
