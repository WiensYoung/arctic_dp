"""Verify documentation does not contain overclaim language."""

import re
from pathlib import Path
import pytest

# Phrases that must NOT appear as positive claims
# These are checked only in files that are NOT limitation/disclaimer docs
FORBIDDEN_PHRASES = [
    r"full-scale validation completed",
    r"formally proven.*safe",
    r"real Copernicus validation completed",
    r"real ERA5 validation completed",
    r"已证明.*安全",
]

# Files that are explicitly limitation/disclaimer documents - exempt from checks
EXEMPT_FILES = [
    "METHOD_HOCBF_LIMITATIONS.md",
    "METHOD_THEORY_SKETCH.md",
    "test_no_overclaim_docs.py",
    "scale_analysis.py",
    "sci1_scale_comparison.yaml",
    "AUDIT_REPORT.md",
    "FIX_SUMMARY.md",
]


def _scan_files(patterns, root: Path, extensions=(".py", ".md", ".yaml", ".json")):
    """Scan files for forbidden patterns, excluding limitation docs."""
    violations = []
    for ext in extensions:
        for fpath in root.rglob(f"*{ext}"):
            if "__pycache__" in str(fpath) or ".git" in str(fpath):
                continue
            # Skip exempt files
            if any(exempt in str(fpath) for exempt in EXEMPT_FILES):
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append((fpath, pattern))
    return violations


class TestNoOverclaimDocs:
    """Verify no overclaim language in source and documentation."""

    def test_no_fullscale_validation_claim(self):
        """Source/docs must not claim full-scale validation."""
        root = Path(__file__).resolve().parents[2]
        violations = _scan_files(FORBIDDEN_PHRASES, root)
        if violations:
            msg = "Found forbidden overclaim phrases:\n"
            for fpath, pattern in violations:
                msg += f"  {fpath}: {pattern}\n"
            pytest.fail(msg)

    def test_mock_fixture_not_called_real(self):
        """Mock fixture must be labeled as mock, not real."""
        from arctic_quasi_dp.sci1.data_sources import PACKAGED_REPLAY_SOURCES
        for src in PACKAGED_REPLAY_SOURCES:
            assert "mock" in src.status.lower() or "fixture" in src.status.lower() or \
                   "mock" in src.access_note.lower() or "not a real" in src.access_note.lower(), \
                   f"Mock fixture {src.name} must be labeled as mock"

    def test_scale_comparison_config_honest(self):
        """scale_comparison config must not claim full-scale validation."""
        config_path = Path(__file__).resolve().parents[2] / "configs" / "sci1" / "sci1_scale_comparison.yaml"
        if not config_path.exists():
            pytest.skip("scale_comparison config not found")
        text = config_path.read_text(encoding="utf-8")
        assert "NOT a full-scale" in text or "not a full-scale" in text or \
               "normalized" in text, "scale_comparison config must state it is not full-scale validation"
