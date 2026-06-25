"""Test that empty experiment results are handled gracefully."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from arctic_quasi_dp.sci1.runner import run_experiments


class TestEmptyControllerSkip:
    """When all controllers are skipped, runner must not crash."""

    def test_all_skipped_generates_skip_report(self):
        """When all controllers fail to import, generate skip_report.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results"

            with patch("arctic_quasi_dp.sci1.runner.build_controller", side_effect=ImportError("no module")):
                result_dir = run_experiments(
                    profile="smoke",
                    seeds=1,
                    controllers=["nmpc"],
                    out_dir=out,
                    save_traces=False,
                )

            assert (result_dir / "metadata" / "skip_report.json").exists()
            report = json.loads((result_dir / "metadata" / "skip_report.json").read_text())
            assert report.get("skipped_all") is True

    def test_all_skipped_no_crash(self):
        """Runner must not crash when all controllers are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results"

            with patch("arctic_quasi_dp.sci1.runner.build_controller", side_effect=ImportError("no module")):
                # Should not raise
                result_dir = run_experiments(
                    profile="smoke",
                    seeds=1,
                    controllers=["nmpc"],
                    out_dir=out,
                    save_traces=False,
                )
            assert result_dir.exists()

    def test_partial_skip_records_skipped(self):
        """When some controllers are skipped, results should still be produced."""
        import arctic_quasi_dp.sci1.runner as runner_mod
        real_build = runner_mod.build_controller

        def mock_build(name):
            if name == "nmpc":
                raise ImportError("casadi not installed")
            return real_build(name)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results"

            with patch.object(runner_mod, "build_controller", side_effect=mock_build):
                result_dir = run_experiments(
                    profile="smoke",
                    seeds=1,
                    controllers=["pid", "nmpc"],
                    out_dir=out,
                    save_traces=False,
                )

            # Should have results (pid ran successfully)
            assert (result_dir / "raw" / "per_seed_metrics.csv").exists()
