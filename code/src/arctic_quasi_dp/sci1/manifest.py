"""Auto-generate experiment manifest from runtime configuration.

This module replaces the static hand-written data_manifest.json with
a runtime-generated manifest that is always consistent with the actual
experiment configuration.

Generates:
- run_manifest.json: experiment metadata and provenance
- artifact_manifest.csv: list of output files with checksums
- data_usage_manifest.csv: actual data files used

Reference:
- ACM Artifact Review and Badging v1.1
- IEEE Code and Data Submission Guidelines
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RunManifest:
    """Experiment run manifest for reproducibility."""
    # Run identification
    run_id: str = ""
    created_at_utc: str = ""
    config_path: str = ""
    config_sha256: str = ""

    # Environment
    python_version: str = ""
    platform: str = ""
    numpy_version: str = ""
    scipy_version: str = ""

    # Experiment configuration
    scenario_ids: List[str] = field(default_factory=list)
    controller_ids: List[str] = field(default_factory=list)
    seed_list: List[int] = field(default_factory=list)
    profile: str = ""
    dt: float = 0.0
    duration_per_scenario: float = 0.0

    # Task accounting
    task_count_expected: int = 0
    task_count_completed: int = 0
    task_count_failed: int = 0
    task_count_skipped: int = 0

    # Data provenance
    data_files_used: List[Dict[str, str]] = field(default_factory=list)
    data_sources_manifest: List[Dict[str, Any]] = field(default_factory=list)

    # Output artifacts
    artifact_files: List[Dict[str, str]] = field(default_factory=list)

    # Vessel configuration
    vessel_name: str = ""
    vessel_mass_kg: float = 0.0
    vessel_length_m: float = 0.0
    vessel_beam_m: float = 0.0

    # Protocol metadata
    protocol: str = ""
    experiment_version: str = "1.0"
    notes: str = ""


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not file_path.exists():
        return ""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_manifest_from_config(
    config: Dict[str, Any],
    config_path: Optional[Path] = None,
    completed_tasks: Optional[List[Dict[str, Any]]] = None,
    data_usage: Optional[List[Dict[str, Any]]] = None,
    artifact_dir: Optional[Path] = None,
) -> RunManifest:
    """Build a RunManifest from experiment configuration.

    Args:
        config: Experiment configuration dict (from YAML)
        config_path: Path to config file
        completed_tasks: List of completed task results
        data_usage: List of data files used
        artifact_dir: Directory containing output artifacts

    Returns:
        Populated RunManifest
    """
    manifest = RunManifest()

    # Run identification
    manifest.run_id = generate_run_id()
    manifest.created_at_utc = datetime.now(timezone.utc).isoformat()

    if config_path and config_path.exists():
        manifest.config_path = str(config_path)
        manifest.config_sha256 = compute_file_sha256(config_path)

    # Environment
    manifest.python_version = sys.version
    manifest.platform = platform.platform()
    try:
        manifest.numpy_version = np.__version__
    except AttributeError:
        manifest.numpy_version = "unknown"
    try:
        import scipy
        manifest.scipy_version = scipy.__version__
    except ImportError:
        manifest.scipy_version = "not_installed"

    # Experiment configuration
    manifest.scenario_ids = list(config.get("scenario_ids", []))
    manifest.controller_ids = list(config.get("controllers", []))
    seeds_val = config.get("seeds", [])
    if isinstance(seeds_val, int):
        manifest.seed_list = list(range(seeds_val))
    else:
        manifest.seed_list = list(seeds_val)
    manifest.profile = config.get("profile", "unknown")

    # dt/duration 来自场景而非全局配置, 0.0 表示 per-scenario (各场景不同)
    sim_config = config.get("simulation", {})
    manifest.dt = sim_config.get("dt", 0.0)
    manifest.duration_per_scenario = sim_config.get("duration", 0.0)

    # Task accounting
    n_scenarios = len(manifest.scenario_ids)
    n_controllers = len(manifest.controller_ids)
    n_seeds = len(manifest.seed_list)
    manifest.task_count_expected = n_scenarios * n_controllers * n_seeds

    if completed_tasks:
        manifest.task_count_completed = sum(
            1 for t in completed_tasks if t.get("status") == "ok"
        )
        manifest.task_count_failed = sum(
            1 for t in completed_tasks if t.get("status") == "failed"
        )
        manifest.task_count_skipped = sum(
            1 for t in completed_tasks if t.get("status") == "skipped"
        )

    # Data provenance
    if data_usage:
        manifest.data_files_used = [
            {
                "path": str(d.get("path", "")),
                "variable": str(d.get("variable", "")),
                "source": str(d.get("source", "")),
            }
            for d in data_usage
        ]

    # Vessel configuration
    vessel_config = config.get("vessel", {})
    manifest.vessel_name = vessel_config.get("name", "simplified_500t")
    manifest.vessel_mass_kg = vessel_config.get("mass_kg", 500000.0)
    manifest.vessel_length_m = vessel_config.get("length_m", 122.5)
    manifest.vessel_beam_m = vessel_config.get("beam_m", 22.0)

    # Protocol
    manifest.protocol = config.get("protocol", "")
    manifest.notes = config.get("notes", "")

    # Artifact files
    if artifact_dir and artifact_dir.exists():
        for f in sorted(artifact_dir.rglob("*")):
            if f.is_file():
                manifest.artifact_files.append({
                    "path": str(f.relative_to(artifact_dir)),
                    "size_bytes": f.stat().st_size,
                    "sha256": compute_file_sha256(f),
                })

    return manifest


def save_manifest(manifest: RunManifest, output_dir: Path) -> Path:
    """Save manifest to JSON file.

    Args:
        manifest: The manifest to save
        output_dir: Directory to write manifest.json

    Returns:
        Path to saved manifest file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "run_manifest.json"

    # Convert to dict, handling non-serializable types
    data = asdict(manifest)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    return output_path


def validate_manifest(manifest: RunManifest) -> List[str]:
    """Validate manifest for consistency.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check task count consistency
    if manifest.task_count_expected > 0:
        total_accounted = (
            manifest.task_count_completed
            + manifest.task_count_failed
            + manifest.task_count_skipped
        )
        if total_accounted != manifest.task_count_expected:
            errors.append(
                f"Task count mismatch: expected={manifest.task_count_expected}, "
                f"accounted={total_accounted} "
                f"(completed={manifest.task_count_completed}, "
                f"failed={manifest.task_count_failed}, "
                f"skipped={manifest.task_count_skipped})"
            )

    # Check scenario/controller/seed consistency
    if manifest.scenario_ids and manifest.controller_ids and manifest.seed_list:
        expected = len(manifest.scenario_ids) * len(manifest.controller_ids) * len(manifest.seed_list)
        if expected != manifest.task_count_expected:
            errors.append(
                f"Task count formula mismatch: "
                f"{len(manifest.scenario_ids)} scenarios x "
                f"{len(manifest.controller_ids)} controllers x "
                f"{len(manifest.seed_list)} seeds = {expected}, "
                f"but task_count_expected = {manifest.task_count_expected}"
            )

    # Check data files exist
    for df in manifest.data_files_used:
        path = Path(df.get("path", ""))
        if path and not path.exists():
            errors.append(f"Data file not found: {path}")

    return errors
