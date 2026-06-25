"""Test that YAML config drives the runner correctly."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from arctic_quasi_dp.sci1.runner import (
    load_yaml_config,
    merge_config,
    _config_hash,
    _check_unknown_keys,
    parse_args,
)


class TestYamlConfig:
    """YAML config loading and merging."""

    def test_load_yaml_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"profile": "paper", "seeds": 50}, f)
            f.flush()
            cfg = load_yaml_config(Path(f.name))
        assert cfg["profile"] == "paper"
        assert cfg["seeds"] == 50

    def test_merge_cli_overrides_yaml(self):
        yaml_cfg = {"profile": "paper", "seeds": 50, "controllers": ["pid", "full"]}
        args = parse_args(["--profile", "smoke", "--seeds", "3"])
        merged = merge_config(yaml_cfg, args)
        # CLI overrides
        assert merged["profile"] == "smoke"
        assert merged["seeds"] == 3
        # YAML preserved
        assert merged["controllers"] == ["pid", "full"]

    def test_merge_yaml_preserved_when_cli_default(self):
        yaml_cfg = {"profile": "paper", "seeds": 50}
        args = parse_args([])  # all defaults
        merged = merge_config(yaml_cfg, args)
        assert merged["profile"] == "paper"
        assert merged["seeds"] == 50

    def test_config_hash_deterministic(self):
        cfg = {"a": 1, "b": 2}
        h1 = _config_hash(cfg)
        h2 = _config_hash(cfg)
        assert h1 == h2

    def test_config_hash_changes_with_content(self):
        h1 = _config_hash({"a": 1})
        h2 = _config_hash({"a": 2})
        assert h1 != h2

    def test_strict_mode_rejects_unknown_keys(self):
        cfg = {"profile": "smoke", "unknown_key": 42}
        with pytest.raises(ValueError, match="Unknown config keys"):
            _check_unknown_keys(cfg, strict=True)

    def test_strict_mode_allows_known_keys(self):
        cfg = {"profile": "smoke", "seeds": 2, "controllers": ["pid"]}
        _check_unknown_keys(cfg, strict=True)  # Should not raise

    def test_no_traces_cli_override(self):
        args = parse_args(["--no-traces"])
        merged = merge_config({}, args)
        assert merged["output"]["save_traces"] is False
