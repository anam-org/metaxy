"""Tests for config CLI commands."""

import json

import pytest
import tomli
from metaxy_testing import TempMetaxyProject


def test_config_print_toml(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test config print with TOML output produces valid, parseable TOML."""

    def features():
        pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["config", "print"], capsys=capsys)

        assert result.returncode == 0

        # Parse the output as TOML to verify it's valid
        config = tomli.loads(result.stdout)

        # Check expected top-level keys
        assert "store" in config
        assert config["store"] == "dev"  # Default store in test project
        assert "stores" in config
        assert isinstance(config["stores"], dict)

        # Check that configured stores are present
        assert "dev" in config["stores"]
        assert "type" in config["stores"]["dev"]
        assert "DuckDBMetadataStore" in config["stores"]["dev"]["type"]


def test_config_print_json(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test config print with JSON output produces valid JSON with expected structure."""

    def features():
        pass

    with metaxy_project.with_features(features):
        result = metaxy_project.run_cli(["config", "print", "--format", "json"], capsys=capsys)

        assert result.returncode == 0
        config = json.loads(result.stdout)

        # Check expected top-level keys
        assert "store" in config
        assert config["store"] == "dev"  # Default store in test project
        assert "stores" in config
        assert isinstance(config["stores"], dict)

        # Check that configured stores are present
        assert "dev" in config["stores"]
        assert "type_path" in config["stores"]["dev"]
        assert "DuckDBMetadataStore" in config["stores"]["dev"]["type_path"]

        # JSON output should include None values (unlike TOML)
        # Just verify it's valid JSON with the expected structure
        assert "project" in config
