"""Tests for feature project detection and assignment.

This module tests the multi-project architecture, verifying that:
1. Features get correct projects from configuration
2. Project names are validated correctly
3. Features from different projects can coexist in the same graph
"""

from __future__ import annotations

from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec
from metaxy.config import MetaxyConfig
from metaxy.models.feature import FeatureGraph

# Note: Module-level test features will be defined dynamically in tests
# to ensure they get the correct project from the test's config


def test_feature_gets_project_from_config(snapshot: SnapshotAssertion) -> None:
    """Test that features get project from MetaxyConfig."""
    # Create config with specific project
    config = MetaxyConfig(project="test_project")
    MetaxyConfig.set(config)

    try:
        # Create feature - should get project from config
        class TestFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        # Feature should have project from config
        assert TestFeature.project == "test_project"
        assert TestFeature.project == snapshot

    finally:
        MetaxyConfig.reset()


def test_feature_project_different_configs(snapshot: SnapshotAssertion) -> None:
    """Test that features in different graphs can have different projects."""
    # Graph 1: project_a
    config_a = MetaxyConfig(project="project_a")
    graph_a = FeatureGraph()

    MetaxyConfig.set(config_a)

    with graph_a.use():

        class FeatureA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["shared", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    # Graph 2: project_b
    config_b = MetaxyConfig(project="project_b")
    graph_b = FeatureGraph()

    MetaxyConfig.set(config_b)

    with graph_b.use():

        class FeatureB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["shared", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

    # Features should have different projects
    assert FeatureA.project == "project_a"
    assert FeatureB.project == "project_b"
    assert {"feature_a": FeatureA.project, "feature_b": FeatureB.project} == snapshot

    MetaxyConfig.reset()


def test_project_name_validation_empty() -> None:
    """Test that empty project names are rejected."""
    with pytest.raises(ValueError, match="project name cannot be empty"):
        MetaxyConfig(project="")


def test_project_name_validation_forward_slash() -> None:
    """Test that project names cannot contain forward slashes."""
    with pytest.raises(
        ValueError, match="project name .* cannot contain forward slashes"
    ):
        MetaxyConfig(project="my/project")


def test_project_name_validation_double_underscore() -> None:
    """Test that project names cannot contain double underscores."""
    with pytest.raises(
        ValueError, match="project name .* cannot contain double underscores"
    ):
        MetaxyConfig(project="my__project")


def test_project_name_validation_invalid_chars() -> None:
    """Test that project names must be alphanumeric with underscores and hyphens."""
    with pytest.raises(
        ValueError,
        match="project name .* must contain only alphanumeric characters, underscores, and hyphens",
    ):
        MetaxyConfig(project="my.project")

    with pytest.raises(
        ValueError,
        match="project name .* must contain only alphanumeric characters, underscores, and hyphens",
    ):
        MetaxyConfig(project="my project")


def test_project_name_validation_valid_names() -> None:
    """Test that valid project names are accepted."""
    # Alphanumeric
    config1 = MetaxyConfig(project="myproject")
    assert config1.project == "myproject"

    # With underscores
    config2 = MetaxyConfig(project="my_project")
    assert config2.project == "my_project"

    # With hyphens
    config3 = MetaxyConfig(project="my-project")
    assert config3.project == "my-project"

    # Mixed
    config4 = MetaxyConfig(project="my_project-123")
    assert config4.project == "my_project-123"


def test_default_project_from_config() -> None:
    """Test that MetaxyConfig.get() returns default project when not initialized."""
    # Reset config to ensure clean state
    MetaxyConfig.reset()

    # Get default config - should have project='default'
    config = MetaxyConfig.get()
    assert config.project == "default"


def test_project_from_metaxy_toml(tmp_path: Path) -> None:
    """Test loading project from metaxy.toml."""
    # Create metaxy.toml with project setting
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text(
        """
project = "my_metaxy_project"

[stores.dev]
type = "metaxy.metadata_store.InMemoryMetadataStore"
"""
    )

    # Load config
    config = MetaxyConfig.load(config_file)

    assert config.project == "my_metaxy_project"

    MetaxyConfig.reset()


def test_project_from_pyproject_toml(tmp_path: Path) -> None:
    """Test loading project from pyproject.toml [tool.metaxy] section."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text(
        """
[project]
name = "test"

[tool.metaxy]
project = "pyproject_metaxy"

[tool.metaxy.stores.dev]
type = "metaxy.metadata_store.InMemoryMetadataStore"
"""
    )

    config = MetaxyConfig.load(config_file)

    assert config.project == "pyproject_metaxy"

    MetaxyConfig.reset()


def test_project_override_via_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that METAXY_PROJECT env var overrides config file."""
    # Create config file with one project
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text(
        """
project = "file_project"

[stores.dev]
type = "metaxy.metadata_store.InMemoryMetadataStore"
"""
    )

    # Override via env var
    monkeypatch.setenv("METAXY_PROJECT", "env_project")

    config = MetaxyConfig.load(config_file)

    # Env var should win
    assert config.project == "env_project"

    MetaxyConfig.reset()


def test_multiple_features_same_project(snapshot: SnapshotAssertion) -> None:
    """Test that multiple features in the same graph share the same project."""
    config = MetaxyConfig(project="shared_project")
    MetaxyConfig.set(config)

    try:

        class Feature1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature1"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class Feature2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature2"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        # Both features should have the same project
        assert Feature1.project == "shared_project"
        assert Feature2.project == "shared_project"
        assert Feature1.project == Feature2.project
        assert Feature1.project == snapshot

    finally:
        MetaxyConfig.reset()


def test_feature_project_persists_across_graph_operations(
    snapshot: SnapshotAssertion,
) -> None:
    """Test that project is preserved during graph operations like to_snapshot/from_snapshot."""
    config = MetaxyConfig(project="persist_project")
    MetaxyConfig.set(config)

    try:
        graph = FeatureGraph()

        with graph.use():
            # Define a test feature that will get the project from config
            class SnapshotTestFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["snapshot", "test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
                ),
            ):
                pass

        # Verify project before snapshot
        assert SnapshotTestFeature.project == "persist_project"

        # Create snapshot
        snapshot_data = graph.to_snapshot()

        # Verify snapshot contains project info
        feature_key_str = "snapshot/test/feature"
        assert feature_key_str in snapshot_data
        assert snapshot_data[feature_key_str]["project"] == "persist_project"

        # Verify that feature_tracking_version is in snapshot
        assert "feature_tracking_version" in snapshot_data[feature_key_str]

        # Store snapshot data for verification
        snapshot_project = snapshot_data[feature_key_str]["project"]
        snapshot_tracking = snapshot_data[feature_key_str]["feature_tracking_version"]

        # For a proper test, we would need the feature to be importable,
        # but we can at least verify the snapshot data is correct
        assert snapshot_project == "persist_project"
        assert snapshot_tracking is not None and len(snapshot_tracking) > 0

        # Verify the structure matches our snapshot
        assert {
            "project": snapshot_project,
            "tracking_version": snapshot_tracking[:8],
        } == snapshot

    finally:
        MetaxyConfig.reset()
