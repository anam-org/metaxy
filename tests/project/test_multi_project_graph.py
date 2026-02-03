"""Tests for multi-project feature graphs.

Tests that FeatureGraph can contain features from multiple projects and that
project information is properly tracked throughout the graph lifecycle.
"""

from __future__ import annotations

from types import ModuleType

from metaxy_testing.models import SampleFeatureSpec
from syrupy.assertion import SnapshotAssertion

from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy._packaging import detect_project_from_package
from metaxy.config import MetaxyConfig
from metaxy.models.feature import FeatureGraph


def create_feature_in_package(
    package_name: str,
    feature_key: list[str],
    deps: list | None = None,
) -> type[BaseFeature]:
    """Create a feature class that appears to be from a specific package."""
    spec_kwargs: dict = {
        "key": FeatureKey(feature_key),
        "fields": [FieldSpec(key=FieldKey(["default"]), code_version="1")],
    }
    if deps:
        spec_kwargs["deps"] = deps

    class TempFeature(
        BaseFeature,
        spec=SampleFeatureSpec(**spec_kwargs),
    ):
        pass

    TempFeature.__module__ = package_name
    TempFeature.__metaxy_project__ = detect_project_from_package(package_name)
    return TempFeature


def test_graph_contains_features_from_multiple_projects(
    snapshot: SnapshotAssertion,
    graph: FeatureGraph,
    project_a_package: ModuleType,
    project_b_package: ModuleType,
) -> None:
    """Test that a single FeatureGraph can contain features from different projects."""
    # Create features from project_a
    FeatureA1 = create_feature_in_package("fake_project_a_pkg", ["project_a", "feature1"])
    FeatureA2 = create_feature_in_package("fake_project_a_pkg", ["project_a", "feature2"])

    # Create features from project_b
    FeatureB1 = create_feature_in_package("fake_project_b_pkg", ["project_b", "feature1"])
    FeatureB2 = create_feature_in_package("fake_project_b_pkg", ["project_b", "feature2"])

    # Graph should contain all features
    assert len(graph.feature_definitions_by_key) == 4

    # Verify projects are correct
    assert FeatureA1.metaxy_project() == "project_a"
    assert FeatureA2.metaxy_project() == "project_a"
    assert FeatureB1.metaxy_project() == "project_b"
    assert FeatureB2.metaxy_project() == "project_b"

    # Snapshot the graph structure
    projects_by_feature = {
        key.to_string(): definition.project for key, definition in graph.feature_definitions_by_key.items()
    }

    assert projects_by_feature == snapshot

    MetaxyConfig.reset()


def test_graph_snapshot_includes_all_projects(
    snapshot: SnapshotAssertion,
    graph: FeatureGraph,
    project_a_package: ModuleType,
    project_b_package: ModuleType,
) -> None:
    """Test that graph.to_snapshot() includes features from all projects."""
    # Create features from both projects
    create_feature_in_package("fake_project_a_pkg", ["feature_a"])
    create_feature_in_package("fake_project_b_pkg", ["feature_b"])

    # Create snapshot
    snapshot_data = graph.to_snapshot()

    # Should have both features
    assert len(snapshot_data) == 2
    assert "feature_a" in snapshot_data
    assert "feature_b" in snapshot_data

    # Snapshot should track feature keys
    assert list(snapshot_data.keys()) == snapshot

    MetaxyConfig.reset()


def test_graph_snapshot_version_uses_configured_project(
    graph: FeatureGraph,
) -> None:
    """Test that graph.snapshot_version only includes features from the configured project."""

    # Add features from multiple projects with __metaxy_project__ defined in class body
    class FeatureA(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["feature_a"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        __metaxy_project__ = "project_a"

    class FeatureB(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["feature_b"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        __metaxy_project__ = "project_b"

    # With project_a configured, snapshot_version should only include project_a features
    config_a = MetaxyConfig(project="project_a")
    with config_a.use():
        snapshot_a = graph.snapshot_version
        assert len(snapshot_a) == 8  # SHA256 hex truncated

    # With project_b configured, snapshot should be different
    config_b = MetaxyConfig(project="project_b")
    with config_b.use():
        snapshot_b = graph.snapshot_version
        assert len(snapshot_b) == 8

    # Different projects should produce different snapshots
    assert snapshot_a != snapshot_b

    # Verify snapshot matches project-specific version
    assert snapshot_a == graph.get_project_snapshot_version("project_a")
    assert snapshot_b == graph.get_project_snapshot_version("project_b")

    MetaxyConfig.reset()


def test_graph_snapshot_version_raises_with_multiple_projects_and_no_config(
    graph: FeatureGraph,
) -> None:
    """Test that graph.snapshot_version raises when features span multiple projects and no project is configured."""
    import pytest

    class FeatureA(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["feature_a"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        __metaxy_project__ = "project_a"

    class FeatureB(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["feature_b"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        __metaxy_project__ = "project_b"

    # Without a project configured and multiple projects in graph, snapshot_version should raise
    config = MetaxyConfig()
    with config.use():
        with pytest.raises(RuntimeError, match="multiple projects"):
            _ = graph.snapshot_version

    MetaxyConfig.reset()


def test_graph_snapshot_version_works_with_single_project_no_config(
    graph: FeatureGraph,
) -> None:
    """Test that graph.snapshot_version works when all features belong to a single project."""

    class FeatureA(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["feature_a"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        __metaxy_project__ = "project_a"

    class FeatureA2(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["feature_a2"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        __metaxy_project__ = "project_a"

    # With single project in graph, snapshot_version should work without config
    config = MetaxyConfig()
    with config.use():
        snapshot = graph.snapshot_version
        assert len(snapshot) == 8
        assert snapshot == graph.get_project_snapshot_version("project_a")

    MetaxyConfig.reset()


def test_graph_from_snapshot_preserves_projects(
    snapshot: SnapshotAssertion,
    project_a_package: ModuleType,
    project_b_package: ModuleType,
) -> None:
    """Test that FeatureGraph.to_snapshot() preserves project information."""
    original_graph = FeatureGraph()

    with original_graph.use():
        # Create features - note that project is captured from module at class
        # definition time, so the create_feature_in_package helper's __metaxy_project__
        # assignment doesn't update the stored FeatureDefinition.
        create_feature_in_package("fake_project_a_pkg", ["restore", "feature_a"])
        create_feature_in_package("fake_project_b_pkg", ["restore", "feature_b"])

    # Get original projects from the stored definitions
    original_projects = {
        key.to_string(): definition.project for key, definition in original_graph.feature_definitions_by_key.items()
    }

    # Create snapshot
    snapshot_data = original_graph.to_snapshot()

    # Verify that snapshot contains project information for all features
    for feature_key_str, feature_data in snapshot_data.items():
        assert "project" in feature_data
        assert "metaxy_definition_version" in feature_data

    # Extract projects from snapshot
    snapshot_projects = {key: data["project"] for key, data in snapshot_data.items()}

    # Projects in snapshot should match those in the stored definitions
    assert original_projects == snapshot_projects
    assert snapshot_projects == snapshot

    MetaxyConfig.reset()


def test_multi_project_dependency_graph(
    snapshot: SnapshotAssertion,
    graph: FeatureGraph,
    project_a_package: ModuleType,
    project_b_package: ModuleType,
) -> None:
    """Test that features can depend on features from other projects."""
    # Project A: upstream feature
    UpstreamFeature = create_feature_in_package("fake_project_a_pkg", ["upstream"])

    # Project B: downstream feature depending on Project A's feature
    DownstreamFeature = create_feature_in_package(
        "fake_project_b_pkg",
        ["downstream"],
        deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
    )

    # Both features should exist in graph
    assert len(graph.feature_definitions_by_key) == 2

    # Verify projects
    assert UpstreamFeature.metaxy_project() == "project_a"
    assert DownstreamFeature.metaxy_project() == "project_b"

    # Verify dependency is tracked
    downstream_def = graph.get_feature_definition(FeatureKey(["downstream"]))
    assert len(downstream_def.spec.deps or []) == 1
    assert downstream_def.spec.deps is not None  # Type assertion for type checker
    assert downstream_def.spec.deps[0].feature == FeatureKey(["upstream"])

    # Snapshot the structure
    assert {
        "upstream_project": UpstreamFeature.metaxy_project(),
        "downstream_project": DownstreamFeature.metaxy_project(),
        "has_dependency": len(downstream_def.spec.deps or []) > 0,
    } == snapshot

    MetaxyConfig.reset()


def test_get_downstream_features_across_projects(
    snapshot: SnapshotAssertion,
    graph: FeatureGraph,
    project_a_package: ModuleType,
    project_b_package: ModuleType,
    project_c_package: ModuleType,
) -> None:
    """Test that get_downstream_features works with multi-project graphs."""
    # Root feature in project A
    RootFeature = create_feature_in_package("fake_project_a_pkg", ["root"])

    # Mid-level feature in project B
    MidFeature = create_feature_in_package(
        "fake_project_b_pkg",
        ["mid"],
        deps=[FeatureDep(feature=FeatureKey(["root"]))],
    )

    # Leaf feature in project C
    LeafFeature = create_feature_in_package(
        "fake_project_c_pkg",
        ["leaf"],
        deps=[FeatureDep(feature=FeatureKey(["mid"]))],
    )

    # Get downstream features of root
    downstream = graph.get_downstream_features([FeatureKey(["root"])])

    # Should include both mid and leaf
    downstream_strings = [key.to_string() for key in downstream]

    assert "mid" in downstream_strings
    assert "leaf" in downstream_strings
    assert len(downstream) == 2

    # Verify projects
    projects = {
        "root": RootFeature.metaxy_project(),
        "mid": MidFeature.metaxy_project(),
        "leaf": LeafFeature.metaxy_project(),
        "downstream_keys": downstream_strings,
    }

    assert projects == snapshot

    MetaxyConfig.reset()


def test_graph_with_same_feature_key_different_projects(
    project_a_package: ModuleType,
    project_b_package: ModuleType,
) -> None:
    """Test that two graphs can have the same feature key in different projects.

    This is important for multi-tenant scenarios where different projects
    might use the same feature naming conventions.
    """
    # Graph 1: project_a
    graph_x = FeatureGraph()

    with graph_x.use():
        FeatureInX = create_feature_in_package("fake_project_a_pkg", ["common", "feature"])

    # Graph 2: project_b with SAME feature key
    graph_y = FeatureGraph()

    with graph_y.use():
        FeatureInY = create_feature_in_package("fake_project_b_pkg", ["common", "feature"])

    # Both should exist in their respective graphs
    assert len(graph_x.feature_definitions_by_key) == 1
    assert len(graph_y.feature_definitions_by_key) == 1

    # Both should have the same feature key
    assert FeatureInX.spec().key == FeatureInY.spec().key

    # But different projects
    assert FeatureInX.metaxy_project() == "project_a"
    assert FeatureInY.metaxy_project() == "project_b"
    assert FeatureInX.metaxy_project() != FeatureInY.metaxy_project

    # And same feature_version (computational definition identical)
    assert FeatureInX.feature_version() == FeatureInY.feature_version()

    MetaxyConfig.reset()
