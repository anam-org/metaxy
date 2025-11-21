"""Tests for multi-project feature graphs.

Tests that FeatureGraph can contain features from multiple projects and that
project information is properly tracked throughout the graph lifecycle.
"""

from __future__ import annotations

from syrupy.assertion import SnapshotAssertion

from metaxy import Feature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec
from metaxy.config import MetaxyConfig
from metaxy.models.feature import FeatureGraph


def test_graph_contains_features_from_multiple_projects(
    snapshot: SnapshotAssertion, graph: FeatureGraph
) -> None:
    """Test that a single FeatureGraph can contain features from different projects."""
    # Create features from project_a
    config_a = MetaxyConfig(project="project_a")
    MetaxyConfig.set(config_a)

    class FeatureA1(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["project_a", "feature1"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    class FeatureA2(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["project_a", "feature2"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Create features from project_b in the same graph
    config_b = MetaxyConfig(project="project_b")
    MetaxyConfig.set(config_b)

    class FeatureB1(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["project_b", "feature1"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    class FeatureB2(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["project_b", "feature2"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Graph should contain all features
    assert len(graph.features_by_key) == 4

    # Verify projects are correct
    assert FeatureA1.project == "project_a"
    assert FeatureA2.project == "project_a"
    assert FeatureB1.project == "project_b"
    assert FeatureB2.project == "project_b"

    # Snapshot the graph structure
    projects_by_feature = {
        key.to_string(): feat.project for key, feat in graph.features_by_key.items()
    }

    assert projects_by_feature == snapshot

    MetaxyConfig.reset()


def test_graph_snapshot_includes_all_projects(
    snapshot: SnapshotAssertion, graph: FeatureGraph
) -> None:
    """Test that graph.to_snapshot() includes features from all projects."""
    # Project A
    config_a = MetaxyConfig(project="snapshot_a")
    MetaxyConfig.set(config_a)

    class FeatureA(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["feature_a"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Project B
    config_b = MetaxyConfig(project="snapshot_b")
    MetaxyConfig.set(config_b)

    class FeatureB(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["feature_b"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Create snapshot
    snapshot_data = graph.to_snapshot()

    # Should have both features
    assert len(snapshot_data) == 2
    assert "feature_a" in snapshot_data
    assert "feature_b" in snapshot_data

    # Snapshot should track feature keys
    assert list(snapshot_data.keys()) == snapshot

    MetaxyConfig.reset()


def test_graph_snapshot_version_includes_all_projects(
    snapshot: SnapshotAssertion, graph: FeatureGraph
) -> None:
    """Test that graph.snapshot_version is computed from all features regardless of project."""
    # Add features from multiple projects
    config_a = MetaxyConfig(project="version_a")
    MetaxyConfig.set(config_a)

    class FeatureA(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["feature_a"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    config_b = MetaxyConfig(project="version_b")
    MetaxyConfig.set(config_b)

    class FeatureB(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["feature_b"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Get snapshot version
    snapshot_version = graph.snapshot_version

    # Should be deterministic
    assert len(snapshot_version) == 64  # SHA256 hex
    assert snapshot_version == snapshot

    # Verify it's based on all features
    # If we create a graph with only one feature, it should be different
    graph_single = FeatureGraph()

    with graph_single.use():

        class FeatureSingle(
            Feature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature_a"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    snapshot_version_single = graph_single.snapshot_version

    # Should be different (only one feature vs two)
    assert snapshot_version != snapshot_version_single

    MetaxyConfig.reset()


def test_graph_from_snapshot_preserves_projects(snapshot: SnapshotAssertion) -> None:
    """Test that FeatureGraph.to_snapshot() preserves project information."""
    original_graph = FeatureGraph()

    # Create features from multiple projects
    config_a = MetaxyConfig(project="restore_a")
    MetaxyConfig.set(config_a)

    with original_graph.use():

        class SnapshotFeatureA(
            Feature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["restore", "feature_a"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    config_b = MetaxyConfig(project="restore_b")
    MetaxyConfig.set(config_b)

    with original_graph.use():

        class SnapshotFeatureB(
            Feature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["restore", "feature_b"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # Get original projects
    original_projects = {
        key.to_string(): feat.project
        for key, feat in original_graph.features_by_key.items()
    }

    # Create snapshot
    snapshot_data = original_graph.to_snapshot()

    # Verify that snapshot contains project information for all features
    for feature_key_str, feature_data in snapshot_data.items():
        assert "project" in feature_data
        assert "metaxy_full_definition_version" in feature_data
        # Project should match what we expect
        if "feature_a" in feature_key_str:
            assert feature_data["project"] == "restore_a"
        elif "feature_b" in feature_key_str:
            assert feature_data["project"] == "restore_b"

    # Extract projects from snapshot
    snapshot_projects = {key: data["project"] for key, data in snapshot_data.items()}

    # Projects should be preserved in snapshot
    assert original_projects == snapshot_projects
    assert snapshot_projects == snapshot

    MetaxyConfig.reset()


def test_multi_project_dependency_graph(
    snapshot: SnapshotAssertion, graph: FeatureGraph
) -> None:
    """Test that features can depend on features from other projects."""
    # Project A: upstream feature
    config_a = MetaxyConfig(project="upstream_project")
    MetaxyConfig.set(config_a)

    class UpstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Project B: downstream feature depending on Project A's feature
    config_b = MetaxyConfig(project="downstream_project")
    MetaxyConfig.set(config_b)

    class DownstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["downstream"]),
            deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Both features should exist in graph
    assert len(graph.features_by_key) == 2

    # Verify projects
    assert UpstreamFeature.project == "upstream_project"
    assert DownstreamFeature.project == "downstream_project"

    # Verify dependency is tracked
    downstream_spec = graph.feature_specs_by_key[FeatureKey(["downstream"])]
    assert len(downstream_spec.deps or []) == 1
    assert downstream_spec.deps is not None  # Type assertion for basedpyright
    assert downstream_spec.deps[0].feature == FeatureKey(["upstream"])

    # Snapshot the structure
    assert {
        "upstream_project": UpstreamFeature.project,
        "downstream_project": DownstreamFeature.project,
        "has_dependency": len(downstream_spec.deps or []) > 0,
    } == snapshot

    MetaxyConfig.reset()


def test_get_downstream_features_across_projects(
    snapshot: SnapshotAssertion, graph: FeatureGraph
) -> None:
    """Test that get_downstream_features works with multi-project graphs."""
    # Root feature in project A
    config_a = MetaxyConfig(project="project_a")
    MetaxyConfig.set(config_a)

    class RootFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["root"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Mid-level feature in project B
    config_b = MetaxyConfig(project="project_b")
    MetaxyConfig.set(config_b)

    class MidFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["mid"]),
            deps=[FeatureDep(feature=FeatureKey(["root"]))],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Leaf feature in project C
    config_c = MetaxyConfig(project="project_c")
    MetaxyConfig.set(config_c)

    class LeafFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["leaf"]),
            deps=[FeatureDep(feature=FeatureKey(["mid"]))],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Get downstream features of root
    downstream = graph.get_downstream_features([FeatureKey(["root"])])

    # Should include both mid and leaf
    downstream_strings = [key.to_string() for key in downstream]

    assert "mid" in downstream_strings
    assert "leaf" in downstream_strings
    assert len(downstream) == 2

    # Verify projects
    projects = {
        "root": RootFeature.project,
        "mid": MidFeature.project,
        "leaf": LeafFeature.project,
        "downstream_keys": downstream_strings,
    }

    assert projects == snapshot

    MetaxyConfig.reset()


def test_graph_with_same_feature_key_different_projects() -> None:
    """Test that two graphs can have the same feature key in different projects.

    This is important for multi-tenant scenarios where different projects
    might use the same feature naming conventions.
    """
    # Graph 1: project_x
    graph_x = FeatureGraph()
    config_x = MetaxyConfig(project="project_x")
    MetaxyConfig.set(config_x)

    with graph_x.use():

        class FeatureInX(
            Feature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["common", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # Graph 2: project_y with SAME feature key
    graph_y = FeatureGraph()
    config_y = MetaxyConfig(project="project_y")
    MetaxyConfig.set(config_y)

    with graph_y.use():

        class FeatureInY(
            Feature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["common", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # Both should exist in their respective graphs
    assert len(graph_x.features_by_key) == 1
    assert len(graph_y.features_by_key) == 1

    # Both should have the same feature key
    assert FeatureInX.spec().key == FeatureInY.spec().key

    # But different projects
    assert FeatureInX.project == "project_x"
    assert FeatureInY.project == "project_y"
    assert FeatureInX.project != FeatureInY.project

    # And same feature_version (computational definition identical)
    assert FeatureInX.feature_version() == FeatureInY.feature_version()

    MetaxyConfig.reset()
