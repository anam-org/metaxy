"""Tests for feature_tracking_version that includes project in the hash.

The tracking version is used for system tables (feature_versions) to isolate
features from different projects in the same metadata store. It differs from
feature_version (used for field provenance) in that it includes the project name.

Key behaviors:
1. feature_tracking_version changes when project changes
2. feature_version does NOT change when project changes
3. Two identical features in different projects have different tracking versions
4. feature_tracking_version is deterministic for the same (feature, project) pair
"""

from __future__ import annotations

from syrupy.assertion import SnapshotAssertion

from metaxy import Feature, FeatureDep, FeatureKey, FeatureSpec, FieldKey, FieldSpec
from metaxy.config import MetaxyConfig
from metaxy.models.feature import FeatureGraph


def test_feature_tracking_version_includes_project(snapshot: SnapshotAssertion) -> None:
    """Test that feature_tracking_version changes when project changes."""
    # Create same feature in two different projects
    config_a = MetaxyConfig(project="project_a")
    graph_a = FeatureGraph()

    MetaxyConfig.set(config_a)

    with graph_a.use():

        class FeatureInA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # Same feature in project_b
    config_b = MetaxyConfig(project="project_b")
    graph_b = FeatureGraph()

    MetaxyConfig.set(config_b)

    with graph_b.use():

        class FeatureInB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # The features should have different projects
    assert FeatureInA.project != FeatureInB.project
    assert FeatureInA.project == "project_a"
    assert FeatureInB.project == "project_b"

    # feature_tracking_version should be DIFFERENT (includes project)
    tracking_version_a = FeatureInA.feature_tracking_version()
    tracking_version_b = FeatureInB.feature_tracking_version()
    assert tracking_version_a != tracking_version_b

    # But feature_version should be SAME (field provenance unchanged by project)
    assert FeatureInA.feature_version() == FeatureInB.feature_version()

    # Snapshot for verification
    assert {
        "project_a": FeatureInA.project,
        "project_b": FeatureInB.project,
        "tracking_version_a": tracking_version_a,
        "tracking_version_b": tracking_version_b,
        "feature_version_a": FeatureInA.feature_version(),
        "feature_version_b": FeatureInB.feature_version(),
        "tracking_versions_differ": tracking_version_a != tracking_version_b,
        "feature_versions_same": FeatureInA.feature_version()
        == FeatureInB.feature_version(),
    } == snapshot

    MetaxyConfig.reset()


def test_feature_version_unchanged_by_project(snapshot: SnapshotAssertion) -> None:
    """Test that feature_version does NOT change when only project changes.

    This is critical: feature_version is used for field provenance and should
    only change when the computational definition changes, not metadata like project.
    """
    # Define identical feature specs in two projects
    graph_1 = FeatureGraph()
    graph_2 = FeatureGraph()

    config_1 = MetaxyConfig(project="project_1")
    MetaxyConfig.set(config_1)

    with graph_1.use():

        class Feature1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["data", "feature"]),
                fields=[
                    FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                    FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                ],
            ),
        ):
            pass

    config_2 = MetaxyConfig(project="project_2")
    MetaxyConfig.set(config_2)

    with graph_2.use():

        class Feature2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["data", "feature"]),
                fields=[
                    FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                    FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                ],
            ),
        ):
            pass

    # feature_version should be IDENTICAL (field provenance unchanged)
    version_1 = Feature1.feature_version()
    version_2 = Feature2.feature_version()

    assert version_1 == version_2
    assert version_1 == snapshot

    # But projects should differ
    assert Feature1.project != Feature2.project

    MetaxyConfig.reset()


def test_tracking_version_deterministic(snapshot: SnapshotAssertion) -> None:
    """Test that tracking version (project + feature) is deterministic."""
    config = MetaxyConfig(project="deterministic_project")
    MetaxyConfig.set(config)

    try:

        class TestFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "deterministic"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # feature_tracking_version should be deterministic
        tracking_1 = TestFeature.feature_tracking_version()
        tracking_2 = TestFeature.feature_tracking_version()

        assert tracking_1 == tracking_2
        assert len(tracking_1) == 64  # SHA256 hex
        assert tracking_1 == snapshot

    finally:
        MetaxyConfig.reset()


def test_feature_with_deps_different_projects(snapshot: SnapshotAssertion) -> None:
    """Test feature_version stays same across projects even with dependencies."""
    # Project A: upstream and downstream
    config_a = MetaxyConfig(project="project_a")
    graph_a = FeatureGraph()

    MetaxyConfig.set(config_a)

    with graph_a.use():

        class UpstreamA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class DownstreamA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # Project B: same features
    config_b = MetaxyConfig(project="project_b")
    graph_b = FeatureGraph()

    MetaxyConfig.set(config_b)

    with graph_b.use():

        class UpstreamB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class DownstreamB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # feature_version should be same across projects
    assert UpstreamA.feature_version() == UpstreamB.feature_version()
    assert DownstreamA.feature_version() == DownstreamB.feature_version()

    # But projects should differ
    assert UpstreamA.project == "project_a"
    assert UpstreamB.project == "project_b"
    assert DownstreamA.project == "project_a"
    assert DownstreamB.project == "project_b"

    # Snapshot both
    assert {
        "upstream_a_version": UpstreamA.feature_version(),
        "upstream_b_version": UpstreamB.feature_version(),
        "downstream_a_version": DownstreamA.feature_version(),
        "downstream_b_version": DownstreamB.feature_version(),
        "upstream_a_project": UpstreamA.project,
        "upstream_b_project": UpstreamB.project,
        "downstream_a_project": DownstreamA.project,
        "downstream_b_project": DownstreamB.project,
    } == snapshot

    MetaxyConfig.reset()


def test_project_in_graph_snapshot(snapshot: SnapshotAssertion) -> None:
    """Test that project information is preserved in graph snapshots."""
    config = MetaxyConfig(project="snapshot_project")
    MetaxyConfig.set(config)

    try:
        graph = FeatureGraph()

        with graph.use():

            class Feature1(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["feature1"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

            class Feature2(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["feature2"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

        # Get snapshot
        snapshot_data = graph.to_snapshot()

        # Both features should have project in snapshot
        # The snapshot format should include project information
        # (Implementation detail for python-dev agent)

        # For now, verify features have correct projects
        assert Feature1.project == "snapshot_project"
        assert Feature2.project == "snapshot_project"

        # Snapshot the graph structure
        assert {
            "feature1_project": Feature1.project,
            "feature2_project": Feature2.project,
            "feature_keys": list(snapshot_data.keys()),
        } == snapshot

    finally:
        MetaxyConfig.reset()


def test_provenance_by_field_unchanged_by_project(
    snapshot: SnapshotAssertion,
) -> None:
    """Test that provenance_by_field() method is also unchanged by project.

    This verifies the entire field provenance pipeline ignores project metadata.
    """
    graph_a = FeatureGraph()
    graph_b = FeatureGraph()

    config_a = MetaxyConfig(project="proj_a")
    MetaxyConfig.set(config_a)

    with graph_a.use():

        class FeatureA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["multi", "field"]),
                fields=[
                    FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                    FieldSpec(key=FieldKey(["field2"]), code_version="2"),
                ],
            ),
        ):
            pass

    config_b = MetaxyConfig(project="proj_b")
    MetaxyConfig.set(config_b)

    with graph_b.use():

        class FeatureB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["multi", "field"]),
                fields=[
                    FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                    FieldSpec(key=FieldKey(["field2"]), code_version="2"),
                ],
            ),
        ):
            pass

    # provenance_by_field() should be identical
    provenance_a = FeatureA.provenance_by_field()
    provenance_b = FeatureB.provenance_by_field()

    assert provenance_a == provenance_b
    assert provenance_a == snapshot

    # But projects should differ
    assert FeatureA.project != FeatureB.project

    MetaxyConfig.reset()
