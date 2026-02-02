"""Tests for feature_definition_version (excludes project from the hash).

The definition version is used for system tables (feature_versions) to track when
the feature definition changes. It differs from feature_version (used for field
provenance) in that it includes the Pydantic model schema.

Key behaviors:
1. feature_definition_version does NOT change when project changes
2. feature_version does NOT change when project changes
3. Two identical features (same spec, same schema) have the same definition version
4. feature_definition_version is deterministic for the same (spec, schema) pair
"""

from __future__ import annotations

from metaxy_testing.models import SampleFeatureSpec
from syrupy.assertion import SnapshotAssertion

from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy.models.feature import FeatureGraph


def test_definition_version_excludes_project(snapshot: SnapshotAssertion) -> None:
    """Test that feature_definition_version does NOT change when project changes."""
    # Create same feature in two different projects (using same class name for identical schema)
    graph_a = FeatureGraph()

    with graph_a.use():

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    graph_b = FeatureGraph()

    with graph_b.use():
        # Same class name ensures identical schema
        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # Get definitions from both graphs
    definition_a = graph_a.get_feature_definition(FeatureKey(["test", "feature"]))
    definition_b = graph_b.get_feature_definition(FeatureKey(["test", "feature"]))

    # feature_definition_version should be SAME (excludes project)
    definition_version_a = definition_a.feature_definition_version
    definition_version_b = definition_b.feature_definition_version
    assert definition_version_a == definition_version_b

    # feature_version should also be SAME (field provenance unchanged by project)
    feature_version_a = graph_a.get_feature_version(FeatureKey(["test", "feature"]))
    feature_version_b = graph_b.get_feature_version(FeatureKey(["test", "feature"]))
    assert feature_version_a == feature_version_b

    # Snapshot for verification
    assert {
        "definition_version_a": definition_version_a,
        "definition_version_b": definition_version_b,
        "feature_version_a": feature_version_a,
        "feature_version_b": feature_version_b,
        "definition_versions_same": definition_version_a == definition_version_b,
        "feature_versions_same": feature_version_a == feature_version_b,
    } == snapshot


def test_feature_version_unchanged_by_project(snapshot: SnapshotAssertion) -> None:
    """Test that feature_version does NOT change when only project changes.

    This is critical: feature_version is used for field provenance and should
    only change when the computational definition changes, not metadata like project.
    """
    # Define identical feature specs in two projects (same class name for identical schema)
    graph_1 = FeatureGraph()
    graph_2 = FeatureGraph()

    with graph_1.use():

        class DataFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["data", "feature"]),
                fields=[
                    FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                    FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                ],
            ),
        ):
            pass

    with graph_2.use():

        class DataFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["data", "feature"]),
                fields=[
                    FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                    FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                ],
            ),
        ):
            pass

    # feature_version should be IDENTICAL (field provenance unchanged)
    version_1 = graph_1.get_feature_version(FeatureKey(["data", "feature"]))
    version_2 = graph_2.get_feature_version(FeatureKey(["data", "feature"]))

    assert version_1 == version_2
    assert version_1 == snapshot


def test_definition_version_deterministic(snapshot: SnapshotAssertion) -> None:
    """Test that definition version is deterministic."""
    graph = FeatureGraph()

    with graph.use():

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "deterministic"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    definition = graph.get_feature_definition(FeatureKey(["test", "deterministic"]))

    # feature_definition_version should be deterministic
    definition_version = definition.feature_definition_version

    assert len(definition_version) == 8  # SHA256 hex
    assert definition_version == snapshot


def test_feature_with_deps_same_definition_version(snapshot: SnapshotAssertion) -> None:
    """Test feature_version and definition_version stay same across identical graphs."""
    # Project A: upstream and downstream
    graph_a = FeatureGraph()

    with graph_a.use():

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # Project B: same features
    graph_b = FeatureGraph()

    with graph_b.use():

        class Upstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class Downstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # feature_version should be same across graphs
    assert graph_a.get_feature_version(FeatureKey(["upstream"])) == graph_b.get_feature_version(
        FeatureKey(["upstream"])
    )
    assert graph_a.get_feature_version(FeatureKey(["downstream"])) == graph_b.get_feature_version(
        FeatureKey(["downstream"])
    )

    # definition_version should also be same
    def_a_up = graph_a.get_feature_definition(FeatureKey(["upstream"]))
    def_b_up = graph_b.get_feature_definition(FeatureKey(["upstream"]))
    def_a_down = graph_a.get_feature_definition(FeatureKey(["downstream"]))
    def_b_down = graph_b.get_feature_definition(FeatureKey(["downstream"]))

    assert def_a_up.feature_definition_version == def_b_up.feature_definition_version
    assert def_a_down.feature_definition_version == def_b_down.feature_definition_version

    # Snapshot both
    assert {
        "upstream_feature_version": graph_a.get_feature_version(FeatureKey(["upstream"])),
        "downstream_feature_version": graph_a.get_feature_version(FeatureKey(["downstream"])),
        "upstream_definition_version": def_a_up.feature_definition_version,
        "downstream_definition_version": def_a_down.feature_definition_version,
    } == snapshot


def test_project_in_graph_snapshot(snapshot: SnapshotAssertion) -> None:
    """Test that project information is preserved in graph snapshots."""
    graph = FeatureGraph()

    with graph.use():

        class Feature1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature1"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class Feature2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature2"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    # Get snapshot
    snapshot_data = graph.to_snapshot()

    # Both features should have project in snapshot (from module name)
    assert "project" in snapshot_data["feature1"]
    assert "project" in snapshot_data["feature2"]

    # Both should have definition_version
    assert "metaxy_definition_version" in snapshot_data["feature1"]
    assert "metaxy_definition_version" in snapshot_data["feature2"]

    # Snapshot the graph structure
    assert {
        "feature_keys": sorted(snapshot_data.keys()),
        "has_project_feature1": "project" in snapshot_data["feature1"],
        "has_project_feature2": "project" in snapshot_data["feature2"],
        "has_definition_version_feature1": "metaxy_definition_version" in snapshot_data["feature1"],
        "has_definition_version_feature2": "metaxy_definition_version" in snapshot_data["feature2"],
    } == snapshot


def test_provenance_by_field_unchanged_across_graphs(
    snapshot: SnapshotAssertion,
) -> None:
    """Test that provenance_by_field is also unchanged across identical graphs.

    This verifies the entire field provenance pipeline is deterministic.
    """
    graph_a = FeatureGraph()
    graph_b = FeatureGraph()

    with graph_a.use():

        class MultiFieldFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["multi", "field"]),
                fields=[
                    FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                    FieldSpec(key=FieldKey(["field2"]), code_version="2"),
                ],
            ),
        ):
            pass

    with graph_b.use():

        class MultiFieldFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["multi", "field"]),
                fields=[
                    FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                    FieldSpec(key=FieldKey(["field2"]), code_version="2"),
                ],
            ),
        ):
            pass

    # provenance_by_field should be identical across graphs
    feature_key = FeatureKey(["multi", "field"])
    provenance_a = graph_a.get_feature_version_by_field(feature_key)
    provenance_b = graph_b.get_feature_version_by_field(feature_key)

    assert provenance_a == provenance_b
    assert provenance_a == snapshot
