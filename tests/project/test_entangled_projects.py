"""Tests for entangled multi-project setups.

Tests scenarios where projects A and B have features depending on each other,
creating a chicken-and-egg dependency that requires project-scoped snapshot versions.
"""

from __future__ import annotations

from pathlib import Path

from metaxy_testing.models import SampleFeatureSpec

from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureDefinition, FeatureGraph


def test_project_can_push_with_unresolved_external_dependency(tmp_path: Path) -> None:
    """Test that a project can push to a fresh store with unresolved external dependency.

    This enables the chicken-and-egg scenario where project A depends on B
    and B depends on A - neither can load the other's definitions from a fresh store.
    """
    # Project A has a feature that depends on an external feature from project B
    graph_a = FeatureGraph()
    with graph_a.use():
        # Add external placeholder for project B's feature
        external_b = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_b", "upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
            feature_schema={"type": "object"},
            project="project_b",
        )
        graph_a.add_feature_definition(external_b)

        # Project A's own feature depending on external B
        class ProjectAFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_a", "downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["project_b", "upstream"]))],
                fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
            ),
        ):
            __metaxy_project__ = "project_a"

        # Should be able to create snapshot even with unresolved external
        snapshot = graph_a.to_snapshot()

        # Only project A's feature should be in snapshot (external excluded)
        assert "project_a/downstream" in snapshot
        assert "project_b/upstream" not in snapshot

        # Should be able to push to fresh store
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            result = storage.push_graph_snapshot(project="project_a")

            # Push should succeed
            assert result.snapshot_version is not None
            assert not result.already_pushed


def test_entangled_projects_can_both_push_to_same_store(tmp_path: Path) -> None:
    """Test that two entangled projects can both push to the same store."""
    # Project A pushes first with external dep on B
    graph_a = FeatureGraph()
    with graph_a.use():
        external_b = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_b", "feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
            feature_schema={"type": "object"},
            project="project_b",
        )
        graph_a.add_feature_definition(external_b)

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_a", "feature"]),
                deps=[FeatureDep(feature=FeatureKey(["project_b", "feature"]))],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            __metaxy_project__ = "project_a"

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            result_a = storage.push_graph_snapshot(project="project_a")
            assert not result_a.already_pushed

    # Project B pushes second with external dep on A
    graph_b = FeatureGraph()
    with graph_b.use():
        external_a = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_a", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]))],
            ),
            feature_schema={"type": "object"},
            project="project_a",
        )
        graph_b.add_feature_definition(external_a)

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_b", "feature"]),
                deps=[FeatureDep(feature=FeatureKey(["project_a", "feature"]))],
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            __metaxy_project__ = "project_b"

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)
            result_b = storage.push_graph_snapshot(project="project_b")
            assert not result_b.already_pushed

    # Both projects should now be in the store
    with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        storage = SystemTableStorage(store)
        features = storage.read_features(current=False, snapshot_version=result_a.snapshot_version)
        assert "project_a/feature" in features["feature_key"].to_list()

        features = storage.read_features(current=False, snapshot_version=result_b.snapshot_version)
        assert "project_b/feature" in features["feature_key"].to_list()


def test_project_snapshot_versions_are_independent(tmp_path: Path) -> None:
    """Test that a project's snapshot version doesn't change when external features change.

    This test uses FeatureDefinition directly (not BaseFeature classes) to avoid
    differences in Pydantic schema generation from different class names.
    """
    from metaxy.models.feature_spec import FeatureSpec

    # Create the FeatureSpec for project A (identical in both graphs)
    feature_a_spec = FeatureSpec(
        key=FeatureKey(["project_a", "feature"]),
        deps=[FeatureDep(feature=FeatureKey(["project_b", "feature"]))],
        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        id_columns=["sample_uid"],
    )
    feature_a_schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    # Graph 1: external B has code_version="1"
    graph1 = FeatureGraph()
    external_b_v1 = FeatureDefinition.external(
        spec=SampleFeatureSpec(
            key=FeatureKey(["project_b", "feature"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
        feature_schema={"type": "object"},
        project="project_b",
    )
    graph1.add_feature_definition(external_b_v1)

    feature_a_def_v1 = FeatureDefinition(
        spec=feature_a_spec,
        feature_schema=feature_a_schema,
        feature_class_path="test.FeatureA",
        project="project_a",
    )
    graph1.add_feature_definition(feature_a_def_v1)

    snapshot_v1 = graph1.get_project_snapshot_version("project_a")

    # Graph 2: external B has code_version="2" (different!)
    graph2 = FeatureGraph()
    external_b_v2 = FeatureDefinition.external(
        spec=SampleFeatureSpec(
            key=FeatureKey(["project_b", "feature"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="2")],  # Different!
        ),
        feature_schema={"type": "object"},
        project="project_b",
    )
    graph2.add_feature_definition(external_b_v2)

    # Same FeatureDefinition for project A
    feature_a_def_v2 = FeatureDefinition(
        spec=feature_a_spec,
        feature_schema=feature_a_schema,
        feature_class_path="test.FeatureA",
        project="project_a",
    )
    graph2.add_feature_definition(feature_a_def_v2)

    snapshot_v2 = graph2.get_project_snapshot_version("project_a")

    assert snapshot_v1 == snapshot_v2, (
        f"Project snapshot version should not change when external features change. v1={snapshot_v1}, v2={snapshot_v2}"
    )


def test_project_snapshot_version_changes_when_own_feature_changes(tmp_path: Path) -> None:
    """Test that a project's snapshot version changes when its own features change."""
    # Create project A v1
    graph1 = FeatureGraph()
    with graph1.use():

        class FeatureAv1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_a", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            __metaxy_project__ = "project_a"

        snapshot_v1 = graph1.get_project_snapshot_version("project_a")

    # Create project A v2 with different code_version
    graph2 = FeatureGraph()
    with graph2.use():

        class FeatureAv2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_a", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="2")],  # Different!
            ),
        ):
            __metaxy_project__ = "project_a"

        snapshot_v2 = graph2.get_project_snapshot_version("project_a")

    assert snapshot_v1 != snapshot_v2, (
        f"Project snapshot version should change when own features change. v1={snapshot_v1}, v2={snapshot_v2}"
    )


def test_global_snapshot_version_still_includes_all_features(tmp_path: Path) -> None:
    """Test that the global snapshot_version still includes all non-external features."""
    graph = FeatureGraph()
    with graph.use():

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_a", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            __metaxy_project__ = "project_a"

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_b", "feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            __metaxy_project__ = "project_b"

        global_snapshot = graph.snapshot_version
        project_a_snapshot = graph.get_project_snapshot_version("project_a")
        project_b_snapshot = graph.get_project_snapshot_version("project_b")

        # Global should be different from project-specific versions
        assert global_snapshot != project_a_snapshot
        assert global_snapshot != project_b_snapshot

        # Project-specific versions should also be different from each other
        assert project_a_snapshot != project_b_snapshot


def test_external_features_excluded_from_project_snapshot(tmp_path: Path) -> None:
    """Test that external features are excluded from project snapshot calculations."""
    graph = FeatureGraph()
    with graph.use():
        # Add external feature to project A
        external_b = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_b", "external"]),
                fields=[FieldSpec(key=FieldKey(["data"]))],
            ),
            feature_schema={"type": "object"},
            project="project_b",
        )
        graph.add_feature_definition(external_b)

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_a", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            __metaxy_project__ = "project_a"

        # Project B's snapshot should be empty (only has external features)
        assert graph.get_project_snapshot_version("project_b") == "empty"

        # Project A's snapshot should only include its own feature
        assert graph.get_project_snapshot_version("project_a") != "empty"


def test_push_idempotent_for_project_snapshot(tmp_path: Path) -> None:
    """Test that pushing the same project snapshot twice is idempotent."""
    graph = FeatureGraph()
    with graph.use():

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["project_a", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            __metaxy_project__ = "project_a"

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            storage = SystemTableStorage(store)

            # First push
            result1 = storage.push_graph_snapshot(project="project_a")
            assert not result1.already_pushed

            # Second push - same snapshot
            result2 = storage.push_graph_snapshot(project="project_a")
            assert result2.already_pushed
            assert result1.snapshot_version == result2.snapshot_version
