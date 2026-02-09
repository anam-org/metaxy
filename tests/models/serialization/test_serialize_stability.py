"""Test that push_graph_snapshot produces stable project_versions."""

from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureGraph


def test_push_graph_snapshot_stability(store: MetadataStore, test_graph: FeatureGraph) -> None:
    """Test that push_graph_snapshot produces consistent project_versions.

    Verifies that:
    1. push_graph_snapshot uses to_snapshot() internally
    2. The project_version from push_graph_snapshot matches graph.get_project_version()
    3. Serializing the same graph multiple times produces the same project_version
    4. The snapshot can be deserialized and produces the same project_version
    """
    graph = test_graph

    # Ensure this graph is active so push_graph_snapshot uses it
    with graph.use():
        with store:
            # Infer project from graph (all features share same project in test_graph)
            snapshot_dict = graph.to_snapshot()
            project = next(iter(snapshot_dict.values()))["project"]

            # Get the project-scoped project_version from the graph
            original_project_version = graph.get_project_version(project)

            # Serialize the graph for the first time
            result_1 = SystemTableStorage(store).push_graph_snapshot()
            project_version_1 = result_1.project_version
            was_already_pushed_1 = result_1.already_pushed

            # Should not be recorded yet (first time)
            assert not was_already_pushed_1, "First serialization should not be marked as already recorded"

            # Should match the graph's project project_version
            assert project_version_1 == original_project_version, (
                f"Serialized project_version {project_version_1} doesn't match graph.get_project_version() {original_project_version}"
            )

            # Serialize again - should be idempotent
            result_2 = SystemTableStorage(store).push_graph_snapshot()
            project_version_2 = result_2.project_version
            was_already_pushed_2 = result_2.already_pushed

            # Should be marked as already recorded
            assert was_already_pushed_2, "Second serialization should be marked as already recorded"

            # Should produce the same project_version
            assert project_version_2 == project_version_1, (
                f"Second serialization produced different project_version: {project_version_2} vs {project_version_1}"
            )

            # Read the snapshot back and verify it can be reconstructed
            # Reconstruct graph from snapshot
            reconstructed_graph = FeatureGraph.from_snapshot(snapshot_dict)
            reconstructed_project_version = reconstructed_graph.get_project_version(project)

            # Reconstructed graph should have the same project project_version
            assert reconstructed_project_version == original_project_version, (
                f"Reconstructed project_version {reconstructed_project_version} doesn't match original {original_project_version}"
            )


def test_serialize_uses_to_snapshot(store: MetadataStore, test_graph: FeatureGraph) -> None:
    """Test that push_graph_snapshot correctly uses to_snapshot().

    Verifies that the serialization format matches what to_snapshot() produces.
    """
    graph = test_graph

    # Get snapshot dict from to_snapshot()
    snapshot_dict = graph.to_snapshot()

    # Ensure this graph is active so push_graph_snapshot uses it
    with graph.use():
        with store:
            # Serialize to the store
            result = SystemTableStorage(store).push_graph_snapshot()

            project_version = result.project_version

            _ = result.already_pushed

            # Read back the serialized data from the store
            from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

            versions_lazy = store._read_feature(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None, "Feature versions should be recorded"

            versions_df = versions_lazy.collect().to_polars()

            # Verify all features from snapshot_dict were recorded
            for feature_key_str, feature_data in snapshot_dict.items():
                # Find matching record in versions_df
                matching_rows = versions_df.filter(versions_df["feature_key"] == feature_key_str)

                assert matching_rows.height == 1, (
                    f"Expected exactly one record for {feature_key_str}, found {matching_rows.height}"
                )

                row = matching_rows.to_dicts()[0]

                # Verify the data matches
                assert row["metaxy_feature_version"] == feature_data["metaxy_feature_version"], (
                    f"Feature version mismatch for {feature_key_str}"
                )
                assert row["feature_class_path"] == feature_data["feature_class_path"], (
                    f"Feature class path mismatch for {feature_key_str}"
                )
                assert row["metaxy_project_version"] == project_version, (
                    f"Snapshot version mismatch for {feature_key_str}"
                )


def test_project_version_deterministic_across_stores(
    test_graph: FeatureGraph,
) -> None:
    """Test that project_version is deterministic regardless of store type.

    The project_version should be computed from the graph structure alone,
    not from any store-specific details.
    """
    graph = test_graph

    # Get project_version directly from graph
    project_version_1 = graph.project_version

    # Get it again
    project_version_2 = graph.project_version

    # Should be identical
    assert project_version_1 == project_version_2, "project_version should be deterministic"

    # Get snapshot dict and reconstruct
    snapshot_dict = graph.to_snapshot()
    reconstructed_graph = FeatureGraph.from_snapshot(snapshot_dict)
    project_version_3 = reconstructed_graph.project_version

    # Should still be identical
    assert project_version_3 == project_version_1, "Reconstructed graph should have same project_version"
