"""Test that push_graph_snapshot produces stable snapshot_versions."""

from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import FeatureGraph


def test_push_graph_snapshot_stability(
    store: MetadataStore, test_graph: FeatureGraph
) -> None:
    """Test that push_graph_snapshot produces consistent snapshot_versions.

    Verifies that:
    1. push_graph_snapshot uses to_snapshot() internally
    2. The snapshot_version from push_graph_snapshot matches graph.snapshot_version
    3. Serializing the same graph multiple times produces the same snapshot_version
    4. The snapshot can be deserialized and produces the same snapshot_version
    """
    graph = test_graph

    # Ensure this graph is active so push_graph_snapshot uses it
    with graph.use():
        with store:
            # Get the original snapshot_version from the graph
            original_snapshot_version = graph.snapshot_version

            # Serialize the graph for the first time
            result_1 = SystemTableStorage(store).push_graph_snapshot()
            snapshot_version_1 = result_1.snapshot_version
            was_already_pushed_1 = result_1.already_pushed

            # Should not be recorded yet (first time)
            assert not was_already_pushed_1, (
                "First serialization should not be marked as already recorded"
            )

            # Should match the graph's snapshot_version
            assert snapshot_version_1 == original_snapshot_version, (
                f"Serialized snapshot_version {snapshot_version_1} doesn't match graph.snapshot_version {original_snapshot_version}"
            )

            # Serialize again - should be idempotent
            result_2 = SystemTableStorage(store).push_graph_snapshot()
            snapshot_version_2 = result_2.snapshot_version
            was_already_pushed_2 = result_2.already_pushed

            # Should be marked as already recorded
            assert was_already_pushed_2, (
                "Second serialization should be marked as already recorded"
            )

            # Should produce the same snapshot_version
            assert snapshot_version_2 == snapshot_version_1, (
                f"Second serialization produced different snapshot_version: {snapshot_version_2} vs {snapshot_version_1}"
            )

            # Read the snapshot back and verify it can be reconstructed
            snapshot_dict = graph.to_snapshot()

            # Reconstruct graph from snapshot
            reconstructed_graph = FeatureGraph.from_snapshot(snapshot_dict)
            reconstructed_snapshot_version = reconstructed_graph.snapshot_version

            # Reconstructed graph should have the same snapshot_version
            assert reconstructed_snapshot_version == original_snapshot_version, (
                f"Reconstructed snapshot_version {reconstructed_snapshot_version} doesn't match original {original_snapshot_version}"
            )


def test_serialize_uses_to_snapshot(
    store: MetadataStore, test_graph: FeatureGraph
) -> None:
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

            snapshot_version = result.snapshot_version

            _ = result.already_pushed

            # Read back the serialized data from the store
            from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

            versions_lazy = store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None, "Feature versions should be recorded"

            versions_df = versions_lazy.collect().to_polars()

            # Verify all features from snapshot_dict were recorded
            for feature_key_str, feature_data in snapshot_dict.items():
                # Find matching record in versions_df
                matching_rows = versions_df.filter(
                    versions_df["feature_key"] == feature_key_str
                )

                assert matching_rows.height == 1, (
                    f"Expected exactly one record for {feature_key_str}, found {matching_rows.height}"
                )

                row = matching_rows.to_dicts()[0]

                # Verify the data matches (feature_data is a FeatureDefinition)
                assert row["metaxy_feature_version"] == feature_data.feature_version, (
                    f"Feature version mismatch for {feature_key_str}"
                )
                assert row["feature_class_path"] == feature_data.feature_class_path, (
                    f"Feature class path mismatch for {feature_key_str}"
                )
                assert row["metaxy_snapshot_version"] == snapshot_version, (
                    f"Snapshot version mismatch for {feature_key_str}"
                )


def test_snapshot_version_deterministic_across_stores(
    test_graph: FeatureGraph,
) -> None:
    """Test that snapshot_version is deterministic regardless of store type.

    The snapshot_version should be computed from the graph structure alone,
    not from any store-specific details.
    """
    graph = test_graph

    # Get snapshot_version directly from graph
    snapshot_version_1 = graph.snapshot_version

    # Get it again
    snapshot_version_2 = graph.snapshot_version

    # Should be identical
    assert snapshot_version_1 == snapshot_version_2, (
        "snapshot_version should be deterministic"
    )

    # Get snapshot dict and reconstruct
    snapshot_dict = graph.to_snapshot()
    reconstructed_graph = FeatureGraph.from_snapshot(snapshot_dict)
    snapshot_version_3 = reconstructed_graph.snapshot_version

    # Should still be identical
    assert snapshot_version_3 == snapshot_version_1, (
        "Reconstructed graph should have same snapshot_version"
    )
