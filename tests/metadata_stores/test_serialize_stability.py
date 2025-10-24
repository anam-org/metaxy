"""Test that record_feature_graph_snapshot produces stable snapshot_ids."""

from typing import Any

from metaxy.metadata_store import MetadataStore
from metaxy.models.feature import FeatureGraph


def test_record_feature_graph_snapshot_stability(
    persistent_store: MetadataStore, test_graph: tuple[FeatureGraph, dict[str, Any]]
) -> None:
    """Test that record_feature_graph_snapshot produces consistent snapshot_ids.

    Verifies that:
    1. record_feature_graph_snapshot uses to_snapshot() internally
    2. The snapshot_id from record_feature_graph_snapshot matches graph.snapshot_id
    3. Serializing the same graph multiple times produces the same snapshot_id
    4. The snapshot can be deserialized and produces the same snapshot_id
    """
    graph, features = test_graph

    # Ensure this graph is active so record_feature_graph_snapshot uses it
    with graph.use():
        with persistent_store:
            # Get the original snapshot_id from the graph
            original_snapshot_id = graph.snapshot_id

            # Serialize the graph for the first time
            snapshot_id_1, was_already_recorded_1 = (
                persistent_store.record_feature_graph_snapshot()
            )

            # Should not be recorded yet (first time)
            assert not was_already_recorded_1, (
                "First serialization should not be marked as already recorded"
            )

            # Should match the graph's snapshot_id
            assert snapshot_id_1 == original_snapshot_id, (
                f"Serialized snapshot_id {snapshot_id_1} doesn't match graph.snapshot_id {original_snapshot_id}"
            )

            # Serialize again - should be idempotent
            snapshot_id_2, was_already_recorded_2 = (
                persistent_store.record_feature_graph_snapshot()
            )

            # Should be marked as already recorded
            assert was_already_recorded_2, (
                "Second serialization should be marked as already recorded"
            )

            # Should produce the same snapshot_id
            assert snapshot_id_2 == snapshot_id_1, (
                f"Second serialization produced different snapshot_id: {snapshot_id_2} vs {snapshot_id_1}"
            )

            # Read the snapshot back and verify it can be reconstructed
            snapshot_dict = graph.to_snapshot()

            # Reconstruct graph from snapshot
            reconstructed_graph = FeatureGraph.from_snapshot(snapshot_dict)
            reconstructed_snapshot_id = reconstructed_graph.snapshot_id

            # Reconstructed graph should have the same snapshot_id
            assert reconstructed_snapshot_id == original_snapshot_id, (
                f"Reconstructed snapshot_id {reconstructed_snapshot_id} doesn't match original {original_snapshot_id}"
            )


def test_serialize_uses_to_snapshot(
    persistent_store: MetadataStore, test_graph: tuple[FeatureGraph, dict[str, Any]]
) -> None:
    """Test that record_feature_graph_snapshot correctly uses to_snapshot().

    Verifies that the serialization format matches what to_snapshot() produces.
    """
    graph, features = test_graph

    # Get snapshot dict from to_snapshot()
    snapshot_dict = graph.to_snapshot()

    # Ensure this graph is active so record_feature_graph_snapshot uses it
    with graph.use():
        with persistent_store:
            # Serialize to the store
            snapshot_id, _ = persistent_store.record_feature_graph_snapshot()

            # Read back the serialized data from the store
            from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

            versions_lazy = persistent_store._read_metadata_native(FEATURE_VERSIONS_KEY)
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

                # Verify the data matches
                assert row["feature_version"] == feature_data["feature_version"], (
                    f"Feature version mismatch for {feature_key_str}"
                )
                assert (
                    row["feature_class_path"] == feature_data["feature_class_path"]
                ), f"Feature class path mismatch for {feature_key_str}"
                assert row["snapshot_id"] == snapshot_id, (
                    f"Snapshot ID mismatch for {feature_key_str}"
                )


def test_snapshot_id_deterministic_across_stores(
    test_graph: tuple[FeatureGraph, dict[str, Any]],
) -> None:
    """Test that snapshot_id is deterministic regardless of store type.

    The snapshot_id should be computed from the graph structure alone,
    not from any store-specific details.
    """
    graph, _ = test_graph

    # Get snapshot_id directly from graph
    snapshot_id_1 = graph.snapshot_id

    # Get it again
    snapshot_id_2 = graph.snapshot_id

    # Should be identical
    assert snapshot_id_1 == snapshot_id_2, "snapshot_id should be deterministic"

    # Get snapshot dict and reconstruct
    snapshot_dict = graph.to_snapshot()
    reconstructed_graph = FeatureGraph.from_snapshot(snapshot_dict)
    snapshot_id_3 = reconstructed_graph.snapshot_id

    # Should still be identical
    assert snapshot_id_3 == snapshot_id_1, (
        "Reconstructed graph should have same snapshot_id"
    )
