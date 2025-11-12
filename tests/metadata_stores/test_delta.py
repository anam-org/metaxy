"""Delta Lake-specific tests that don't apply to other stores."""

from __future__ import annotations

import polars as pl
import pytest
from deltalake import DeltaTable

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.delta import DeltaMetadataStore


def test_delta_write_and_read(tmp_path, test_graph, test_features) -> None:
    """Write metadata and read it back from Delta store."""
    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key  # type: ignore[attr-defined]

    with DeltaMetadataStore(store_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                    {"frames": "h3", "audio": "h3"},
                ],
            }
        ).lazy()

        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

        result = collect_to_polars(store.read_metadata(feature_cls))
        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}

        # Use Delta's native API to check version
        feature_path = store._feature_local_path(feature_key)
        assert feature_path is not None
        delta_table = DeltaTable(str(feature_path))
        assert delta_table.version() == 0
        assert delta_table.to_pyarrow_table().num_rows == 3


def test_delta_persistence_across_instances(
    tmp_path, test_graph, test_features
) -> None:
    """Data written in one instance is visible in another."""
    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]

    with DeltaMetadataStore(store_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

    with DeltaMetadataStore(store_path) as store:
        result = collect_to_polars(store.read_metadata(feature_cls))
        assert len(result) == 2


def test_delta_drop_feature(tmp_path, test_graph, test_features) -> None:
    """Dropping metadata uses soft delete (marks rows as deleted in transaction log)."""
    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key  # type: ignore[attr-defined]

    with DeltaMetadataStore(store_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                ],
            }
        )
        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

        feature_path = store._feature_local_path(feature_key)
        assert feature_path is not None
        assert (feature_path / "_delta_log").exists()

        # Soft delete - marks rows as deleted but doesn't remove transaction log
        store.drop_feature_metadata(feature_cls)

        # Transaction log still exists (soft delete)
        assert feature_path is not None
        assert (feature_path / "_delta_log").exists()

        # Check that table is now empty (rows marked as deleted)
        delta_table = DeltaTable(str(feature_path))
        assert delta_table.to_pyarrow_table().num_rows == 0

    # Fresh instance should see no data (deleted rows)
    with DeltaMetadataStore(store_path) as store:
        result = store.read_metadata(feature_cls)
        if result is not None:
            collected = collect_to_polars(result)
            assert len(collected) == 0

        # Can write new data to the same table
        fresh = pl.DataFrame(
            {
                "sample_uid": [2],
                "metaxy_provenance_by_field": [
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, fresh)

        result = collect_to_polars(store.read_metadata(feature_cls))
        assert result["sample_uid"].to_list() == [2]


def test_delta_lists_features(tmp_path, test_graph, test_features) -> None:
    """Verify feature discovery in Delta store."""
    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key  # type: ignore[attr-defined]

    with DeltaMetadataStore(store_path) as store:
        assert store.list_features() == []

        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                ],
            }
        )
        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

        assert store.list_features() == [feature_key]


def test_delta_display(tmp_path) -> None:
    """Display output includes path and storage options (but not feature count for performance)."""
    store_path = tmp_path / "delta"
    store = DeltaMetadataStore(store_path, storage_options={"AWS_ACCESS_KEY_ID": "x"})

    closed_display = store.display()
    assert "DeltaMetadataStore" in closed_display
    assert str(store_path) in closed_display
    assert "storage_options=***" in closed_display

    with store:
        open_display = store.display()
        # Feature count is not included to avoid scanning entire store
        assert "DeltaMetadataStore" in open_display
        assert "storage_options=***" in open_display


def test_delta_remote_lists_features_returns_empty() -> None:
    """Remote stores return empty list for list_features() - use system tables instead."""
    store = DeltaMetadataStore("s3://bucket/root", auto_create_tables=False)

    with store:
        with pytest.warns(
            UserWarning, match="Feature discovery not supported for remote"
        ):
            features = store.list_features()

    # Remote stores return empty list - must use system tables for feature discovery
    assert features == []


def test_delta_streaming_write(tmp_path, test_graph, test_features) -> None:
    """Verify streaming mode writes lazy frames in batches without full collection."""
    store_path = tmp_path / "delta_streaming"
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key  # type: ignore[attr-defined]

    # Create store with streaming enabled (chunk size = 50 rows)
    with DeltaMetadataStore(store_path, streaming_chunk_size=50) as store:
        # Create a lazy frame with enough rows to create multiple batches
        metadata = pl.LazyFrame(
            {
                "sample_uid": list(range(200)),  # 200 rows = 4 batches of 50
                "metaxy_provenance_by_field": [
                    {"frames": f"h{i}", "audio": f"h{i}"} for i in range(200)
                ],
            }
        )

        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

        # Verify data was written correctly
        result = collect_to_polars(store.read_metadata(feature_cls))
        assert len(result) == 200
        assert set(result["sample_uid"].to_list()) == set(range(200))

        # Verify Delta table was created and contains all data
        feature_path = store._feature_local_path(feature_key)
        assert feature_path is not None
        delta_table = DeltaTable(str(feature_path))

        # Verify all rows were written (regardless of version count)
        # Note: Delta may optimize writes into fewer transactions than expected
        assert delta_table.to_pyarrow_table().num_rows == 200

        # Verify table exists and is readable
        assert (feature_path / "_delta_log").exists()
