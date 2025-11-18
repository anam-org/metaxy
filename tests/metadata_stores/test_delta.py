"""Delta Lake-specific tests that don't apply to other stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
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
        store.write_metadata(feature_cls, fresh)

        result = collect_to_polars(store.read_metadata(feature_cls))
        assert result["sample_uid"].to_list() == [2]


def test_delta_lists_features(tmp_path, test_graph, test_features) -> None:
    """Verify feature discovery in Delta store."""
    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key  # type: ignore[attr-defined]

    with DeltaMetadataStore(store_path) as store:
        assert store._list_features_local() == []

        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                ],
            }
        )
        store.write_metadata(feature_cls, metadata)

        assert store._list_features_local() == [feature_key]


def test_delta_nested_layout_creates_directories(
    tmp_path, test_graph, test_features
) -> None:
    """Nested layout stores feature tables in per-part directories."""
    store_path = tmp_path / "delta_nested"
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key  # type: ignore[attr-defined]

    with DeltaMetadataStore(store_path, layout="nested") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [42],
                "metaxy_provenance_by_field": [
                    {"frames": "h42", "audio": "h42"},
                ],
            }
        )
        store.write_metadata(feature_cls, metadata)

        feature_dir = store._feature_local_path(feature_key)
        assert feature_dir is not None
        assert feature_dir.exists()
        assert store._local_root_path is not None
        assert feature_dir.relative_to(store._local_root_path) == Path(
            "/".join(feature_key.parts)
        )
        assert store._list_features_local() == [feature_key]


def test_delta_display(tmp_path) -> None:
    """Display output includes path and storage options (but not feature count for performance)."""
    store_path = tmp_path / "delta"
    store = DeltaMetadataStore(store_path, storage_options={"AWS_ACCESS_KEY_ID": "x"})

    closed_display = store.display()
    assert "DeltaMetadataStore" in closed_display
    assert str(store_path) in closed_display
    assert "storage_options=***" in closed_display
    assert "layout=flat" in closed_display

    with store:
        open_display = store.display()
        # Feature count is not included to avoid scanning entire store
        assert "DeltaMetadataStore" in open_display
        assert "storage_options=***" in open_display
        assert "layout=flat" in open_display


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


# ===== S3 and Object Store Configuration Tests =====


def test_delta_s3_store_initialization() -> None:
    """Test S3 store initialization with storage options."""
    store_path = "s3://my-bucket/delta_store"
    storage_options = {
        "AWS_REGION": "us-west-2",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
    }

    store = DeltaMetadataStore(store_path, storage_options=storage_options)

    # Verify store configuration
    assert store._root_uri == "s3://my-bucket/delta_store"
    assert store._is_remote is True
    assert store._local_root_path is None
    assert store.storage_options == storage_options


def test_delta_s3_store_display_masks_credentials() -> None:
    """Display output masks storage options to avoid exposing credentials."""
    store_path = "s3://my-bucket/delta_store"
    storage_options = {
        "AWS_ACCESS_KEY_ID": "secret_key",
        "AWS_SECRET_ACCESS_KEY": "super_secret",
    }

    store = DeltaMetadataStore(store_path, storage_options=storage_options)

    display = store.display()
    assert "storage_options=***" in display
    assert "secret_key" not in display
    assert "super_secret" not in display
    assert store_path in display


def test_delta_feature_uri_for_s3() -> None:
    """Test feature URI generation for S3 paths."""
    from metaxy import FeatureKey

    store_path = "s3://my-bucket/delta_store"
    store = DeltaMetadataStore(store_path, auto_create_tables=False)

    feature_key = FeatureKey(["test_stores", "my_feature"])
    uri = store._feature_uri(feature_key)

    assert uri == "s3://my-bucket/delta_store/test_stores__my_feature"


def test_delta_s3_path_detection() -> None:
    """Test that S3 paths are correctly detected as remote."""
    s3_store = DeltaMetadataStore("s3://bucket/path", auto_create_tables=False)
    assert s3_store._is_remote is True
    assert s3_store._local_root_path is None

    # Test with different cloud providers
    azure_store = DeltaMetadataStore(
        "abfss://container@account.dfs.core.windows.net/path",
        auto_create_tables=False,
    )
    assert azure_store._is_remote is True

    gcs_store = DeltaMetadataStore("gs://bucket/path", auto_create_tables=False)
    assert gcs_store._is_remote is True


def test_delta_local_path_detection(tmp_path) -> None:
    """Test that local paths are correctly detected."""
    local_store = DeltaMetadataStore(tmp_path, auto_create_tables=False)
    assert local_store._is_remote is False
    assert local_store._local_root_path == tmp_path

    # Test file:// URL
    file_url_store = DeltaMetadataStore(f"file://{tmp_path}", auto_create_tables=False)
    assert file_url_store._is_remote is False

    # Test local:// URL
    local_uri_store = DeltaMetadataStore(
        f"local://{tmp_path}", auto_create_tables=False
    )
    assert local_uri_store._is_remote is False
    assert local_uri_store._local_root_path == tmp_path.resolve()


def test_delta_storage_options_passed_through(test_graph, test_features) -> None:
    """Test that storage options are properly passed to Delta operations."""
    from unittest.mock import patch

    store_path = "s3://test-bucket/delta"
    storage_options = {
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
    }

    feature_cls = test_features["UpstreamFeatureA"]

    store = DeltaMetadataStore(
        store_path, storage_options=storage_options, auto_create_tables=False
    )

    # Verify storage_options are stored
    assert store.storage_options == storage_options

    # Test _table_exists passes storage options
    with patch("deltalake.DeltaTable.is_deltatable") as mock_is_deltatable:
        mock_is_deltatable.return_value = False
        feature_key = feature_cls.spec().key  # type: ignore[attr-defined]
        table_uri = store._feature_uri(feature_key)

        result = store._table_exists(table_uri)

        # Verify is_deltatable was called with storage_options (as keyword arg)
        mock_is_deltatable.assert_called_once_with(
            table_uri, storage_options=storage_options
        )
        assert result is False


def test_delta_custom_delta_write_options_used(
    tmp_path, test_graph, test_features, monkeypatch
) -> None:
    """delta_write_options override the default schema_mode=merge."""
    import deltalake

    store_path = tmp_path / "delta_opts"
    feature_cls = test_features["UpstreamFeatureA"]
    recorded: dict[str, Any] = {}

    def fake_write_deltalake(table_uri, data, **kwargs):
        recorded["kwargs"] = kwargs
        recorded["table_uri"] = table_uri

    monkeypatch.setattr(deltalake, "write_deltalake", fake_write_deltalake)

    with DeltaMetadataStore(
        store_path,
        delta_write_options={"schema_mode": "ignore_nullable", "max_workers": 2},
        auto_create_tables=True,
    ) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [7],
                "metaxy_provenance_by_field": [
                    {"frames": "h7", "audio": "h7"},
                ],
            }
        )
        store.write_metadata(feature_cls, metadata)

    # Check that custom options override defaults
    assert recorded["kwargs"]["mode"] == "append"
    assert recorded["kwargs"]["schema_mode"] == "ignore_nullable"
    assert recorded["kwargs"]["max_workers"] == 2
    assert str(store_path) in recorded["table_uri"]


def test_delta_streaming_with_s3_path() -> None:
    """Test that streaming mode can be configured for S3 paths."""
    store_path = "s3://my-bucket/delta_store"
    storage_options = {"AWS_REGION": "us-west-2"}

    store = DeltaMetadataStore(
        store_path,
        storage_options=storage_options,
        streaming_chunk_size=100,
        auto_create_tables=False,
    )

    assert store.streaming_chunk_size == 100
    assert store._is_remote is True


def test_delta_s3_with_no_credentials() -> None:
    """Test S3 store initialization without explicit credentials (uses IAM role)."""
    store_path = "s3://my-bucket/delta_store"

    # No storage_options - would use IAM role or environment variables in production
    store = DeltaMetadataStore(store_path, auto_create_tables=False)

    assert store.storage_options == {}
    assert store._is_remote is True


def test_delta_azure_adls_configuration() -> None:
    """Test Azure ADLS Gen2 configuration."""
    store_path = "abfss://container@account.dfs.core.windows.net/delta_store"
    storage_options = {
        "AZURE_STORAGE_ACCOUNT_NAME": "account",
        "AZURE_STORAGE_ACCOUNT_KEY": "key",
    }

    store = DeltaMetadataStore(
        store_path, storage_options=storage_options, auto_create_tables=False
    )

    assert store._is_remote is True
    assert store.storage_options == storage_options


def test_delta_gcs_configuration() -> None:
    """Test Google Cloud Storage configuration."""
    store_path = "gs://my-bucket/delta_store"
    storage_options = {
        "GOOGLE_SERVICE_ACCOUNT": "/path/to/service-account.json",
    }

    store = DeltaMetadataStore(
        store_path, storage_options=storage_options, auto_create_tables=False
    )

    assert store._is_remote is True
    assert store.storage_options == storage_options


def test_delta_with_fallback_stores(tmp_path, test_graph, test_features) -> None:
    """Test Delta store with fallback stores."""
    from metaxy.metadata_store.memory import InMemoryMetadataStore

    primary_path = tmp_path / "primary"
    feature_cls = test_features["UpstreamFeatureA"]

    # Create fallback store with data (keep it open for reads)
    fallback = InMemoryMetadataStore()
    with fallback:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        with fallback.allow_cross_project_writes():
            fallback.write_metadata(feature_cls, metadata)

        # Create Delta store with fallback while fallback is still open
        with DeltaMetadataStore(primary_path, fallback_stores=[fallback]) as primary:
            # Should read from fallback since not in primary
            result = collect_to_polars(primary.read_metadata(feature_cls))
            assert len(result) == 2


def test_delta_append_to_existing_table(tmp_path, test_graph, test_features) -> None:
    """Test appending new data to existing Delta table."""
    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]

    with DeltaMetadataStore(store_path) as store:
        # First write
        metadata1 = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        store.write_metadata(feature_cls, metadata1)

        # Second write (append)
        metadata2 = pl.DataFrame(
            {
                "sample_uid": [3, 4],
                "metaxy_provenance_by_field": [
                    {"frames": "h3", "audio": "h3"},
                    {"frames": "h4", "audio": "h4"},
                ],
            }
        )
        store.write_metadata(feature_cls, metadata2)

        # Verify all rows
        result = collect_to_polars(store.read_metadata(feature_cls))
        assert len(result) == 4
        assert set(result["sample_uid"].to_list()) == {1, 2, 3, 4}


def test_delta_read_with_filters(tmp_path, test_graph, test_features) -> None:
    """Test reading with filters."""
    import narwhals as nw

    store_path = tmp_path / "delta"
    feature_cls = test_features["UpstreamFeatureA"]

    with DeltaMetadataStore(store_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 4, 5],
                "metaxy_provenance_by_field": [
                    {"frames": f"h{i}", "audio": f"h{i}"} for i in range(1, 6)
                ],
            }
        )
        store.write_metadata(feature_cls, metadata)

        # Read with filter
        result = store.read_metadata(feature_cls, filters=[nw.col("sample_uid") > 3])
        assert result is not None
        collected = collect_to_polars(result)
        assert len(collected) == 2
        assert set(collected["sample_uid"].to_list()) == {4, 5}


def test_delta_read_with_column_selection(tmp_path, test_graph, test_features) -> None:
    """Test reading with column selection."""
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
        store.write_metadata(feature_cls, metadata)

        # Read with column selection
        result = store.read_metadata(feature_cls, columns=["sample_uid"])
        assert result is not None
        collected = collect_to_polars(result)
        assert list(collected.columns) == ["sample_uid"]
