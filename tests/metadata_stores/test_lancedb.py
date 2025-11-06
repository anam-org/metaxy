"""LanceDB-specific tests."""

from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("lancedb")

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.lancedb import LanceDBMetadataStore


def test_lancedb_write_and_read(tmp_path, test_graph, test_features) -> None:
    """Write metadata and read it back."""
    database_path = tmp_path / "lancedb"
    feature_cls = test_features["UpstreamFeatureA"]

    with LanceDBMetadataStore(database_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                    {"frames": "h3", "audio": "h3"},
                ],
            }
        )

        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

        result = collect_to_polars(store.read_metadata(feature_cls))
        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_lancedb_persistence_across_instances(
    tmp_path, test_graph, test_features
) -> None:
    """Ensure data persists across store instances."""
    database_path = tmp_path / "lancedb"
    feature_cls = test_features["UpstreamFeatureA"]

    with LanceDBMetadataStore(database_path) as store:
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

    with LanceDBMetadataStore(database_path) as store:
        result = collect_to_polars(store.read_metadata(feature_cls))
        assert len(result) == 2


def test_lancedb_drop_feature(tmp_path, test_graph, test_features) -> None:
    """Dropping a feature removes the Lance table and allows re-creation."""
    database_path = tmp_path / "lancedb"
    feature_cls = test_features["UpstreamFeatureA"]

    with LanceDBMetadataStore(database_path) as store:
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

        feature_key = feature_cls.spec().key  # type: ignore[attr-defined]
        assert feature_key in store.list_features()

        store.drop_feature_metadata(feature_cls)
        assert feature_key not in store.list_features()

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


def test_lancedb_display(tmp_path) -> None:
    """Display includes path and feature count."""
    database_path = tmp_path / "lancedb"
    store = LanceDBMetadataStore(database_path)

    closed_display = store.display()
    assert "LanceDBMetadataStore" in closed_display
    assert str(database_path) in closed_display

    with store:
        open_display = store.display()
        assert "features=0" in open_display


def test_lancedb_table_naming(tmp_path, test_graph, test_features) -> None:
    """Test that feature keys are converted to table names correctly."""
    database_path = tmp_path / "lancedb"
    feature_cls = test_features["UpstreamFeatureA"]

    with LanceDBMetadataStore(database_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

        # Check table was created with correct name
        table_names = store.conn.table_names()  # type: ignore[attr-defined]
        assert "test_stores__upstream_a" in table_names


def test_lancedb_close_idempotent(tmp_path, test_graph, test_features) -> None:
    """Test that close() can be called multiple times safely."""
    database_path = tmp_path / "lancedb"
    store = LanceDBMetadataStore(database_path)

    with store:
        pass

    # Close again manually (should not raise)
    store.close()
    store.close()


def test_lancedb_conn_property_enforcement(tmp_path, test_graph, test_features) -> None:
    """Test that conn property enforces store is open."""
    from metaxy.metadata_store import StoreNotOpenError

    database_path = tmp_path / "lancedb"
    store = LanceDBMetadataStore(database_path)

    # Should raise when accessing conn while closed
    with pytest.raises(StoreNotOpenError, match="LanceDB connection is not open"):
        _ = store.conn

    # Should work when open
    with store:
        conn = store.conn
        assert conn is not None


def test_lancedb_system_tables_filtered(tmp_path, test_graph, test_features) -> None:
    """Test that system tables are not included in list_features()."""
    database_path = tmp_path / "lancedb"

    with LanceDBMetadataStore(database_path) as store:
        # Write some user feature
        feature_cls = test_features["UpstreamFeatureA"]
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        with store.allow_cross_project_writes():
            store.write_metadata(feature_cls, metadata)

        # Manually create a system table (simulating metaxy_graph_push)
        from metaxy.metadata_store.system_tables import FEATURE_VERSIONS_KEY

        system_metadata = pl.DataFrame(
            {
                "project": ["test"],
                "feature_key": ["test__feature"],
                "metaxy_feature_version": ["v1"],
                "metaxy_feature_spec_version": ["spec1"],
                "metaxy_feature_tracking_version": ["track1"],
                "recorded_at": [pl.datetime(2024, 1, 1)],
                "feature_spec": ['{"key": ["test", "feature"]}'],
                "feature_class_path": ["test.Feature"],
                "metaxy_snapshot_version": ["snap1"],
            }
        )
        store._write_metadata_impl(FEATURE_VERSIONS_KEY, system_metadata)

        # list_features() should only return user features, not system tables
        features = store.list_features()
        assert len(features) == 1
        assert features[0] == feature_cls.spec().key  # type: ignore[attr-defined]


def test_lancedb_hash_algorithm(tmp_path, test_graph, test_features) -> None:
    """Test that custom hash algorithm is respected."""
    from metaxy.data_versioning.hash_algorithms import HashAlgorithm

    database_path = tmp_path / "lancedb"

    with LanceDBMetadataStore(
        database_path, hash_algorithm=HashAlgorithm.SHA256
    ) as store:
        assert store.hash_algorithm == HashAlgorithm.SHA256


def test_lancedb_with_fallback_stores(tmp_path, test_graph, test_features) -> None:
    """Test LanceDB store with fallback stores."""
    prod_path = tmp_path / "prod"
    dev_path = tmp_path / "dev"
    feature_cls = test_features["UpstreamFeatureA"]

    # Write to production store
    with LanceDBMetadataStore(prod_path) as prod_store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        with prod_store.allow_cross_project_writes():
            prod_store.write_metadata(feature_cls, metadata)

    # Create dev store with prod as fallback
    with LanceDBMetadataStore(prod_path) as prod_store_readonly:
        with LanceDBMetadataStore(
            dev_path, fallback_stores=[prod_store_readonly]
        ) as dev_store:
            # Should be able to read from fallback when feature not in local store
            result = collect_to_polars(dev_store.read_metadata(feature_cls))
            assert len(result) == 2
            assert set(result["sample_uid"].to_list()) == {1, 2}

            # Verify feature is not in local dev store
            assert not dev_store.has_feature(feature_cls, check_fallback=False)

            # But exists when checking fallback
            assert dev_store.has_feature(feature_cls, check_fallback=True)
