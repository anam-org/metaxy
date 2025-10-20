"""ClickHouse-specific tests that don't apply to other stores."""

import pytest

# Skip all tests in this module if ClickHouse not available
pytest.importorskip("ibis")

try:
    import ibis.backends.clickhouse  # noqa: F401
except ImportError:
    pytest.skip("ibis-clickhouse not installed", allow_module_level=True)

from metaxy.metadata_store.clickhouse import ClickHouseMetadataStore
from metaxy.models.feature import Feature


def test_clickhouse_table_naming(
    clickhouse_db: str, test_registry, test_features: dict[str, type[Feature]]
) -> None:
    """Test that feature keys are converted to table names correctly.

    Args:
        clickhouse_db: Connection string fixture
        test_registry: Feature registry fixture (for context)
        test_features: Dict with test feature classes
    """
    with ClickHouseMetadataStore(clickhouse_db) as store:
        import polars as pl

        metadata = pl.DataFrame(
            {
                "sample_id": [1],
                "data_version": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], metadata)

        # Check table was created with correct name using Ibis
        table_names = store.ibis_conn.list_tables()
        assert "test_stores__upstream_a" in table_names


def test_clickhouse_uses_ibis_backend(
    clickhouse_db: str, test_registry, test_features: dict[str, type[Feature]]
) -> None:
    """Test that ClickHouse store uses Ibis backend.

    Args:
        clickhouse_db: Connection string fixture
        test_registry: Feature registry fixture (for context)
        test_features: Dict with test feature classes
    """
    with ClickHouseMetadataStore(clickhouse_db) as store:
        # Should have ibis_conn
        assert hasattr(store, "ibis_conn")
        # Backend should be clickhouse
        assert store._conn is not None


def test_clickhouse_conn_property_enforcement(
    clickhouse_db: str, test_registry, test_features: dict[str, type[Feature]]
) -> None:
    """Test that conn property enforces store is open.

    Args:
        clickhouse_db: Connection string fixture
        test_registry: Feature registry fixture (for context)
        test_features: Dict with test feature classes
    """
    from metaxy.metadata_store import StoreNotOpenError

    store = ClickHouseMetadataStore(clickhouse_db)

    # Should raise when accessing conn while closed (Ibis error message)
    with pytest.raises(StoreNotOpenError, match="Ibis connection is not open"):
        _ = store.conn

    # Should work when open
    with store:
        conn = store.conn
        assert conn is not None


def test_clickhouse_persistence(
    clickhouse_db: str, test_registry, test_features: dict[str, type[Feature]]
) -> None:
    """Test that data persists across different store instances.

    Args:
        clickhouse_db: Connection string fixture
        test_registry: Feature registry fixture (for context)
        test_features: Dict with test feature classes
    """
    import polars as pl

    # Write data in first instance
    with ClickHouseMetadataStore(clickhouse_db) as store1:
        metadata = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "data_version": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                    {"frames": "h3", "audio": "h3"},
                ],
            }
        )
        store1.write_metadata(test_features["UpstreamFeatureA"], metadata)

    # Read data in second instance
    with ClickHouseMetadataStore(clickhouse_db) as store2:
        result = store2.read_metadata(test_features["UpstreamFeatureA"])

        assert len(result) == 3
        assert set(result["sample_id"].to_list()) == {1, 2, 3}


def test_clickhouse_close_idempotent(
    clickhouse_db: str, test_registry, test_features: dict[str, type[Feature]]
) -> None:
    """Test that close() can be called multiple times safely.

    Args:
        clickhouse_db: Connection string fixture
        test_registry: Feature registry fixture (for context)
        test_features: Dict with test feature classes
    """
    store = ClickHouseMetadataStore(clickhouse_db)

    with store:
        pass

    # Close again manually (should not raise)
    store.close()
    store.close()


def test_clickhouse_hash_algorithms(
    clickhouse_db: str, test_registry, test_features: dict[str, type[Feature]]
) -> None:
    """Test that ClickHouse supports MD5, XXHASH32, and XXHASH64 hash algorithms.

    Args:
        clickhouse_db: Connection string fixture
        test_registry: Feature registry fixture (for context)
        test_features: Dict with test feature classes
    """
    import polars as pl

    from metaxy.data_versioning.hash_algorithms import HashAlgorithm

    # Test each supported algorithm
    for algorithm in [
        HashAlgorithm.MD5,
        HashAlgorithm.XXHASH32,
        HashAlgorithm.XXHASH64,
    ]:
        with ClickHouseMetadataStore(clickhouse_db, hash_algorithm=algorithm) as store:
            # Drop the feature metadata before each iteration to ensure clean state
            # Since we're testing the same feature with different hash algorithms
            store.drop_feature_metadata(test_features["UpstreamFeatureA"])

            metadata = pl.DataFrame(
                {
                    "sample_id": [1, 2],
                    "data_version": [
                        {"frames": "h1", "audio": "h1"},
                        {"frames": "h2", "audio": "h2"},
                    ],
                }
            )
            store.write_metadata(test_features["UpstreamFeatureA"], metadata)

            result = store.read_metadata(test_features["UpstreamFeatureA"])
            assert len(result) == 2
