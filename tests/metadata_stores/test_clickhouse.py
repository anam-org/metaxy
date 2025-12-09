"""ClickHouse-specific tests that don't apply to other stores."""

import polars as pl
import pytest

# Skip all tests in this module if ClickHouse not available
pytest.importorskip("ibis")

try:
    import ibis.backends.clickhouse  # noqa: F401
except ImportError:
    pytest.skip("ibis-clickhouse not installed", allow_module_level=True)

from metaxy._testing.models import SampleFeature
from metaxy._utils import collect_to_polars
from metaxy.metadata_store.clickhouse import ClickHouseMetadataStore


def test_clickhouse_table_naming(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test that feature keys are converted to table names correctly.

    Args:
        clickhouse_db: Connection string fixture
        test_graph: Feature graph fixture (for context)
        test_features: Dict with test feature classes
    """
    with ClickHouseMetadataStore(clickhouse_db) as store:
        import polars as pl

        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], metadata)

        # Check table was created with correct name using Ibis
        table_names = store.ibis_conn.list_tables()
        assert "test_stores__upstream_a" in table_names


def test_clickhouse_uses_ibis_backend(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test that ClickHouse store uses Ibis backend.

    Args:
        clickhouse_db: Connection string fixture
        test_graph: Feature graph fixture (for context)
        test_features: Dict with test feature classes
    """
    with ClickHouseMetadataStore(clickhouse_db) as store:
        # Should have ibis_conn
        assert hasattr(store, "ibis_conn")
        # Backend should be clickhouse
        assert store._conn is not None


def test_clickhouse_conn_property_enforcement(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test that conn property enforces store is open.

    Args:
        clickhouse_db: Connection string fixture
        test_graph: Feature graph fixture (for context)
        test_features: Dict with test feature classes
    """
    from metaxy.metadata_store import StoreNotOpenError

    store = ClickHouseMetadataStore(clickhouse_db)

    # Should raise when accessing conn while closed (Ibis error message)
    with pytest.raises(StoreNotOpenError, match="Ibis connection is not open"):
        _ = store.conn

    # Should work when open
    with store.open():
        conn = store.conn
        assert conn is not None

    with store.open("write"):
        conn = store.conn
        assert conn is not None


def test_clickhouse_persistence(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test that data persists across different store instances.

    Args:
        clickhouse_db: Connection string fixture
        test_graph: Feature graph fixture (for context)
        test_features: Dict with test feature classes
    """

    # Write data in first instance
    with ClickHouseMetadataStore(clickhouse_db) as store1:
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
        store1.write_metadata(test_features["UpstreamFeatureA"], metadata)

    # Read data in second instance
    with ClickHouseMetadataStore(clickhouse_db) as store2:
        result = collect_to_polars(
            store2.read_metadata(test_features["UpstreamFeatureA"])
        )

        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_clickhouse_hash_algorithms(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test that ClickHouse supports MD5, XXHASH32, and XXHASH64 hash algorithms.

    Args:
        clickhouse_db: Connection string fixture
        test_graph: Feature graph fixture (for context)
        test_features: Dict with test feature classes
    """

    from metaxy.versioning.types import HashAlgorithm

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
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"frames": "h1", "audio": "h1"},
                        {"frames": "h2", "audio": "h2"},
                    ],
                }
            )
            store.write_metadata(test_features["UpstreamFeatureA"], metadata)

            result = collect_to_polars(
                store.read_metadata(test_features["UpstreamFeatureA"])
            )
            assert len(result) == 2


def test_clickhouse_config_instantiation(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test instantiating ClickHouse store via MetaxyConfig."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "clickhouse_store": StoreConfig(
                type="metaxy.metadata_store.clickhouse.ClickHouseMetadataStore",
                config={
                    "connection_string": clickhouse_db,
                },
            )
        }
    )

    store = config.get_store("clickhouse_store")
    assert isinstance(store, ClickHouseMetadataStore)

    # Verify store can be opened
    with store.open("write"):
        assert store._is_open


def test_clickhouse_config_with_connection_params(
    test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test ClickHouse store config with connection_params."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "clickhouse_store": StoreConfig(
                type="metaxy.metadata_store.clickhouse.ClickHouseMetadataStore",
                config={
                    "connection_params": {
                        "host": "localhost",
                        "port": 9000,
                        "database": "default",
                        "user": "default",
                        "password": "",
                    },
                },
            )
        }
    )

    store = config.get_store("clickhouse_store")
    assert isinstance(store, ClickHouseMetadataStore)


def test_clickhouse_config_with_hash_algorithm(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test ClickHouse store config with specific hash algorithm."""
    from metaxy.config import MetaxyConfig, StoreConfig
    from metaxy.versioning.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "clickhouse_store": StoreConfig(
                type="metaxy.metadata_store.clickhouse.ClickHouseMetadataStore",
                config={
                    "connection_string": clickhouse_db,
                    "hash_algorithm": "md5",
                },
            )
        }
    )

    store = config.get_store("clickhouse_store")
    assert isinstance(store, ClickHouseMetadataStore)
    assert store.hash_algorithm == HashAlgorithm.MD5

    with store.open("write"):
        assert store._is_open


def test_clickhouse_config_with_fallback_stores(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test ClickHouse store config with fallback stores."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.clickhouse.ClickHouseMetadataStore",
                config={
                    "connection_string": clickhouse_db,
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.clickhouse.ClickHouseMetadataStore",
                config={
                    "connection_string": clickhouse_db,
                },
            ),
        }
    )

    dev_store = config.get_store("dev")
    assert isinstance(dev_store, ClickHouseMetadataStore)
    assert len(dev_store.fallback_stores) == 1
    assert isinstance(dev_store.fallback_stores[0], ClickHouseMetadataStore)

    with dev_store.open("write"):
        assert dev_store._is_open


def test_clickhouse_struct_to_map_conversion(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test that Polars Struct columns are converted to ClickHouse Map format.

    This tests the automatic conversion from Polars Struct to ClickHouse Map
    when the database schema defines Map(String, String) columns.

    Args:
        clickhouse_db: Connection string fixture
        test_graph: Feature graph fixture (for context)
        test_features: Dict with test feature classes
    """
    # Use a unique table prefix to avoid conflicts with other tests
    table_prefix = "map_test_"
    table_name = f"{table_prefix}test_stores__upstream_a"

    with ClickHouseMetadataStore(
        clickhouse_db, auto_create_tables=False, table_prefix=table_prefix
    ) as store:
        # Drop table if exists and create with explicit Map type
        conn = store.conn
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with Map(String, String) column using raw SQL
        conn.raw_sql(f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(3, 'UTC'),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """)

        # Write data with Polars Struct (dict) - should be auto-converted to Map
        # Use "frames" and "audio" field names to match UpstreamFeatureA's fields
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "hash1", "audio": "hash2"},
                    {"frames": "hash3", "audio": "hash4"},
                ],
                "metaxy_data_version_by_field": [
                    {"frames": "v1", "audio": "v2"},
                    {"frames": "v3", "audio": "v4"},
                ],
            }
        )

        store.write_metadata(test_features["UpstreamFeatureA"], metadata)

        # Verify data was written correctly by reading back
        result = collect_to_polars(
            store.read_metadata(test_features["UpstreamFeatureA"])
        )

        assert len(result) == 2

        # Query via Ibis table to verify Map values are stored correctly
        table = conn.table(table_name)
        raw_result = (
            table.select("sample_uid", "metaxy_provenance_by_field")
            .order_by("sample_uid")
            .to_pyarrow()
        )

        # Verify that the Map values are properly stored
        # PyArrow represents Map as list of key-value structs
        map_values = raw_result["metaxy_provenance_by_field"].to_pylist()
        assert len(map_values) == 2
        # Check that first row has the expected values (order within map may vary)
        assert dict(map_values[0]) == {"frames": "hash1", "audio": "hash2"}
        assert dict(map_values[1]) == {"frames": "hash3", "audio": "hash4"}

        # Clean up
        conn.drop_table(table_name)
