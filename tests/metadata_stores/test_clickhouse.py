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


def test_clickhouse_json_column_type(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test that native ClickHouse JSON columns are handled correctly.

    When tables are created via SQLModel/Alembic with sa_type=JSON, ClickHouse
    creates native JSON columns. The ClickHouse driver returns dict objects for
    these, which PyArrow cannot handle. The store casts them to String.

    Note: We cast to String rather than Struct because ClickHouse's
    String -> Tuple CAST expects tuple syntax `('v1', 'v2')`, not JSON
    syntax `{"k": "v"}`. The JSON string can be parsed downstream if needed.

    This test simulates production usage where tables are pre-created with
    JSON columns (like metaxy_provenance_by_field, metaxy_data_version_by_field).
    """
    import json

    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with native JSON columns (like SQLModel/Alembic would)
        conn.raw_sql(  # pyright: ignore[reportAttributeAccessIssue]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                metaxy_provenance_by_field JSON,
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field JSON,
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Insert data with JSON values via SQL
        # UpstreamFeatureA has fields: frames, audio
        provenance_json = json.dumps({"frames": "hash1", "audio": "hash2"})
        version_json = json.dumps({"frames": "v1", "audio": "v1"})
        conn.raw_sql(  # pyright: ignore[reportAttributeAccessIssue]
            f"""
            INSERT INTO {table_name} (
                sample_uid,
                metaxy_provenance_by_field,
                metaxy_provenance,
                metaxy_feature_version,
                metaxy_snapshot_version,
                metaxy_data_version_by_field,
                metaxy_data_version,
                metaxy_created_at,
                metaxy_materialization_id,
                metaxy_feature_spec_version
            ) VALUES
            (1, '{provenance_json}', 'prov1', 'v1', 'sv1', '{version_json}', 'dv1', now(), 'm1', 'fs1'),
            (2, '{provenance_json}', 'prov2', 'v1', 'sv1', '{version_json}', 'dv2', now(), 'm1', 'fs1')
        """
        )

        # Read via read_metadata_in_store (no feature_version filter)
        # This uses transform_after_read internally
        # Without the fix, this would raise:
        # "pyarrow.lib.ArrowTypeError: Expected bytes, got a 'dict' object"
        read_result = store.read_metadata_in_store(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result)

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}
        # JSON columns are cast to String (not Struct, due to ClickHouse CAST limitations)
        assert isinstance(result["metaxy_provenance_by_field"][0], str)
        assert isinstance(result["metaxy_data_version_by_field"][0], str)
        # The JSON string can be parsed if needed
        parsed = json.loads(result["metaxy_provenance_by_field"][0])
        assert "frames" in parsed
        assert "audio" in parsed

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_map_column_type(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test that ClickHouse Map(K,V) columns are converted to Struct on read.

    When tables have Map(String, String) columns (common in ClickHouse for
    key-value data), the store should convert them to Struct for compatibility
    with Narwhals/Polars downstream processing.

    This also tests the write path: Polars Struct columns should be converted
    to Map-compatible format when inserting into tables with Map columns.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with Map columns (alternative to JSON for key-value data)
        conn.raw_sql(  # pyright: ignore[reportAttributeAccessIssue]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Insert data with Map values via SQL
        # UpstreamFeatureA has fields: frames, audio
        conn.raw_sql(  # pyright: ignore[reportAttributeAccessIssue]
            f"""
            INSERT INTO {table_name} (
                sample_uid,
                metaxy_provenance_by_field,
                metaxy_provenance,
                metaxy_feature_version,
                metaxy_snapshot_version,
                metaxy_data_version_by_field,
                metaxy_data_version,
                metaxy_created_at,
                metaxy_materialization_id,
                metaxy_feature_spec_version
            ) VALUES
            (1, {{'frames': 'hash1', 'audio': 'hash2'}}, 'prov1', 'v1', 'sv1', {{'frames': 'v1', 'audio': 'v1'}}, 'dv1', now(), 'm1', 'fs1'),
            (2, {{'frames': 'hash3', 'audio': 'hash4'}}, 'prov2', 'v1', 'sv1', {{'frames': 'v2', 'audio': 'v2'}}, 'dv2', now(), 'm1', 'fs1')
        """
        )

        # Read via read_metadata_in_store
        # This uses transform_after_read which should convert Map to Struct
        read_result = store.read_metadata_in_store(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result)

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}

        # Map columns should be converted to Struct (dict in Python)
        provenance = result["metaxy_provenance_by_field"][0]
        assert isinstance(provenance, dict), f"Expected dict, got {type(provenance)}"
        assert "frames" in provenance
        assert "audio" in provenance
        assert provenance["frames"] == "hash1"
        assert provenance["audio"] == "hash2"

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_map_column_empty_table_read(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test reading from an EMPTY table with Map(String, String) columns.

    This tests the critical scenario where:
    1. A table exists with Map(String, String) columns (e.g., metaxy_data_version_by_field)
    2. The table has NO data yet (empty)
    3. transform_after_read converts Map -> Struct
    4. The Struct schema must still be correctly typed even when the Map is empty

    The bug was that when converting Map to Struct for an empty table, Ibis
    used map[key] which validates keys exist, failing with KeyError.

    The fix uses map.get(key, "") instead of map[key] to safely handle empty maps.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create EMPTY table with Map columns (like production ClickHouse schema)
        conn.raw_sql(  # pyright: ignore[reportAttributeAccessIssue]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Try to read from the empty table
        # This should NOT raise KeyError even though the Map is empty
        # The error was: KeyError: 'frames'
        # Because the Map->Struct conversion couldn't handle empty maps
        read_result = store.read_metadata_in_store(feature_cls)

        # Reading from an empty table should return None or empty result
        if read_result is not None:
            result = collect_to_polars(read_result)
            assert len(result) == 0

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_map_column_resolve_update_write_metadata(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test resolve_update and write_metadata with Map(String, String) columns.

    This tests the full workflow that was failing in production:
    1. Create table with Map columns
    2. Call resolve_update with Polars DataFrame (has Struct columns)
    3. Call write_metadata which should transform Struct -> JSON for Map insertion
    4. Read back and verify data

    The key issue was that Polars Struct -> pl.Object conversion failed because
    Ibis doesn't know how to handle pl.Object type.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with Map columns (the production schema)
        conn.raw_sql(  # pyright: ignore[reportAttributeAccessIssue]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Create sample data with Struct columns (like production Polars DataFrame)
        samples = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "hash1", "audio": "hash2"},
                    {"frames": "hash3", "audio": "hash4"},
                    {"frames": "hash5", "audio": "hash6"},
                ],
            }
        )

        # resolve_update should work (materializes to Polars for comparison)
        increment = store.resolve_update(feature_cls, samples=samples)
        assert increment is not None
        assert len(increment.added) == 3
        assert len(increment.changed) == 0
        assert len(increment.removed) == 0

        # write_metadata should work (Struct -> JSON string for Map columns)
        # This is where the original error occurred: KeyError: Object
        store.write_metadata(feature_cls, samples)

        # Read back and verify
        read_result = store.read_metadata_in_store(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result)

        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}

        # Map columns should be converted back to Struct (dict in Python)
        provenance = result["metaxy_provenance_by_field"][0]
        assert isinstance(provenance, dict), f"Expected dict, got {type(provenance)}"
        assert "frames" in provenance
        assert "audio" in provenance

        # Clean up
        conn.drop_table(table_name)
