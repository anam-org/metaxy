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


def test_clickhouse_table_naming(clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]) -> None:
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
        table_names = store.conn.list_tables()
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
        # Should have conn
        assert hasattr(store, "conn")
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


def test_clickhouse_persistence(clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]) -> None:
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
        result = collect_to_polars(store2.read_metadata(test_features["UpstreamFeatureA"]))

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

            result = collect_to_polars(store.read_metadata(test_features["UpstreamFeatureA"]))
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


def test_clickhouse_config_with_connection_params(test_graph, test_features: dict[str, type[SampleFeature]]) -> None:
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
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
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
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
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
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
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
                metaxy_deleted_at,
                metaxy_materialization_id,
                metaxy_feature_spec_version
            ) VALUES
            (1, '{provenance_json}', 'prov1', 'v1', 'sv1', '{version_json}', 'dv1', now(), NULL, 'm1', 'fs1'),
            (2, '{provenance_json}', 'prov2', 'v1', 'sv1', '{version_json}', 'dv2', now(), NULL, 'm1', 'fs1')
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
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
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
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Insert data with Map values via SQL
        # UpstreamFeatureA has fields: frames, audio
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
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
                metaxy_deleted_at,
                metaxy_materialization_id,
                metaxy_feature_spec_version
            ) VALUES
            (1, {{'frames': 'hash1', 'audio': 'hash2'}}, 'prov1', 'v1', 'sv1', {{'frames': 'v1', 'audio': 'v1'}}, 'dv1', now(), NULL, 'm1', 'fs1'),
            (2, {{'frames': 'hash3', 'audio': 'hash4'}}, 'prov2', 'v1', 'sv1', {{'frames': 'v2', 'audio': 'v2'}}, 'dv2', now(), NULL, 'm1', 'fs1')
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
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
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
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
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
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
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
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
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


def test_clickhouse_map_column_write_from_ibis_struct(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test writing Ibis-backed DataFrame with Struct columns to Map columns.

    This tests the scenario where metadata is computed using Ibis (e.g., from
    a SQL query or join), resulting in a Narwhals DataFrame backed by Ibis
    with Struct columns for metaxy_provenance_by_field.

    The bug was that _transform_struct_to_map only handled Polars DataFrames,
    so Ibis-backed DataFrames with Struct columns were passed through unchanged,
    causing ClickHouse to fail with:
        "CAST AS Map from tuple requires 2 elements"

    The fix adds _transform_ibis_struct_to_map which uses ibis.map() to convert
    Ibis Struct columns to Map columns.
    """
    import ibis
    import narwhals as nw

    from metaxy.models.constants import (
        METAXY_DATA_VERSION_BY_FIELD,
        METAXY_PROVENANCE_BY_FIELD,
    )

    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key
    plan = test_graph.get_feature_plan(feature_key)

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with Map columns (the production schema)
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
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
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Create an Ibis memtable - simulating data from a SQL query
        ibis_table = ibis.memtable(
            {
                "sample_uid": [1, 2, 3],
                # Temporary columns that will be used to build the struct
                "_hash_frames": ["hash1", "hash2", "hash3"],
                "_hash_audio": ["hash_a1", "hash_a2", "hash_a3"],
            }
        )

        # Wrap in Narwhals and use the actual versioning engine to build struct
        nw_df = nw.from_native(ibis_table, eager_only=False)

        # Use the store's versioning engine to build the struct column
        # This is exactly how resolve_update builds metaxy_provenance_by_field
        with store.create_versioning_engine(plan, implementation=nw.Implementation.IBIS) as engine:
            # Build struct using the engine's method (same as production code)
            nw_df = engine.record_field_versions(
                nw_df,
                METAXY_PROVENANCE_BY_FIELD,
                {"frames": "_hash_frames", "audio": "_hash_audio"},
            )
            nw_df = engine.record_field_versions(
                nw_df,
                METAXY_DATA_VERSION_BY_FIELD,
                {"frames": "_hash_frames", "audio": "_hash_audio"},
            )

        # Drop temporary columns
        nw_df = nw_df.drop("_hash_frames", "_hash_audio")

        # Verify it's still Ibis-backed
        assert nw_df.implementation == nw.Implementation.IBIS

        # Write to the store - this should transform Ibis Struct -> Map
        # Before the fix, this raised:
        # "CAST AS Map from tuple requires 2 elements. Left type: Tuple(Nullable(String)), right type: Map(String, String)"
        store.write_metadata(feature_cls, nw_df)

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
        assert provenance["frames"] == "hash1"
        assert provenance["audio"] == "hash_a1"

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_user_defined_map_column(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test that user-defined Map(String, T) columns are preserved (not transformed).

    Users may define their own Map columns in ClickHouse tables. Unlike metaxy's
    system columns (metaxy_provenance_by_field, metaxy_data_version_by_field),
    user Map columns are NOT converted to Struct.

    In Polars, ClickHouse Map columns appear as List[Struct{key, value}] because
    that's how Arrow serializes Map types. This is different from metaxy columns
    which are explicitly converted to named Struct for downstream compatibility.

    This test verifies:
    1. User Map columns are readable (no Ibis/PyArrow errors)
    2. User Map columns remain as List[Struct{key,value}] format (not dict)
    3. Metaxy Map columns are converted to dict (Struct)
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with both metaxy Map columns AND a user-defined Map column
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                user_metadata Map(String, String),
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Insert data with user Map column
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            INSERT INTO {table_name} (
                sample_uid,
                user_metadata,
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
            (1, {{'source': 'camera1', 'quality': 'high'}}, {{'frames': 'hash1', 'audio': 'hash2'}}, 'prov1', 'v1', 'sv1', {{'frames': 'v1', 'audio': 'v1'}}, 'dv1', now(), 'm1', 'fs1'),
            (2, {{'source': 'camera2', 'resolution': '4k'}}, {{'frames': 'hash3', 'audio': 'hash4'}}, 'prov2', 'v1', 'sv1', {{'frames': 'v2', 'audio': 'v2'}}, 'dv2', now(), 'm1', 'fs1')
        """
        )

        # Read via read_metadata_in_store
        # This uses transform_after_read which should:
        # - Convert metaxy Map columns to Struct (dict)
        # - Leave user_metadata Map column as-is (List[Struct{key,value}])
        read_result = store.read_metadata_in_store(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result)

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}

        # Metaxy Map columns should be converted to Struct (dict in Python)
        provenance = result["metaxy_provenance_by_field"][0]
        assert isinstance(provenance, dict), f"Expected dict, got {type(provenance)}"
        assert provenance["frames"] == "hash1"

        # User Map column remains as List[Struct{key,value}] in Polars
        # This is the Arrow Map representation - NOT converted to dict
        user_meta = result["user_metadata"][0]
        # In Polars, each row's Map value is a Series of struct{key, value}
        assert isinstance(user_meta, pl.Series), f"Expected pl.Series, got {type(user_meta)}"
        # Verify we can access the data
        assert len(user_meta) == 2  # Two key-value pairs: source, quality

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_auto_cast_struct_for_map_true(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test auto_cast_struct_for_map=True converts user Struct columns to Map on write.

    When auto_cast_struct_for_map=True (default), DataFrame Struct columns are
    automatically converted to Map format when the corresponding ClickHouse column
    is Map type. This allows users to write Polars Struct data directly to Map columns.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    # auto_cast_struct_for_map=True is the default
    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with user Map column
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                user_tags Map(String, String),
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Create DataFrame with user Struct column (user_tags)
        samples = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "user_tags": [
                    {"env": "prod", "team": "ml"},
                    {"env": "staging", "team": "data"},
                ],
                "metaxy_provenance_by_field": [
                    {"frames": "hash1", "audio": "hash2"},
                    {"frames": "hash3", "audio": "hash4"},
                ],
            }
        )

        # Write should succeed - user Struct is auto-converted to Map
        store.write_metadata(feature_cls, samples)

        # Read back and verify data was written
        read_result = store.read_metadata_in_store(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result)

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}

        # Verify user_tags Map data is readable (as List[Struct{key,value}] in Polars)
        user_tags = result["user_tags"][0]
        assert isinstance(user_tags, pl.Series), f"Expected pl.Series, got {type(user_tags)}"
        # Convert to dict for easier assertion
        tags_dict = {row["key"]: row["value"] for row in user_tags.to_list()}
        assert tags_dict["env"] == "prod"
        assert tags_dict["team"] == "ml"

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_auto_cast_struct_for_map_false(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test auto_cast_struct_for_map=False does NOT convert user Struct columns.

    When auto_cast_struct_for_map=False, only metaxy system columns are converted.
    User Struct columns are not converted, which will cause ClickHouse insert to fail
    when the target column is Map type (ClickHouse can't insert Struct into Map).
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False, auto_cast_struct_for_map=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with user Map column
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                user_tags Map(String, String),
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Create DataFrame with user Struct column
        samples = pl.DataFrame(
            {
                "sample_uid": [1],
                "user_tags": [{"env": "prod", "team": "ml"}],
                "metaxy_provenance_by_field": [{"frames": "hash1", "audio": "hash2"}],
            }
        )

        # Write should FAIL because user Struct won't be converted to Map
        # ClickHouse will reject the insert with a type mismatch error
        with pytest.raises(Exception):  # Ibis/ClickHouse error on type mismatch
            store.write_metadata(feature_cls, samples)

        # Clean up
        if table_name in conn.list_tables():
            conn.drop_table(table_name)


def test_clickhouse_auto_cast_struct_for_map_ibis_dataframe(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test auto_cast_struct_for_map=True works with Ibis-backed DataFrames.

    This test verifies that user-defined Struct columns in Ibis DataFrames
    are correctly converted to Map format when writing to ClickHouse Map columns.
    """
    import ibis
    import narwhals as nw

    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with user Map column
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                user_tags Map(String, String),
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Create an Ibis memtable with columns to build structs from
        ibis_table = ibis.memtable(
            {
                "sample_uid": [1, 2],
                "_tag_env": ["prod", "staging"],
                "_tag_team": ["ml", "data"],
                "_prov_frames": ["hash1", "hash2"],
                "_prov_audio": ["hash_a1", "hash_a2"],
            }
        )

        # Build struct columns using ibis.struct()
        ibis_table = ibis_table.mutate(
            user_tags=ibis.struct({"env": ibis_table["_tag_env"], "team": ibis_table["_tag_team"]}),
            metaxy_provenance_by_field=ibis.struct(
                {
                    "frames": ibis_table["_prov_frames"],
                    "audio": ibis_table["_prov_audio"],
                }
            ),
        )
        ibis_table = ibis_table.drop("_tag_env", "_tag_team", "_prov_frames", "_prov_audio")

        # Wrap in Narwhals
        nw_df = nw.from_native(ibis_table, eager_only=False)

        # Verify it's Ibis-backed
        assert nw_df.implementation == nw.Implementation.IBIS

        # Write should succeed - both user and metaxy Struct columns are converted to Map
        store.write_metadata(feature_cls, nw_df)

        # Read back and verify
        read_result = store.read_metadata_in_store(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result)

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}

        # Verify user_tags Map data is readable
        user_tags = result["user_tags"][0]
        assert isinstance(user_tags, pl.Series), f"Expected pl.Series, got {type(user_tags)}"
        tags_dict = {row["key"]: row["value"] for row in user_tags.to_list()}
        assert tags_dict["env"] == "prod"
        assert tags_dict["team"] == "ml"

        # Verify metaxy columns still work
        provenance = result["metaxy_provenance_by_field"][0]
        assert isinstance(provenance, dict), f"Expected dict, got {type(provenance)}"
        assert provenance["frames"] == "hash1"

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_auto_cast_struct_for_map_non_string_values(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test auto_cast_struct_for_map with non-string Map value types.

    This test verifies that when a ClickHouse table has Map(String, Int64) columns,
    Struct fields are correctly cast to Int64 before insertion.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with Map(String, Int64) column for counts
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                field_counts Map(String, Int64),
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Create DataFrame with Struct column containing integers
        samples = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "field_counts": [
                    {"frames_count": 10, "audio_count": 5},
                    {"frames_count": 20, "audio_count": 8},
                ],
                "metaxy_provenance_by_field": [
                    {"frames": "hash1", "audio": "hash2"},
                    {"frames": "hash3", "audio": "hash4"},
                ],
            }
        )

        # Write should succeed - Struct int values are cast to Int64 for Map(String, Int64)
        store.write_metadata(feature_cls, samples)

        # Read back and verify
        read_result = store.read_metadata_in_store(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result)

        assert len(result) == 2

        # Verify field_counts Map data has correct integer values
        field_counts = result["field_counts"][0]
        assert isinstance(field_counts, pl.Series), f"Expected pl.Series, got {type(field_counts)}"
        counts_dict = {row["key"]: row["value"] for row in field_counts.to_list()}
        assert counts_dict["frames_count"] == 10
        assert counts_dict["audio_count"] == 5

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_auto_cast_struct_for_map_empty_struct(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test auto_cast_struct_for_map handles empty structs gracefully.

    Empty structs (struct[0] with no fields) should be skipped during
    transformation since they can't be converted to a Map.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with Map column for empty struct
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                empty_metadata Map(String, String),
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Create DataFrame with empty Struct column (struct[0])
        # This simulates the preview_summary: <struct[0]> {}, {} case
        empty_struct_schema = pl.Struct([])  # Empty struct with no fields
        samples = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "empty_metadata": pl.Series([{}, {}]).cast(empty_struct_schema),
                "metaxy_provenance_by_field": [
                    {"frames": "hash1", "audio": "hash2"},
                    {"frames": "hash3", "audio": "hash4"},
                ],
            }
        )

        # Verify the empty struct has no fields
        assert isinstance(samples.schema["empty_metadata"], pl.Struct)
        assert len(samples.schema["empty_metadata"].fields) == 0

        # Write should succeed - empty struct is skipped, other structs converted
        store.write_metadata(feature_cls, samples)

        # Read back and verify data was written
        read_result = store.read_metadata_in_store(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result)

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_auto_cast_struct_for_map_null_values(
    clickhouse_db: str, test_graph, test_features: dict[str, type[SampleFeature]]
) -> None:
    """Test that Struct columns with NULL values are handled correctly.

    When a Polars Struct has fields with NULL values (e.g., `{'a': 1, 'b': None}`),
    and the target ClickHouse column is Map(String, Int64) (non-nullable values),
    the NULL entries should be filtered out since ClickHouse Maps don't support
    NULL values unless explicitly declared as Nullable.

    This tests the fix for the error:
    "Cannot convert NULL value to non-Nullable type: while converting source
    column selected_frame_indices to destination column selected_frame_indices"
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec().key

    with ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False, auto_cast_struct_for_map=True) as store:
        conn = store.conn
        table_name = store.get_table_name(feature_key)

        # Clean up if exists
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

        # Create table with Map(String, Int64) column - non-nullable values
        conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid Int64,
                selected_frame_indices Map(String, Int64),
                metaxy_provenance_by_field Map(String, String),
                metaxy_provenance String,
                metaxy_feature_version String,
                metaxy_snapshot_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String,
                metaxy_feature_spec_version String
            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Create DataFrame with Struct column containing NULL values
        # This simulates: selected_frame_indices: {'eyes_open_true': 38, 'eyes_open_false': None}
        samples = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "selected_frame_indices": [
                    {
                        "eyes_open_true": 38,
                        "eyes_open_false": None,
                        "head_pitch_min": 5,
                    },
                    {
                        "eyes_open_true": None,
                        "eyes_open_false": 67,
                        "head_pitch_min": None,
                    },
                ],
                "metaxy_provenance_by_field": [
                    {"frames": "hash1", "audio": "hash2"},
                    {"frames": "hash3", "audio": "hash4"},
                ],
            }
        )

        # Verify the struct has nullable int fields
        struct_type = samples.schema["selected_frame_indices"]
        assert isinstance(struct_type, pl.Struct)

        # Write should succeed - NULL values are filtered out
        store.write_metadata(feature_cls, samples)

        # Read back and verify data was written
        read_result = store.read_metadata_in_store(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result)

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}

        # Verify the Map data - NULL entries should be filtered out
        # Row 1: {'eyes_open_true': 38, 'head_pitch_min': 5} (eyes_open_false filtered)
        # Row 2: {'eyes_open_false': 67} (eyes_open_true and head_pitch_min filtered)
        row1 = result.filter(pl.col("sample_uid") == 1)["selected_frame_indices"][0]
        row2 = result.filter(pl.col("sample_uid") == 2)["selected_frame_indices"][0]

        # Convert list of {key, value} structs to dict for easier checking
        row1_dict = {item["key"]: item["value"] for item in row1}
        row2_dict = {item["key"]: item["value"] for item in row2}

        assert row1_dict == {"eyes_open_true": 38, "head_pitch_min": 5}
        assert row2_dict == {"eyes_open_false": 67}

        # Clean up
        conn.drop_table(table_name)
