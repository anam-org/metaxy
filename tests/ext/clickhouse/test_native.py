"""ClickHouse metadata store tests."""

import ibis.backends.clickhouse  # noqa: F401
import ibis.expr.datatypes as dt
import polars as pl
import polars_map  # noqa: F401  # registers the `.map` accessor on Polars Series
import pytest
from metaxy import FeatureDefinition, HashAlgorithm
from metaxy.ext.clickhouse import ClickHouseMetadataStore
from metaxy.metadata_store import MetadataStore
from metaxy.models.feature import FeatureGraph
from metaxy.utils import collect_to_polars
from polars_map import Map

from tests.metadata_stores.shared import (
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    IbisMapTests,
    MapDtypeTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
)


@pytest.fixture
def clickhouse_store(clickhouse_db: str) -> ClickHouseMetadataStore:
    """ClickHouseMetadataStore with default settings."""
    return ClickHouseMetadataStore(clickhouse_db)


@pytest.fixture
def clickhouse_store_no_autocreate(clickhouse_db: str) -> ClickHouseMetadataStore:
    """ClickHouseMetadataStore with auto_create_tables=False."""
    return ClickHouseMetadataStore(clickhouse_db, auto_create_tables=False)


@pytest.mark.ibis
@pytest.mark.native
@pytest.mark.clickhouse
class TestClickHouse(
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    IbisMapTests,
    MapDtypeTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
):
    @pytest.fixture
    def store(self, request: pytest.FixtureRequest) -> MetadataStore:
        connection_string = request.getfixturevalue("clickhouse_db")
        return ClickHouseMetadataStore(
            connection_string=connection_string,
            hash_algorithm=HashAlgorithm.XXHASH64,
        )

    @pytest.fixture
    def named_store(self, request: pytest.FixtureRequest) -> MetadataStore:
        connection_string = request.getfixturevalue("clickhouse_db")
        return ClickHouseMetadataStore(
            connection_string=connection_string,
            hash_algorithm=HashAlgorithm.XXHASH64,
            name="clickhouse-test",
        )


def test_clickhouse_table_naming(
    clickhouse_store: ClickHouseMetadataStore, test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test that feature keys are converted to table names correctly."""
    with clickhouse_store.open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(test_features["UpstreamFeatureA"], metadata)

        # Check table was created with correct name using Ibis
        table_names = store.conn.list_tables()
        assert "test_stores__upstream_a" in table_names


def test_clickhouse_uses_ibis_backend(
    clickhouse_store: ClickHouseMetadataStore, test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test that ClickHouse store uses Ibis backend."""
    with clickhouse_store as store:
        # Should have conn
        assert hasattr(store, "conn")
        # Backend should be clickhouse
        assert store._conn is not None


def test_clickhouse_conn_property_enforcement(
    clickhouse_store: ClickHouseMetadataStore, test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test that conn property enforces store is open."""
    from metaxy.metadata_store import StoreNotOpenError

    # Should raise when accessing conn while closed (Ibis error message)
    with pytest.raises(StoreNotOpenError, match="Ibis connection is not open"):
        _ = clickhouse_store.conn

    # Should work when open
    with clickhouse_store.open():
        conn = clickhouse_store.conn
        assert conn is not None

    with clickhouse_store.open("w"):
        conn = clickhouse_store.conn
        assert conn is not None


def test_clickhouse_persistence(
    clickhouse_db: str, test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test that data persists across different store instances."""

    # Write data in first instance
    with ClickHouseMetadataStore(clickhouse_db).open("w") as store1:
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
        store1.write(test_features["UpstreamFeatureA"], metadata)

    # Read data in second instance
    with ClickHouseMetadataStore(clickhouse_db) as store2:
        result = collect_to_polars(store2.read(test_features["UpstreamFeatureA"]))

        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_clickhouse_hash_algorithms(
    clickhouse_db: str, test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test that ClickHouse supports MD5, XXH3_64, and XXHASH hash algorithms.

    Args:
        clickhouse_db: Connection string fixture
        test_graph: Feature graph fixture (for context)
        test_features: Dict with test feature classes
    """

    from metaxy.versioning.types import HashAlgorithm

    # Test each supported algorithm
    for algorithm in [
        HashAlgorithm.MD5,
        HashAlgorithm.XXH3_64,
        HashAlgorithm.XXHASH32,
        HashAlgorithm.XXHASH64,
    ]:
        with ClickHouseMetadataStore(clickhouse_db, hash_algorithm=algorithm).open("w") as store:
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
            store.write(test_features["UpstreamFeatureA"], metadata)

            result = collect_to_polars(store.read(test_features["UpstreamFeatureA"]))
            assert len(result) == 2


def test_clickhouse_xxh3_hash_functions_build_native_expressions() -> None:
    """XXH3 hash functions should build native ClickHouse Ibis expressions."""
    import ibis
    from metaxy.config import MetaxyConfig

    with MetaxyConfig().use():
        store = ClickHouseMetadataStore("clickhouse://localhost/default")

    hash_functions = store._create_hash_functions()
    sample = ibis.literal("sample")

    assert str(hash_functions[HashAlgorithm.XXH3_64](sample))


def test_clickhouse_config_instantiation(
    clickhouse_db: str, test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test instantiating ClickHouse store via MetaxyConfig."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "clickhouse_store": StoreConfig(
                type="metaxy.ext.clickhouse.ClickHouseMetadataStore",
                config={
                    "connection_string": clickhouse_db,
                },
            )
        }
    )

    store = config.get_store("clickhouse_store")
    assert isinstance(store, ClickHouseMetadataStore)
    assert store.hash_algorithm == HashAlgorithm.XXHASH32

    # Verify store can be opened
    with store.open("w"):
        assert store._is_open


def test_clickhouse_config_with_connection_params(test_graph, test_features: dict[str, FeatureDefinition]) -> None:
    """Test ClickHouse store config with connection_params."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "clickhouse_store": StoreConfig(
                type="metaxy.ext.clickhouse.ClickHouseMetadataStore",
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
    clickhouse_db: str, test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test ClickHouse store config with specific hash algorithm."""
    from metaxy.config import MetaxyConfig, StoreConfig
    from metaxy.versioning.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "clickhouse_store": StoreConfig(
                type="metaxy.ext.clickhouse.ClickHouseMetadataStore",
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

    with store.open("w"):
        assert store._is_open


def test_clickhouse_config_with_fallback_stores(
    clickhouse_db: str, test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test ClickHouse store config with fallback stores."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.ext.clickhouse.ClickHouseMetadataStore",
                config={
                    "connection_string": clickhouse_db,
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.ext.clickhouse.ClickHouseMetadataStore",
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

    with dev_store.open("w"):
        assert dev_store._is_open


def test_clickhouse_json_column_type(
    clickhouse_store_no_autocreate: ClickHouseMetadataStore,
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
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
    feature_key = feature_cls.spec.key

    with clickhouse_store_no_autocreate as store:
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
                metaxy_project_version String,
                metaxy_data_version_by_field JSON,
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_updated_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String            ) ENGINE = MergeTree()
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
                metaxy_project_version,
                metaxy_data_version_by_field,
                metaxy_data_version,
                metaxy_created_at,
                metaxy_deleted_at,
                metaxy_materialization_id
            ) VALUES
            (1, '{provenance_json}', 'prov1', 'v1', 'sv1', '{version_json}', 'dv1', now(), NULL, 'm1'),
            (2, '{provenance_json}', 'prov2', 'v1', 'sv1', '{version_json}', 'dv2', now(), NULL, 'm1')
        """
        )

        # Read via _read_feature (no feature_version filter)
        # This uses transform_after_read internally
        # Without the fix, this would raise:
        # "pyarrow.lib.ArrowTypeError: Expected bytes, got a 'dict' object"
        read_result = store._read_feature(feature_cls)
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
    clickhouse_store_no_autocreate: ClickHouseMetadataStore,
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    """Test that ClickHouse Map(String, String) columns are read back as native Map.

    When tables have Map(String, String) columns (common in ClickHouse for
    key-value data), the store reads them back as native Map columns, which
    materialize as ``polars_map.Map`` in Polars.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec.key

    with clickhouse_store_no_autocreate as store:
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
                metaxy_project_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_updated_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String            ) ENGINE = MergeTree()
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
                metaxy_project_version,
                metaxy_data_version_by_field,
                metaxy_data_version,
                metaxy_created_at,
                metaxy_deleted_at,
                metaxy_materialization_id
            ) VALUES
            (1, {{'frames': 'hash1', 'audio': 'hash2'}}, 'prov1', 'v1', 'sv1', {{'frames': 'v1', 'audio': 'v1'}}, 'dv1', now(), NULL, 'm1'),
            (2, {{'frames': 'hash3', 'audio': 'hash4'}}, 'prov2', 'v1', 'sv1', {{'frames': 'v2', 'audio': 'v2'}}, 'dv2', now(), NULL, 'm1')
        """
        )

        # Read via _read_feature. Map columns come back as native Map.
        read_result = store._read_feature(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result).sort("sample_uid")

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}

        # Map columns materialize as polars_map.Map
        assert result.schema["metaxy_provenance_by_field"] == Map(pl.String(), pl.String())
        frames = result["metaxy_provenance_by_field"].map.get("frames").to_list()  # ty: ignore[unresolved-attribute]
        audio = result["metaxy_provenance_by_field"].map.get("audio").to_list()  # ty: ignore[unresolved-attribute]
        assert frames == ["hash1", "hash3"]
        assert audio == ["hash2", "hash4"]

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_map_column_empty_table_read(
    clickhouse_store_no_autocreate: ClickHouseMetadataStore,
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    """Test reading from an EMPTY table with Map(String, String) columns.

    Reading an empty table with native Map columns should return an empty
    result (or None) without raising, with the Map columns read back as
    native Map.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec.key

    with clickhouse_store_no_autocreate as store:
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
                metaxy_project_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_updated_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String            ) ENGINE = MergeTree()
            ORDER BY sample_uid
        """
        )

        # Reading from the empty table should not raise, even though the Map is empty
        read_result = store._read_feature(feature_cls)

        # Reading from an empty table should return None or empty result
        if read_result is not None:
            result = collect_to_polars(read_result)
            assert len(result) == 0

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_map_column_resolve_update_write(
    clickhouse_store_no_autocreate: ClickHouseMetadataStore,
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    """Test resolve_update and write with Map(String, String) columns.

    This tests the full workflow that was failing in production:
    1. Create table with Map columns
    2. Call resolve_update with Polars DataFrame (has Struct columns)
    3. Call write which should transform Struct -> JSON for Map insertion
    4. Read back and verify data

    The key issue was that Polars Struct -> pl.Object conversion failed because
    Ibis doesn't know how to handle pl.Object type.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec.key

    with clickhouse_store_no_autocreate.open("w") as store:
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
                metaxy_project_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_updated_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String            ) ENGINE = MergeTree()
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
        assert len(increment.new) == 3
        assert len(increment.stale) == 0
        assert len(increment.orphaned) == 0

        # write should work (Struct -> JSON string for Map columns)
        # This is where the original error occurred: KeyError: Object
        store.write(feature_cls, samples)

        # Read back and verify
        read_result = store._read_feature(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result).sort("sample_uid")

        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}

        # Map columns are read back as native Map
        assert result.schema["metaxy_provenance_by_field"] == Map(pl.String(), pl.String())
        frames = result["metaxy_provenance_by_field"].map.get("frames").to_list()  # ty: ignore[unresolved-attribute]
        assert frames == ["hash1", "hash3", "hash5"]

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_map_column_write_from_ibis_struct(
    clickhouse_store_no_autocreate: ClickHouseMetadataStore,
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    """Test writing an Ibis-backed DataFrame with Map columns to Map columns.

    This tests the scenario where metadata is computed using Ibis (e.g., from
    a SQL query or join), and metaxy_provenance_by_field / metaxy_data_version_by_field
    are built as native Map columns via the versioning engine's build_map_column.
    The Ibis Map columns are written to ClickHouse Map columns and read back as
    native Map.
    """
    import ibis
    import narwhals as nw
    from metaxy.models.constants import (
        METAXY_DATA_VERSION_BY_FIELD,
        METAXY_PROVENANCE_BY_FIELD,
    )

    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec.key
    plan = test_graph.get_feature_plan(feature_key)

    with clickhouse_store_no_autocreate.open("w") as store:
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
                metaxy_project_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_updated_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String            ) ENGINE = MergeTree()
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

        # Wrap in Narwhals and use the actual versioning engine to build the Map columns
        nw_df = nw.from_native(ibis_table, eager_only=False)

        # Use the store's versioning engine to build the Map columns.
        # This is exactly how resolve_update builds metaxy_provenance_by_field.
        with store.create_versioning_engine(plan, implementation=nw.Implementation.IBIS) as engine:
            # Build the Map columns using the engine's method (same as production code)
            nw_df = engine.build_map_column(
                nw_df,
                METAXY_PROVENANCE_BY_FIELD,
                {"frames": "_hash_frames", "audio": "_hash_audio"},
            )
            nw_df = engine.build_map_column(
                nw_df,
                METAXY_DATA_VERSION_BY_FIELD,
                {"frames": "_hash_frames", "audio": "_hash_audio"},
            )

        # Drop temporary columns
        nw_df = nw_df.drop("_hash_frames", "_hash_audio")

        # Verify it's still Ibis-backed
        assert nw_df.implementation == nw.Implementation.IBIS

        # Write the Ibis-backed Map columns to the store
        store.write(feature_cls, nw_df)

        # Read back and verify
        read_result = store._read_feature(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result).sort("sample_uid")

        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}

        # Map columns are read back as native Map
        assert result.schema["metaxy_provenance_by_field"] == Map(pl.String(), pl.String())
        prov = [{e["key"]: e["value"] for e in row} for row in result["metaxy_provenance_by_field"].to_list()]
        assert [p["frames"] for p in prov] == ["hash1", "hash2", "hash3"]
        assert [p["audio"] for p in prov] == ["hash_a1", "hash_a2", "hash_a3"]

        # Clean up
        conn.drop_table(table_name)


def test_clickhouse_user_defined_map_column(
    clickhouse_store_no_autocreate: ClickHouseMetadataStore,
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    """Test that user-defined Map(String, T) columns are read back as native Map.

    Users may define their own Map columns in ClickHouse tables. Both metaxy's
    system columns (metaxy_provenance_by_field, metaxy_data_version_by_field) and
    user-defined Map columns are read back as native Map, materialized as
    ``polars_map.Map`` in Polars.

    This test verifies:
    1. User Map columns are readable (no Ibis/PyArrow errors)
    2. User Map columns materialize as polars_map.Map
    3. Metaxy Map columns materialize as polars_map.Map
    """
    feature_cls = test_features["UpstreamFeatureA"]
    feature_key = feature_cls.spec.key

    with clickhouse_store_no_autocreate as store:
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
                metaxy_project_version String,
                metaxy_data_version_by_field Map(String, String),
                metaxy_data_version String,
                metaxy_created_at DateTime64(6, 'UTC'),
                metaxy_updated_at DateTime64(6, 'UTC'),
                metaxy_deleted_at Nullable(DateTime64(6, 'UTC')),
                metaxy_materialization_id String            ) ENGINE = MergeTree()
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
                metaxy_project_version,
                metaxy_data_version_by_field,
                metaxy_data_version,
                metaxy_created_at,
                metaxy_materialization_id
            ) VALUES
            (1, {{'source': 'camera1', 'quality': 'high'}}, {{'frames': 'hash1', 'audio': 'hash2'}}, 'prov1', 'v1', 'sv1', {{'frames': 'v1', 'audio': 'v1'}}, 'dv1', now(), 'm1'),
            (2, {{'source': 'camera2', 'resolution': '4k'}}, {{'frames': 'hash3', 'audio': 'hash4'}}, 'prov2', 'v1', 'sv1', {{'frames': 'v2', 'audio': 'v2'}}, 'dv2', now(), 'm1')
        """
        )

        # Read via _read_feature. Both metaxy and user Map columns come back as native Map.
        read_result = store._read_feature(feature_cls)
        assert read_result is not None
        result = collect_to_polars(read_result).sort("sample_uid")

        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}

        # Metaxy Map columns materialize as polars_map.Map
        assert result.schema["metaxy_provenance_by_field"] == Map(pl.String(), pl.String())
        frames = result["metaxy_provenance_by_field"].map.get("frames").to_list()  # ty: ignore[unresolved-attribute]
        assert frames == ["hash1", "hash3"]

        # User Map column also materializes as polars_map.Map
        assert result.schema["user_metadata"] == Map(pl.String(), pl.String())
        sources = result["user_metadata"].map.get("source").to_list()  # ty: ignore[unresolved-attribute]
        assert sources == ["camera1", "camera2"]

        # Clean up
        conn.drop_table(table_name)


# -- ibis_type_to_polars override (no live ClickHouse needed) --


@pytest.fixture
def ch_store() -> ClickHouseMetadataStore:
    """ClickHouseMetadataStore for type conversion tests (no connection needed)."""
    return ClickHouseMetadataStore(connection_string="clickhouse://localhost:8443/default")


def test_clickhouse_ibis_type_to_polars_map(ch_store: ClickHouseMetadataStore) -> None:
    result = ch_store.ibis_type_to_polars(dt.Map(key_type=dt.String(), value_type=dt.Int64()))
    assert result == pl.List(pl.Struct({"key": pl.String, "value": pl.Int64}))


def test_clickhouse_ibis_type_to_polars_nested_map(ch_store: ClickHouseMetadataStore) -> None:
    nested = dt.Map(
        key_type=dt.String(),
        value_type=dt.Map(key_type=dt.String(), value_type=dt.Float64()),
    )
    inner = pl.List(pl.Struct({"key": pl.String, "value": pl.Float64}))
    assert ch_store.ibis_type_to_polars(nested) == pl.List(pl.Struct({"key": pl.String, "value": inner}))


def test_clickhouse_ibis_type_to_polars_delegates_non_map(ch_store: ClickHouseMetadataStore) -> None:
    assert ch_store.ibis_type_to_polars(dt.String()) == pl.String
    assert ch_store.ibis_type_to_polars(dt.UUID()) == pl.String
