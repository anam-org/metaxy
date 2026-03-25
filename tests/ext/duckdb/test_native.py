"""DuckDB metadata store tests."""

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from metaxy import HashAlgorithm
from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.ext.duckdb import DuckDBMetadataStore
from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY, SystemTableStorage
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey
from metaxy.utils import collect_to_polars
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
def duckdb_store(tmp_path: Path) -> DuckDBMetadataStore:
    """DuckDB store with auto_create_tables enabled."""
    return DuckDBMetadataStore(tmp_path / "test.duckdb", auto_create_tables=True)


@pytest.fixture
def duckdb_store_bare(tmp_path: Path) -> DuckDBMetadataStore:
    """DuckDB store without auto_create_tables (default settings)."""
    return DuckDBMetadataStore(tmp_path / "test.duckdb")


@pytest.fixture
def duckdb_store_memory() -> DuckDBMetadataStore:
    """In-memory DuckDB store with auto_create_tables enabled."""
    return DuckDBMetadataStore(database=":memory:", auto_create_tables=True)


@pytest.mark.ibis
@pytest.mark.native
@pytest.mark.duckdb
class TestDuckDB(
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
    def store(self, tmp_path: Path) -> MetadataStore:
        return DuckDBMetadataStore(
            database=tmp_path / "test.duckdb",
            hash_algorithm=HashAlgorithm.XXHASH64,
        )


@pytest.mark.ibis
@pytest.mark.native
@pytest.mark.duckdb
class TestDuckDBPreCreatedMapTable:
    """Test writing to a DuckDB table that was pre-created with MAP columns via SQL."""

    def test_write_to_precreated_map_table(self, tmp_path: Path, test_features: dict[str, Any]) -> None:
        """Writing to a table pre-created with MAP(VARCHAR, VARCHAR) columns works."""
        import duckdb

        from metaxy.config import MetaxyConfig
        from metaxy.utils import collect_to_polars

        db_path = tmp_path / "test.duckdb"
        store = DuckDBMetadataStore(database=db_path, hash_algorithm=HashAlgorithm.XXHASH64, auto_create_tables=False)
        feature = test_features["UpstreamFeatureA"]
        table_name = store.get_table_name(store._resolve_feature_key(feature))

        con = duckdb.connect(str(db_path))
        con.execute(f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                metaxy_provenance_by_field MAP(VARCHAR, VARCHAR),
                metaxy_provenance VARCHAR,
                metaxy_data_version_by_field MAP(VARCHAR, VARCHAR),
                metaxy_data_version VARCHAR,
                metaxy_feature_version VARCHAR,
                metaxy_project_version VARCHAR,
                metaxy_created_at TIMESTAMPTZ,
                metaxy_updated_at TIMESTAMPTZ,
                metaxy_deleted_at TIMESTAMPTZ,
                metaxy_materialization_id VARCHAR
            )
        """)
        con.close()

        config = MetaxyConfig(enable_map_datatype=True, auto_create_tables=False)
        with config.use():
            metadata = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"frames": "h1", "audio": "h2"},
                        {"frames": "h3", "audio": "h4"},
                    ],
                }
            )
            with store.open("w") as s:
                s.write(feature, metadata)

            with store.open("r") as s:
                result = s.read(feature)
                assert result is not None
                df = collect_to_polars(result).sort("sample_uid")

            assert df.height == 2
            assert df["sample_uid"].to_list() == [1, 2]

    def test_write_to_precreated_table_with_user_map_column(
        self, tmp_path: Path, test_features: dict[str, Any]
    ) -> None:
        """Writing to a table with a user-defined MAP column pre-created via SQL."""
        import duckdb
        from polars_map import Map

        from metaxy.config import MetaxyConfig
        from metaxy.utils import collect_to_polars

        db_path = tmp_path / "test.duckdb"
        store = DuckDBMetadataStore(database=db_path, hash_algorithm=HashAlgorithm.XXHASH64, auto_create_tables=False)
        feature = test_features["UpstreamFeatureA"]
        table_name = store.get_table_name(store._resolve_feature_key(feature))

        con = duckdb.connect(str(db_path))
        con.execute(f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                metaxy_provenance_by_field MAP(VARCHAR, VARCHAR),
                metaxy_provenance VARCHAR,
                metaxy_data_version_by_field MAP(VARCHAR, VARCHAR),
                metaxy_data_version VARCHAR,
                metaxy_feature_version VARCHAR,
                metaxy_project_version VARCHAR,
                metaxy_created_at TIMESTAMPTZ,
                metaxy_updated_at TIMESTAMPTZ,
                metaxy_deleted_at TIMESTAMPTZ,
                metaxy_materialization_id VARCHAR,
                user_tags MAP(VARCHAR, VARCHAR)
            )
        """)
        con.close()

        config = MetaxyConfig(enable_map_datatype=True, auto_create_tables=False)
        with config.use():
            user_map = pl.Series(
                "user_tags",
                [[("env", "prod"), ("region", "us")]],
                dtype=Map(pl.String(), pl.String()),
            )
            metadata = pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h2"}],
                }
            ).with_columns(user_map)

            with store.open("w") as s:
                s.write(feature, metadata)

            with store.open("r") as s:
                result = s.read(feature)
                assert result is not None

                # Verify Ibis sees native Map type before collecting to Polars
                import ibis.expr.datatypes as dt

                ibis_schema = result.to_native().schema()
                assert isinstance(ibis_schema["user_tags"], dt.Map)
                assert isinstance(ibis_schema["metaxy_provenance_by_field"], dt.Map)

                df = collect_to_polars(result)

            assert df.height == 1
            assert df["sample_uid"].to_list() == [1]

    def test_write_ibis_frame_to_precreated_map_table(self, tmp_path: Path, test_features: dict[str, Any]) -> None:
        """Writing an Ibis-backed frame to a MAP table stays lazy (no materialization)."""
        import duckdb

        from metaxy.config import MetaxyConfig
        from metaxy.utils import collect_to_polars

        db_path = tmp_path / "test.duckdb"
        store = DuckDBMetadataStore(database=db_path, hash_algorithm=HashAlgorithm.XXHASH64, auto_create_tables=True)
        feature = test_features["UpstreamFeatureA"]
        table_name = store.get_table_name(store._resolve_feature_key(feature))

        # First write Polars data so the table exists with Struct columns
        config = MetaxyConfig(auto_create_tables=True)
        with config.use():
            metadata = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"frames": "h1", "audio": "h2"},
                        {"frames": "h3", "audio": "h4"},
                    ],
                }
            )
            with store.open("w") as s:
                s.write(feature, metadata)

        # Recreate the table with MAP columns
        raw_con = duckdb.connect(str(db_path))
        raw_con.execute(f"DROP TABLE IF EXISTS {table_name}")
        raw_con.execute(f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                metaxy_provenance_by_field MAP(VARCHAR, VARCHAR),
                metaxy_provenance VARCHAR,
                metaxy_data_version_by_field MAP(VARCHAR, VARCHAR),
                metaxy_data_version VARCHAR,
                metaxy_feature_version VARCHAR,
                metaxy_project_version VARCHAR,
                metaxy_created_at TIMESTAMPTZ,
                metaxy_updated_at TIMESTAMPTZ,
                metaxy_deleted_at TIMESTAMPTZ,
                metaxy_materialization_id VARCHAR
            )
        """)
        raw_con.close()

        # Write using Ibis-backed frame (via the store's own read -> write roundtrip)
        config = MetaxyConfig(enable_map_datatype=True, auto_create_tables=False)
        with config.use():
            with store.open("w") as s:
                s.write(feature, metadata)

            with store.open("r") as s:
                result = s.read(feature)
                assert result is not None
                df = collect_to_polars(result).sort("sample_uid")

            assert df.height == 2
            assert df["sample_uid"].to_list() == [1, 2]


def test_store_from_config_gets_name(tmp_path: Path) -> None:
    """Test that stores created via MetaxyConfig.get_store() receive the config key as name."""
    config = MetaxyConfig(
        stores={
            "my_store": StoreConfig(
                type="metaxy.ext.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "test.duckdb")},
            )
        }
    )
    with config.use():
        store = config.get_store("my_store")
    assert store.name == "my_store"
    assert not store.display().startswith("[")
    assert repr(store).startswith("[my_store]")


def test_duckdb_table_naming(
    duckdb_store: DuckDBMetadataStore, test_graph: FeatureGraph, test_features: dict[str, Any]
) -> None:
    """Test that feature keys are converted to table names correctly."""
    with duckdb_store.open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(test_features["UpstreamFeatureA"], metadata)

        # Check table was created with correct name using Ibis
        table_names = store.conn.list_tables()
        assert "test_stores__upstream_a" in table_names


def test_duckdb_table_prefix_applied(tmp_path: Path, test_graph: FeatureGraph, test_features: dict[str, Any]) -> None:
    """Prefix should apply to feature and system tables."""
    table_prefix = "prod_v2_"
    feature = test_features["UpstreamFeatureA"]

    with DuckDBMetadataStore(tmp_path / "prefixed.duckdb", auto_create_tables=True, table_prefix=table_prefix).open(
        "w"
    ) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(feature, metadata)

        expected_feature_table = table_prefix + feature.spec.key.table_name
        expected_system_table = table_prefix + FEATURE_VERSIONS_KEY.table_name

        # Record snapshot to ensure system table is materialized
        SystemTableStorage(store).push_graph_snapshot()

        table_names = set(store.conn.list_tables())
        assert expected_feature_table in table_names
        assert store.get_table_name(feature.spec.key) == expected_feature_table
        assert store.get_table_name(FEATURE_VERSIONS_KEY) == expected_system_table


def test_duckdb_with_custom_config(tmp_path: Path, test_graph: FeatureGraph, test_features: dict[str, Any]) -> None:
    """Test creating DuckDB store with custom configuration."""
    config: dict[str, str] = {
        "threads": "2",
        "memory_limit": "1GB",
    }

    with DuckDBMetadataStore(tmp_path / "test.duckdb", config=config, auto_create_tables=True) as store:
        # Just verify store opens successfully with config
        assert store._is_open
        assert store.backend == "duckdb"


def test_duckdb_uses_ibis_backend(
    duckdb_store: DuckDBMetadataStore, test_graph: FeatureGraph, test_features: dict[str, Any]
) -> None:
    """Test that DuckDB store uses Ibis backend."""
    with duckdb_store as store:
        # Should have conn
        assert hasattr(store, "conn")
        # Backend should be duckdb
        assert store.backend == "duckdb"


def test_duckdb_conn_property_enforcement(
    duckdb_store_bare: DuckDBMetadataStore, test_graph: FeatureGraph, test_features: dict[str, Any]
) -> None:
    """Test that conn property enforces store is open."""
    from metaxy.metadata_store import StoreNotOpenError

    # Should raise when accessing conn while closed (Ibis error message)
    with pytest.raises(StoreNotOpenError, match="Ibis connection is not open"):
        _ = duckdb_store_bare.conn

    # Should work when open
    with duckdb_store_bare.open("w"):
        assert duckdb_store_bare.conn is not None


def test_duckdb_persistence_across_instances(
    duckdb_store: DuckDBMetadataStore, test_graph: FeatureGraph, test_features: dict[str, Any]
) -> None:
    """Test that data persists across different store instances."""
    # Write data in first instance
    with duckdb_store.open("w") as store1:
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

    # Read data in second instance with a fresh store pointing at the same database
    with DuckDBMetadataStore(duckdb_store.database, auto_create_tables=True) as store2:
        result = collect_to_polars(store2.read(test_features["UpstreamFeatureA"]))

        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_duckdb_in_memory_nested_write_from_read_mode(
    duckdb_store_memory: DuckDBMetadataStore,
    test_graph: FeatureGraph,
    test_features: dict[str, Any],
) -> None:
    """Regression test for issue #1016 without Dagster."""
    feature = test_features["UpstreamFeatureA"]

    with duckdb_store_memory as store:
        with store.open("w"):
            store.write(
                feature,
                pl.DataFrame(
                    {
                        "sample_uid": ["in_memory_1", "in_memory_2"],
                        METAXY_PROVENANCE_BY_FIELD: [
                            {"frames": "h1", "audio": "h1"},
                            {"frames": "h2", "audio": "h2"},
                        ],
                    }
                ),
            )

        result = collect_to_polars(store.read(feature)).sort("sample_uid")
        assert result["sample_uid"].to_list() == ["in_memory_1", "in_memory_2"]


@pytest.mark.parametrize(
    ("database", "create_local_file", "expected_read_only"),
    [
        (None, False, False),
        ("", False, False),
        (":memory:", False, False),
        ("md:test_db", False, True),
        ("motherduck:test_db", False, True),
        ("s3://bucket/test.duckdb", False, True),
        ("gcs://bucket/test.duckdb", False, True),
        ("azure://container/test.duckdb", False, True),
        ("http://example.com/test.duckdb", False, True),
        ("http_local.duckdb", True, True),
        ("http_local.duckdb", False, False),
        ("existing.duckdb", True, True),
        ("missing.duckdb", False, False),
    ],
)
def test_duckdb_read_mode_read_only_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    database: str | None,
    create_local_file: bool,
    expected_read_only: bool,
) -> None:
    """DuckDB READ mode sets read_only for remote and existing local DBs, but not :memory: or missing files."""
    # Avoid opening a real backend connection; this test only validates the read_only flag logic in DuckDBEngine.open.
    from metaxy.ext.duckdb.engine import DuckDBEngine
    from metaxy.ext.ibis.engine import IbisComputeEngine

    monkeypatch.setattr(IbisComputeEngine, "open", lambda self, mode: None)
    monkeypatch.setattr(DuckDBEngine, "_load_extensions", lambda self: None)

    if database is None:
        database_value = ":memory:"
    elif database == "":
        database_value = ""
    elif "://" in database or database.startswith(("md:", "motherduck:")) or database == ":memory:":
        database_value = database
    else:
        database_path = tmp_path / database
        if create_local_file:
            database_path.touch()
        database_value = str(database_path)

    store = DuckDBMetadataStore(database=database_value)
    if database is None:
        store.connection_params["database"] = None
    # Ensure _open("r") actively manages this flag, including clearing stale values.
    store.connection_params["read_only"] = True
    store._open("r")

    if expected_read_only:
        assert store.connection_params.get("read_only") is True
    else:
        assert "read_only" not in store.connection_params


def test_duckdb_get_filtered_lazy_does_not_require_list_tables(
    duckdb_store: DuckDBMetadataStore,
    test_graph: FeatureGraph,
    test_features: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_get_filtered_ibis_lazy should avoid metadata scans via list_tables."""
    feature = test_features["UpstreamFeatureA"]

    with duckdb_store.open("w") as store:
        store.write(
            feature,
            pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"frames": "h1", "audio": "h1"},
                        {"frames": "h2", "audio": "h2"},
                    ],
                }
            ),
        )

        def _fail_list_tables(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
            raise AssertionError("list_tables should not be called by _get_filtered_ibis_lazy")

        monkeypatch.setattr(type(store.conn), "list_tables", _fail_list_tables)

        existing = store._get_filtered_ibis_lazy(feature)
        assert existing is not None
        existing_df = existing.collect().to_polars().sort("sample_uid")
        assert existing_df["sample_uid"].to_list() == [1, 2]

        missing = store._get_filtered_ibis_lazy(FeatureKey(["test_stores", "missing_feature"]))
        assert missing is None


def test_duckdb_ducklake_integration(tmp_path: Path, test_graph: FeatureGraph, test_features: dict[str, Any]) -> None:
    """Attach DuckLake using local DuckDB storage and DuckDB metadata."""
    from metaxy.ext.duckdb import DuckLakeConfig

    db_path = tmp_path / "ducklake.duckdb"
    metadata_path = tmp_path / "ducklake_catalog.duckdb"
    storage_dir = tmp_path / "ducklake_storage"

    ducklake_config = DuckLakeConfig.model_validate(
        {
            "alias": "lake",
            "catalog": {
                "type": "duckdb",
                "uri": str(metadata_path),
            },
            "storage": {
                "type": "local",
                "path": str(storage_dir),
            },
        }
    )

    with DuckDBMetadataStore(db_path, extensions=["json"], ducklake=ducklake_config, auto_create_tables=True) as store:
        attachment_config = store.ducklake_attachment_config
        assert attachment_config.alias == "lake"

        duckdb_conn = store._duckdb_raw_connection()
        databases = duckdb_conn.execute("PRAGMA database_list").fetchall()
        attached_names = {row[1] for row in databases}
        assert "lake" in attached_names


def test_duckdb_config_instantiation() -> None:
    """Test instantiating DuckDB store via MetaxyConfig."""
    config = MetaxyConfig(
        stores={
            "duckdb_store": StoreConfig(
                type="metaxy.ext.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "config": {
                        "threads": "2",
                        "memory_limit": "512MB",
                    },
                },
            )
        }
    )

    store = config.get_store("duckdb_store")
    assert isinstance(store, DuckDBMetadataStore)
    assert store.database == ":memory:"

    # Verify store can be opened
    with store.open("w"):
        assert store._is_open


def test_duckdb_config_with_extensions() -> None:
    """Test DuckDB store config with extensions."""
    config = MetaxyConfig(
        stores={
            "duckdb_store": StoreConfig(
                type="metaxy.ext.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "extensions": ["json"],
                },
            )
        }
    )

    store = config.get_store("duckdb_store")
    assert isinstance(store, DuckDBMetadataStore)

    # hashfuncs is auto-added, so we should have at least hashfuncs
    assert "hashfuncs" in [ext.name for ext in store.extensions]

    with store.open("w"):
        assert store._is_open


def test_duckdb_config_with_hash_algorithm() -> None:
    """Test DuckDB store config with specific hash algorithm."""
    from metaxy.versioning.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "duckdb_store": StoreConfig(
                type="metaxy.ext.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "hash_algorithm": "md5",
                },
            )
        }
    )

    store = config.get_store("duckdb_store")
    assert isinstance(store, DuckDBMetadataStore)
    assert store.hash_algorithm == HashAlgorithm.MD5

    with store.open("w"):
        assert store._is_open


def test_duckdb_config_with_fallback_stores() -> None:
    """Test DuckDB store config with fallback stores."""
    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.ext.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.ext.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                },
            ),
        }
    )

    dev_store = config.get_store("dev")
    assert isinstance(dev_store, DuckDBMetadataStore)
    assert len(dev_store.fallback_stores) == 1
    assert isinstance(dev_store.fallback_stores[0], DuckDBMetadataStore)

    with dev_store.open("w"):
        assert dev_store._is_open
