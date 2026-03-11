"""DuckDB-specific tests that don't apply to other stores."""

from pathlib import Path
from typing import Any

import ibis.expr.types as ir
import polars as pl
import pytest

# Skip all tests in this module if DuckDB not available

pytest.importorskip("pyarrow")

from metaxy._utils import collect_to_polars
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY, SystemTableStorage
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD


def test_duckdb_table_naming(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Test that feature keys are converted to table names correctly.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """
    db_path = tmp_path / "test.duckdb"

    with DuckDBMetadataStore(db_path, auto_create_tables=True).open("w") as store:
        import polars as pl

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


def test_duckdb_table_prefix_applied(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Prefix should apply to feature and system tables."""
    db_path = tmp_path / "prefixed.duckdb"
    table_prefix = "prod_v2_"
    feature = test_features["UpstreamFeatureA"]

    with DuckDBMetadataStore(db_path, auto_create_tables=True, table_prefix=table_prefix).open("w") as store:
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


def test_duckdb_with_custom_config(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Test creating DuckDB store with custom configuration.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """
    db_path = tmp_path / "test.duckdb"

    config: dict[str, str] = {
        "threads": "2",
        "memory_limit": "1GB",
    }

    with DuckDBMetadataStore(db_path, config=config, auto_create_tables=True) as store:
        # Just verify store opens successfully with config
        assert store._is_open
        assert store.backend == "duckdb"


def test_duckdb_uses_ibis_backend(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Test that DuckDB store uses Ibis backend.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """
    db_path = tmp_path / "test.duckdb"

    with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
        # Should have conn
        assert hasattr(store, "conn")
        # Backend should be duckdb
        assert store.backend == "duckdb"


def test_duckdb_conn_property_enforcement(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Test that conn property enforces store is open.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """
    from metaxy.metadata_store import StoreNotOpenError

    db_path = tmp_path / "test.duckdb"
    store = DuckDBMetadataStore(db_path)

    # Should raise when accessing conn while closed (Ibis error message)
    with pytest.raises(StoreNotOpenError, match="Ibis connection is not open"):
        _ = store.conn

    # Should work when open
    with store.open("w"):
        conn = store.conn
        assert conn is not None


def test_duckdb_persistence_across_instances(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Test that data persists across different store instances.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """

    db_path = tmp_path / "test.duckdb"

    # Write data in first instance
    with DuckDBMetadataStore(db_path, auto_create_tables=True).open("w") as store1:
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
    with DuckDBMetadataStore(db_path, auto_create_tables=True) as store2:
        result = collect_to_polars(store2.read(test_features["UpstreamFeatureA"]))

        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_duckdb_in_memory_nested_write_from_read_mode(
    test_graph,
    test_features: dict[str, Any],
) -> None:
    """Regression test for issue #1016 without Dagster."""
    feature = test_features["UpstreamFeatureA"]
    store = DuckDBMetadataStore(database=":memory:", auto_create_tables=True)

    with store:
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


def test_duckdb_resolve_update_collects_with_active_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_graph,
    test_features: dict[str, Any],
) -> None:
    """Eager resolve_update should not depend on Ibis default backend lookup."""
    upstream = test_features["UpstreamFeatureA"]
    downstream = test_features["DownstreamFeature"]
    store = DuckDBMetadataStore(tmp_path / "resolve-update.duckdb", auto_create_tables=True)

    upstream_metadata = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field": [
                {"frames": "hash_f1", "audio": "hash_a1"},
                {"frames": "hash_f2", "audio": "hash_a2"},
                {"frames": "hash_f3", "audio": "hash_a3"},
            ],
        }
    )

    with store.open("w"):
        store.write(upstream, upstream_metadata)

        original_find_backend = ir.Expr._find_backend

        def fail_default_backend_lookup(self, *, use_default=False):  # type: ignore[no-untyped-def]
            if use_default:
                raise AssertionError("resolve_update eager collection should use the active store backend")
            return original_find_backend(self, use_default=use_default)

        monkeypatch.setattr(ir.Expr, "_find_backend", fail_default_backend_lookup)

        increment = store.resolve_update(downstream, lazy=False)

    result = increment.new.to_polars().sort("sample_uid")
    assert result["sample_uid"].to_list() == [1, 2, 3]


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
    # Avoid opening a real backend connection; this test only validates selection logic in DuckDBMetadataStore._open.
    monkeypatch.setattr(IbisMetadataStore, "_open", lambda self, mode: None)
    monkeypatch.setattr(DuckDBMetadataStore, "_load_extensions", lambda self: None)

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


def test_duckdb_ducklake_integration(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Attach DuckLake using local DuckDB storage and DuckDB metadata."""
    from metaxy.ext.metadata_stores.ducklake import DuckLakeConfig

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
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "duckdb_store": StoreConfig(
                type="metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
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
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "duckdb_store": StoreConfig(
                type="metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
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


def test_duckdb_config_with_extension_specs() -> None:
    """DuckDB store config should accept serialized ExtensionSpec mappings."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "duckdb_store": StoreConfig(
                type="metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "extensions": [
                        {
                            "name": "hashfuncs",
                            "repository": "community",
                        }
                    ],
                },
            )
        }
    )

    store = config.get_store("duckdb_store")
    assert isinstance(store, DuckDBMetadataStore)
    assert ("hashfuncs", "community") in [(ext.name, ext.repository) for ext in store.extensions]


def test_duckdb_config_with_hash_algorithm() -> None:
    """Test DuckDB store config with specific hash algorithm."""
    from metaxy.config import MetaxyConfig, StoreConfig
    from metaxy.versioning.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "duckdb_store": StoreConfig(
                type="metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
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
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
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


def test_motherduck_ducklake_connects_via_memory_and_attaches() -> None:
    """MotherDuck stores connect via :memory: and ATTACH md: to allow hashfuncs to load first.

    duckdb.connect("md:...") activates MotherDuck immediately, preventing community
    extensions from being loaded.  Instead we connect to :memory:, load hashfuncs,
    then ATTACH the MotherDuck database via init_sql on the motherduck extension.
    """
    from metaxy.ext.metadata_stores.ducklake import DuckLakeConfig

    store = DuckDBMetadataStore(
        "md:my_database?motherduck_token=fake_token",
        ducklake=DuckLakeConfig.model_validate({"catalog": {"type": "motherduck", "database": "my_database"}}),
    )

    # Connection must go to :memory:, not md: directly
    assert store.connection_params["database"] == ":memory:"

    ext_names = [ext.name for ext in store.extensions]
    assert "hashfuncs" in ext_names
    assert "motherduck" in ext_names
    assert ext_names.index("hashfuncs") < ext_names.index("motherduck"), (
        f"hashfuncs must come before motherduck, got order: {ext_names}"
    )

    # motherduck init_sql must set the token and ATTACH md:
    md_ext = next(e for e in store.extensions if e.name == "motherduck")
    init_sql = list(md_ext.init_sql)
    assert any("SET motherduck_token" in s for s in init_sql), f"expected SET motherduck_token in init_sql: {init_sql}"
    assert any("ATTACH 'md:'" in s for s in init_sql), f"expected ATTACH 'md:' in init_sql: {init_sql}"


def test_xxh32_available_after_memory_attach_and_use(tmp_path: Path) -> None:
    """xxh32 must work when hashfuncs is loaded before motherduck via :memory: + ATTACH.

    Simulates the exact sequence metaxy uses: connect to :memory:, load hashfuncs,
    load motherduck, ATTACH an external db, USE it — xxh32 must remain callable.
    """
    import duckdb

    other_db = tmp_path / "other.db"
    conn = duckdb.connect(":memory:")

    # Load hashfuncs BEFORE any external database is attached
    conn.install_extension("hashfuncs", repository="community")
    conn.load_extension("hashfuncs")

    # Simulate ATTACH + USE (what MotherDuck init_sql + configure() does)
    conn.execute(f"ATTACH '{other_db}' AS otherdb")
    conn.execute("USE otherdb")

    result = conn.execute("SELECT xxh32('hello'::VARCHAR)").fetchone()
    assert result is not None
    assert isinstance(result[0], int), f"expected int hash, got {result[0]!r}"


def test_motherduck_ducklake_open_loads_extensions_once_before_configure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MotherDuck DuckLake loads extensions once, before configure() executes USE <db>.

    hashfuncs is inserted before motherduck in the extension list so it is already
    available in the switched database context — no second reload needed.
    """
    from metaxy.ext.metadata_stores.ducklake import DuckLakeConfig

    events: list[str] = []

    monkeypatch.setattr(IbisMetadataStore, "_open", lambda self, mode: None)
    monkeypatch.setattr(DuckDBMetadataStore, "_duckdb_raw_connection", lambda self: object())
    monkeypatch.setattr(DuckDBMetadataStore, "_load_extensions", lambda self: events.append("load_extensions"))

    store = DuckDBMetadataStore(
        database="md:?motherduck_token=dummy",
        ducklake=DuckLakeConfig.model_validate(
            {
                "catalog": {"type": "motherduck", "database": "my_lake"},
            }
        ),
    )

    def fake_configure(conn) -> None:
        events.append("configure")

    monkeypatch.setattr(store.ducklake_attachment, "configure", fake_configure)

    store._open("w")

    assert events == ["load_extensions", "configure"]


def test_non_motherduck_ducklake_open_loads_extensions_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-managed DuckLake should keep the original single extension load."""
    from metaxy.ext.metadata_stores.ducklake import DuckLakeConfig

    events: list[str] = []

    monkeypatch.setattr(IbisMetadataStore, "_open", lambda self, mode: None)
    monkeypatch.setattr(DuckDBMetadataStore, "_duckdb_raw_connection", lambda self: object())
    monkeypatch.setattr(DuckDBMetadataStore, "_load_extensions", lambda self: events.append("load_extensions"))

    store = DuckDBMetadataStore(
        database=":memory:",
        ducklake=DuckLakeConfig.model_validate(
            {
                "catalog": {"type": "duckdb", "uri": str(tmp_path / "catalog.duckdb")},
                "storage": {"type": "local", "path": str(tmp_path / "storage")},
            }
        ),
    )

    def fake_configure(conn) -> None:
        events.append("configure")

    monkeypatch.setattr(store.ducklake_attachment, "configure", fake_configure)

    store._open("w")

    assert events == ["load_extensions", "configure"]
