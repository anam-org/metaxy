"""DuckDB-specific tests that don't apply to other stores."""

from pathlib import Path
from typing import Any

import polars as pl
import pytest

# Skip all tests in this module if DuckDB not available

pytest.importorskip("pyarrow")

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY
from metaxy.metadata_store.types import AccessMode
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD


def test_duckdb_table_naming(
    tmp_path: Path, test_graph, test_features: dict[str, Any]
) -> None:
    """Test that feature keys are converted to table names correctly.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """
    db_path = tmp_path / "test.duckdb"

    with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
        import polars as pl

        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], metadata)

        # Check table was created with correct name using Ibis
        table_names = store.ibis_conn.list_tables()
        assert "test_stores__upstream_a" in table_names


def test_duckdb_table_prefix_applied(
    tmp_path: Path, test_graph, test_features: dict[str, Any]
) -> None:
    """Prefix should apply to feature and system tables."""
    db_path = tmp_path / "prefixed.duckdb"
    table_prefix = "prod_v2_"
    feature = test_features["UpstreamFeatureA"]

    with DuckDBMetadataStore(
        db_path, auto_create_tables=True, table_prefix=table_prefix
    ) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(feature, metadata)

        expected_feature_table = table_prefix + feature.spec().key.table_name
        expected_system_table = table_prefix + FEATURE_VERSIONS_KEY.table_name

        # Record snapshot to ensure system table is materialized
        store.record_feature_graph_snapshot()

        table_names = set(store.ibis_conn.list_tables())
        assert expected_feature_table in table_names
        assert store.get_table_name(feature.spec().key) == expected_feature_table
        assert store.get_table_name(FEATURE_VERSIONS_KEY) == expected_system_table


def test_duckdb_with_custom_config(
    tmp_path: Path, test_graph, test_features: dict[str, Any]
) -> None:
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


def test_duckdb_uses_ibis_backend(
    tmp_path: Path, test_graph, test_features: dict[str, Any]
) -> None:
    """Test that DuckDB store uses Ibis backend.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """
    db_path = tmp_path / "test.duckdb"

    with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
        # Should have ibis_conn
        assert hasattr(store, "ibis_conn")
        # Backend should be duckdb
        assert store.backend == "duckdb"


def test_duckdb_conn_property_enforcement(
    tmp_path: Path, test_graph, test_features: dict[str, Any]
) -> None:
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
    with store.open(AccessMode.WRITE):
        conn = store.conn
        assert conn is not None


def test_duckdb_persistence_across_instances(
    tmp_path: Path, test_graph, test_features: dict[str, Any]
) -> None:
    """Test that data persists across different store instances.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """

    db_path = tmp_path / "test.duckdb"

    # Write data in first instance
    with DuckDBMetadataStore(db_path, auto_create_tables=True) as store1:
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
    with DuckDBMetadataStore(db_path, auto_create_tables=True) as store2:
        result = collect_to_polars(
            store2.read_metadata(test_features["UpstreamFeatureA"])
        )

        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_duckdb_ducklake_integration(
    tmp_path: Path, test_graph, test_features: dict[str, Any]
) -> None:
    """Attach DuckLake using local DuckDB storage and DuckDB metadata."""

    db_path = tmp_path / "ducklake.duckdb"
    metadata_path = tmp_path / "ducklake_catalog.duckdb"
    storage_dir = tmp_path / "ducklake_storage"

    ducklake_config = {
        "alias": "lake",
        "metadata_backend": {
            "type": "duckdb",
            "path": str(metadata_path),
        },
        "storage_backend": {
            "type": "local",
            "path": str(storage_dir),
        },
    }

    with DuckDBMetadataStore(
        db_path, extensions=["json"], ducklake=ducklake_config, auto_create_tables=True
    ) as store:
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
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
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
    with store.open(AccessMode.WRITE):
        assert store._is_open


def test_duckdb_config_with_extensions() -> None:
    """Test DuckDB store config with extensions."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "duckdb_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "extensions": ["hashfuncs", "json"],
                },
            )
        }
    )

    store = config.get_store("duckdb_store")
    assert isinstance(store, DuckDBMetadataStore)

    # hashfuncs is auto-added, so we should have at least hashfuncs
    from metaxy.metadata_store.duckdb import ExtensionSpec

    extension_names = []
    for ext in store.extensions:
        if isinstance(ext, str):
            extension_names.append(ext)
        elif isinstance(ext, ExtensionSpec):
            extension_names.append(ext.name)
    assert "hashfuncs" in extension_names

    with store.open(AccessMode.WRITE):
        assert store._is_open


def test_duckdb_config_with_hash_algorithm() -> None:
    """Test DuckDB store config with specific hash algorithm."""
    from metaxy.config import MetaxyConfig, StoreConfig
    from metaxy.provenance.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "duckdb_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
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

    with store.open(AccessMode.WRITE):
        assert store._is_open


def test_duckdb_config_with_fallback_stores() -> None:
    """Test DuckDB store config with fallback stores."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
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

    with dev_store.open(AccessMode.WRITE):
        assert dev_store._is_open
