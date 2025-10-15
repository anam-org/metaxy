"""SQLite-specific tests that don't apply to other stores."""

from pathlib import Path

import polars as pl
import pytest

# Skip all tests in this module if Ibis not available
pytest.importorskip("ibis")

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.sqlite import SQLiteMetadataStore


def test_sqlite_table_naming(tmp_path: Path, test_graph, test_features: dict) -> None:
    """Test that feature keys are converted to table names correctly.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
        test_features: Dict of test feature classes
    """
    db_path = tmp_path / "test.sqlite"

    with SQLiteMetadataStore(db_path) as store:
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


def test_sqlite_uses_ibis_backend(
    tmp_path: Path, test_graph, test_features: dict
) -> None:
    """Test that SQLite store uses Ibis backend.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """
    db_path = tmp_path / "test.sqlite"

    with SQLiteMetadataStore(db_path) as store:
        # Should have ibis_conn
        assert hasattr(store, "ibis_conn")
        # Backend should be sqlite
        assert store.backend == "sqlite"


def test_sqlite_conn_property_enforcement(
    tmp_path: Path, test_graph, test_features: dict
) -> None:
    """Test that conn property enforces store is open.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """
    from metaxy.metadata_store import StoreNotOpenError

    db_path = tmp_path / "test.sqlite"
    store = SQLiteMetadataStore(db_path)

    # Should raise when accessing conn while closed (Ibis error message)
    with pytest.raises(StoreNotOpenError, match="Ibis connection is not open"):
        _ = store.conn

    # Should work when open
    with store:
        conn = store.conn
        assert conn is not None


def test_sqlite_persistence_across_instances(
    tmp_path: Path, test_graph, test_features: dict
) -> None:
    """Test that data persists across different store instances.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
        test_features: Dict of test feature classes
    """

    db_path = tmp_path / "test.sqlite"

    # Write data in first instance
    with SQLiteMetadataStore(db_path) as store1:
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
    with SQLiteMetadataStore(db_path) as store2:
        result = collect_to_polars(
            store2.read_metadata(test_features["UpstreamFeatureA"])
        )

        assert len(result) == 3
        assert set(result["sample_id"].to_list()) == {1, 2, 3}


def test_sqlite_in_memory(test_graph, test_features: dict) -> None:
    """Test SQLite in-memory database.

    Args:
        test_graph: Registry with test features
        test_features: Dict of test feature classes
    """

    # Use :memory: for in-memory database
    with SQLiteMetadataStore(":memory:") as store:
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

        result = collect_to_polars(
            store.read_metadata(test_features["UpstreamFeatureA"])
        )
        assert len(result) == 2
        assert set(result["sample_id"].to_list()) == {1, 2}


def test_sqlite_close_idempotent(
    tmp_path: Path, test_graph, test_features: dict
) -> None:
    """Test that close() can be called multiple times safely.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Registry with test features
    """
    db_path = tmp_path / "test.sqlite"
    store = SQLiteMetadataStore(db_path)

    with store:
        pass

    # Close again manually (should not raise)
    store.close()
    store.close()


def test_sqlite_config_instantiation() -> None:
    """Test instantiating SQLite store via MetaxyConfig."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "sqlite_store": StoreConfig(
                type="metaxy.metadata_store.sqlite.SQLiteMetadataStore",
                config={
                    "database": ":memory:",
                },
            )
        }
    )

    store = config.get_store("sqlite_store")
    assert isinstance(store, SQLiteMetadataStore)
    assert store.database == ":memory:"

    # Verify store can be opened
    with store:
        assert store._is_open


def test_sqlite_config_with_hash_algorithm() -> None:
    """Test SQLite store config with specific hash algorithm."""
    from metaxy.config import MetaxyConfig, StoreConfig
    from metaxy.data_versioning.hash_algorithms import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "sqlite_store": StoreConfig(
                type="metaxy.metadata_store.sqlite.SQLiteMetadataStore",
                config={
                    "database": ":memory:",
                    "hash_algorithm": "md5",
                },
            )
        }
    )

    store = config.get_store("sqlite_store")
    assert isinstance(store, SQLiteMetadataStore)
    assert store.hash_algorithm == HashAlgorithm.MD5

    with store:
        assert store._is_open


def test_sqlite_config_with_fallback_stores() -> None:
    """Test SQLite store config with fallback stores."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.sqlite.SQLiteMetadataStore",
                config={
                    "database": ":memory:",
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.sqlite.SQLiteMetadataStore",
                config={
                    "database": ":memory:",
                },
            ),
        }
    )

    dev_store = config.get_store("dev")
    assert isinstance(dev_store, SQLiteMetadataStore)
    assert len(dev_store.fallback_stores) == 1
    assert isinstance(dev_store.fallback_stores[0], SQLiteMetadataStore)

    with dev_store:
        assert dev_store._is_open
