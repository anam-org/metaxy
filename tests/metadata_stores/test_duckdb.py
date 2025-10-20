"""DuckDB-specific tests that don't apply to other stores."""

from pathlib import Path

import pytest

# Skip all tests in this module if DuckDB not available
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from metaxy.metadata_store.duckdb import DuckDBMetadataStore


def test_duckdb_table_naming(
    tmp_path: Path, test_registry, test_features: dict
) -> None:
    """Test that feature keys are converted to table names correctly.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_registry: Registry with test features
    """
    db_path = tmp_path / "test.duckdb"

    with DuckDBMetadataStore(db_path) as store:
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


def test_duckdb_with_custom_config(
    tmp_path: Path, test_registry, test_features: dict
) -> None:
    """Test creating DuckDB store with custom configuration.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_registry: Registry with test features
    """
    db_path = tmp_path / "test.duckdb"

    config: dict[str, str] = {
        "threads": "2",
        "memory_limit": "1GB",
    }

    with DuckDBMetadataStore(db_path, config=config) as store:
        # Just verify store opens successfully with config
        assert store._is_open
        assert store.backend == "duckdb"


def test_duckdb_uses_ibis_backend(
    tmp_path: Path, test_registry, test_features: dict
) -> None:
    """Test that DuckDB store uses Ibis backend.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_registry: Registry with test features
    """
    db_path = tmp_path / "test.duckdb"

    with DuckDBMetadataStore(db_path) as store:
        # Should have ibis_conn
        assert hasattr(store, "ibis_conn")
        # Backend should be duckdb
        assert store.backend == "duckdb"


def test_duckdb_conn_property_enforcement(
    tmp_path: Path, test_registry, test_features: dict
) -> None:
    """Test that conn property enforces store is open.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_registry: Registry with test features
    """
    from metaxy.metadata_store import StoreNotOpenError

    db_path = tmp_path / "test.duckdb"
    store = DuckDBMetadataStore(db_path)

    # Should raise when accessing conn while closed (Ibis error message)
    with pytest.raises(StoreNotOpenError, match="Ibis connection is not open"):
        _ = store.conn

    # Should work when open
    with store:
        conn = store.conn
        assert conn is not None


def test_duckdb_persistence_across_instances(
    tmp_path: Path, test_registry, test_features: dict
) -> None:
    """Test that data persists across different store instances.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_registry: Registry with test features
    """
    import polars as pl

    db_path = tmp_path / "test.duckdb"

    # Write data in first instance
    with DuckDBMetadataStore(db_path) as store1:
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
    with DuckDBMetadataStore(db_path) as store2:
        result = store2.read_metadata(test_features["UpstreamFeatureA"])

        assert len(result) == 3
        assert set(result["sample_id"].to_list()) == {1, 2, 3}


def test_duckdb_close_idempotent(
    tmp_path: Path, test_registry, test_features: dict
) -> None:
    """Test that close() can be called multiple times safely.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_registry: Registry with test features
    """
    db_path = tmp_path / "test.duckdb"
    store = DuckDBMetadataStore(db_path)

    with store:
        pass

    # Close again manually (should not raise)
    store.close()
    store.close()
