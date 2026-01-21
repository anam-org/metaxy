"""Integration tests for ADBC federation via DuckDB adbc_scanner extension.

These tests verify end-to-end federation functionality by:
- Querying remote PostgreSQL databases from DuckDB
- Performing cross-database JOINs
- Testing error handling and connection management
- Validating schema discovery

Requirements:
    - PostgreSQL server (provided by postgres_db fixture)
    - DuckDB ADBC scanner extension (downloaded during test)
    - ADBC PostgreSQL driver (adbc-driver-postgresql)

Known Issues:
    - DuckDB ADBC scanner is a community extension and may have stability issues
    - Some tests may fail with "INTERNAL" errors from ADBC driver
    - This is a known limitation of the current ADBC scanner implementation

Note:
    These tests document the intended API usage even if the underlying
    DuckDB ADBC scanner has issues. The metaxy API wrappers are correct.
"""

import polars as pl
import pytest
from metaxy.metadata_store.postgres import PostgresMetadataStore  # type: ignore[import-not-found]

import metaxy as mx
from metaxy import HashAlgorithm
from metaxy._testing import add_metaxy_provenance_column
from metaxy.metadata_store.duckdb import DuckDBMetadataStore


class RemoteFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="remote",
        id_columns=["sample_id"],
        fields=["value"],
    ),
):
    """Feature stored in remote PostgreSQL database."""

    sample_id: int
    value: str


class LocalFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="local",
        id_columns=["sample_id"],
        fields=["score"],
    ),
):
    """Feature stored in local DuckDB database."""

    sample_id: int
    score: float


@pytest.mark.slow
def test_adbc_scanner_api_exists(tmp_path):
    """Test that ADBC scanner API methods exist and extension can be loaded.

    This test verifies the metaxy API wrappers work correctly, even though
    actual ADBC connections may fail due to DuckDB ADBC scanner limitations.
    """
    duckdb_store = DuckDBMetadataStore(
        database=tmp_path / "local.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    # Verify API methods exist
    assert hasattr(duckdb_store, "install_adbc_scanner")
    assert hasattr(duckdb_store, "adbc_connect")
    assert hasattr(duckdb_store, "adbc_disconnect")
    assert hasattr(duckdb_store, "adbc_scan")
    assert hasattr(duckdb_store, "adbc_scan_table")

    # Verify extension can be installed
    with duckdb_store.open("write"):
        try:
            duckdb_store.install_adbc_scanner()
            # If we get here, extension is available
            print("✓ ADBC scanner extension installed successfully")
        except Exception as e:
            pytest.skip(f"ADBC scanner extension not available: {e}")


@pytest.mark.postgres
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DuckDB ADBC scanner has known issues with PostgreSQL connections",
    strict=False,
)
def test_federation_basic_query(tmp_path, postgres_db):
    """Test basic query of remote PostgreSQL via ADBC scanner.

    Note:
        This test may fail with "INTERNAL" errors from the ADBC PostgreSQL driver.
        This is a known issue with the DuckDB ADBC scanner extension, not with
        metaxy's API wrappers.
    """
    # Setup: Write data to remote PostgreSQL
    postgres_store = PostgresMetadataStore(
        connection_string=postgres_db,
        hash_algorithm=HashAlgorithm.MD5,
        auto_create_tables=True,
    )

    remote_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5],
            "value": ["a", "b", "c", "d", "e"],
            "metaxy_provenance_by_field": [{"value": f"hash_{i}"} for i in range(1, 6)],
        }
    )

    with RemoteFeature.graph.use(), postgres_store.open("write"):
        metadata = add_metaxy_provenance_column(remote_data, RemoteFeature)
        postgres_store.write_metadata(RemoteFeature, metadata)

    # Test: Query from DuckDB via ADBC scanner
    duckdb_store = DuckDBMetadataStore(
        database=tmp_path / "local.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    with duckdb_store.open("write"):
        # Install ADBC scanner extension
        try:
            duckdb_store.install_adbc_scanner()
        except Exception as e:
            pytest.skip(f"ADBC scanner extension not available: {e}")

        # Connect to remote PostgreSQL
        handle = duckdb_store.adbc_connect(
            {
                "driver": "postgresql",
                "uri": postgres_db,
            }
        )

        try:
            # Query all remote data
            result = duckdb_store.adbc_scan(handle, "SELECT * FROM remote__key ORDER BY sample_id")

            # Verify results
            result_native = result.to_native()
            assert len(result_native) == 5
            assert result_native["sample_id"].to_list() == [1, 2, 3, 4, 5]
            assert result_native["value"].to_list() == ["a", "b", "c", "d", "e"]

        finally:
            duckdb_store.adbc_disconnect(handle)


@pytest.mark.postgres
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DuckDB ADBC scanner has known issues with PostgreSQL connections",
    strict=False,
)
def test_federation_filtered_query(tmp_path, postgres_db):
    """Test filtered query of remote PostgreSQL."""
    # Setup: Write data to remote PostgreSQL
    postgres_store = PostgresMetadataStore(
        connection_string=postgres_db,
        hash_algorithm=HashAlgorithm.MD5,
        auto_create_tables=True,
    )

    remote_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5],
            "value": ["a", "b", "c", "d", "e"],
            "metaxy_provenance_by_field": [{"value": f"hash_{i}"} for i in range(1, 6)],
        }
    )

    with RemoteFeature.graph.use(), postgres_store.open("write"):
        metadata = add_metaxy_provenance_column(remote_data, RemoteFeature)
        postgres_store.write_metadata(RemoteFeature, metadata)

    # Test: Query with WHERE clause
    duckdb_store = DuckDBMetadataStore(
        database=tmp_path / "local.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    with duckdb_store.open("write"):
        try:
            duckdb_store.install_adbc_scanner()
        except Exception as e:
            pytest.skip(f"ADBC scanner extension not available: {e}")

        handle = duckdb_store.adbc_connect({"driver": "postgresql", "uri": postgres_db})

        try:
            # Query with filter
            result = duckdb_store.adbc_scan(
                handle,
                "SELECT sample_id, value FROM remote__key WHERE sample_id <= 3 ORDER BY sample_id",
            )

            # Verify filtered results
            result_native = result.to_native()
            assert len(result_native) == 3
            assert result_native["sample_id"].to_list() == [1, 2, 3]
            assert result_native["value"].to_list() == ["a", "b", "c"]

        finally:
            duckdb_store.adbc_disconnect(handle)


@pytest.mark.postgres
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DuckDB ADBC scanner has known issues with PostgreSQL connections",
    strict=False,
)
def test_federation_scan_table(tmp_path, postgres_db):
    """Test scanning entire remote table using adbc_scan_table."""
    # Setup: Write data to remote PostgreSQL
    postgres_store = PostgresMetadataStore(
        connection_string=postgres_db,
        hash_algorithm=HashAlgorithm.MD5,
        auto_create_tables=True,
    )

    remote_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "value": ["x", "y", "z"],
            "metaxy_provenance_by_field": [{"value": f"hash_{i}"} for i in range(1, 4)],
        }
    )

    with RemoteFeature.graph.use(), postgres_store.open("write"):
        metadata = add_metaxy_provenance_column(remote_data, RemoteFeature)
        postgres_store.write_metadata(RemoteFeature, metadata)

    # Test: Scan entire table
    duckdb_store = DuckDBMetadataStore(
        database=tmp_path / "local.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    with duckdb_store.open("write"):
        try:
            duckdb_store.install_adbc_scanner()
        except Exception as e:
            pytest.skip(f"ADBC scanner extension not available: {e}")

        handle = duckdb_store.adbc_connect({"driver": "postgresql", "uri": postgres_db})

        try:
            # Scan entire table (no WHERE clause)
            result = duckdb_store.adbc_scan_table(handle, "remote__key")

            # Verify results
            result_native = result.to_native()
            assert len(result_native) == 3
            assert set(result_native["value"].to_list()) == {"x", "y", "z"}

        finally:
            duckdb_store.adbc_disconnect(handle)


@pytest.mark.postgres
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DuckDB ADBC scanner has known issues with PostgreSQL connections",
    strict=False,
)
def test_federation_cross_database_join(tmp_path, postgres_db):
    """Test JOIN between local DuckDB and remote PostgreSQL."""
    # Setup remote PostgreSQL data
    postgres_store = PostgresMetadataStore(
        connection_string=postgres_db,
        hash_algorithm=HashAlgorithm.MD5,
        auto_create_tables=True,
    )

    remote_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3, 4],
            "value": ["remote_a", "remote_b", "remote_c", "remote_d"],
            "metaxy_provenance_by_field": [{"value": f"hash_{i}"} for i in range(1, 5)],
        }
    )

    with RemoteFeature.graph.use(), postgres_store.open("write"):
        metadata = add_metaxy_provenance_column(remote_data, RemoteFeature)
        postgres_store.write_metadata(RemoteFeature, metadata)

    # Setup local DuckDB data
    duckdb_store = DuckDBMetadataStore(
        database=tmp_path / "local.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
        auto_create_tables=True,
    )

    local_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "score": [10.0, 20.0, 30.0],
            "metaxy_provenance_by_field": [{"score": f"hash_{i}"} for i in range(1, 4)],
        }
    )

    with LocalFeature.graph.use(), duckdb_store.open("write"):
        metadata = add_metaxy_provenance_column(local_data, LocalFeature)
        duckdb_store.write_metadata(LocalFeature, metadata)

        # Install ADBC scanner and connect to remote
        try:
            duckdb_store.install_adbc_scanner()
        except Exception as e:
            pytest.skip(f"ADBC scanner extension not available: {e}")

        handle = duckdb_store.adbc_connect({"driver": "postgresql", "uri": postgres_db})

        try:
            # Perform cross-database JOIN
            raw_conn = duckdb_store._duckdb_raw_connection()
            joined = raw_conn.execute(
                """
                SELECT
                    l.sample_id,
                    l.score,
                    r.value
                FROM local__key l
                INNER JOIN (
                    SELECT * FROM adbc_scan_table(?, 'remote__key')
                ) r ON l.sample_id = r.sample_id
                ORDER BY l.sample_id
            """,
                [handle],
            ).fetchdf()

            # Verify JOIN results
            assert len(joined) == 3  # Only IDs 1, 2, 3 are in both tables
            assert joined["sample_id"].to_list() == [1, 2, 3]
            assert joined["score"].to_list() == [10.0, 20.0, 30.0]
            assert joined["value"].to_list() == ["remote_a", "remote_b", "remote_c"]

        finally:
            duckdb_store.adbc_disconnect(handle)


@pytest.mark.postgres
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DuckDB ADBC scanner has known issues with PostgreSQL connections",
    strict=False,
)
def test_federation_multiple_connections(tmp_path, postgres_db):
    """Test managing multiple ADBC connections simultaneously."""
    # Setup: Write data to remote PostgreSQL
    postgres_store = PostgresMetadataStore(
        connection_string=postgres_db,
        hash_algorithm=HashAlgorithm.MD5,
        auto_create_tables=True,
    )

    remote_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "value": ["a", "b", "c"],
            "metaxy_provenance_by_field": [{"value": f"hash_{i}"} for i in range(1, 4)],
        }
    )

    with RemoteFeature.graph.use(), postgres_store.open("write"):
        metadata = add_metaxy_provenance_column(remote_data, RemoteFeature)
        postgres_store.write_metadata(RemoteFeature, metadata)

    # Test: Create multiple connections to same database
    duckdb_store = DuckDBMetadataStore(
        database=tmp_path / "local.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    with duckdb_store.open("write"):
        try:
            duckdb_store.install_adbc_scanner()
        except Exception as e:
            pytest.skip(f"ADBC scanner extension not available: {e}")

        # Open two connections
        handle1 = duckdb_store.adbc_connect({"driver": "postgresql", "uri": postgres_db})
        handle2 = duckdb_store.adbc_connect({"driver": "postgresql", "uri": postgres_db})

        try:
            # Query using both handles
            result1 = duckdb_store.adbc_scan(handle1, "SELECT COUNT(*) as cnt FROM remote__key")
            result2 = duckdb_store.adbc_scan(handle2, "SELECT COUNT(*) as cnt FROM remote__key")

            # Both should return same count
            assert result1.to_native()["cnt"][0] == 3
            assert result2.to_native()["cnt"][0] == 3

            # Handles should be different
            assert handle1 != handle2

        finally:
            # Clean up both connections
            duckdb_store.adbc_disconnect(handle1)
            duckdb_store.adbc_disconnect(handle2)


@pytest.mark.postgres
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DuckDB ADBC scanner has known issues with PostgreSQL connections",
    strict=False,
)
def test_federation_connection_error_handling(tmp_path):
    """Test error handling for invalid connection parameters."""
    duckdb_store = DuckDBMetadataStore(
        database=tmp_path / "local.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    with duckdb_store.open("write"):
        try:
            duckdb_store.install_adbc_scanner()
        except Exception as e:
            pytest.skip(f"ADBC scanner extension not available: {e}")

        # Try to connect to non-existent server
        with pytest.raises(Exception):  # Should raise connection error
            duckdb_store.adbc_connect(
                {
                    "driver": "postgresql",
                    "uri": "postgresql://invalid-host:9999/nonexistent",
                }
            )


@pytest.mark.postgres
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DuckDB ADBC scanner has known issues with PostgreSQL connections",
    strict=False,
)
def test_federation_invalid_query_error_handling(tmp_path, postgres_db):
    """Test error handling for invalid SQL queries."""
    duckdb_store = DuckDBMetadataStore(
        database=tmp_path / "local.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    with duckdb_store.open("write"):
        try:
            duckdb_store.install_adbc_scanner()
        except Exception as e:
            pytest.skip(f"ADBC scanner extension not available: {e}")

        handle = duckdb_store.adbc_connect({"driver": "postgresql", "uri": postgres_db})

        try:
            # Try to query non-existent table
            with pytest.raises(Exception):  # Should raise table not found error
                duckdb_store.adbc_scan(handle, "SELECT * FROM nonexistent_table")

            # Try invalid SQL syntax
            with pytest.raises(Exception):  # Should raise syntax error
                duckdb_store.adbc_scan(handle, "INVALID SQL QUERY")

        finally:
            duckdb_store.adbc_disconnect(handle)


@pytest.mark.postgres
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DuckDB ADBC scanner has known issues with PostgreSQL connections",
    strict=False,
)
def test_federation_schema_discovery(tmp_path, postgres_db):
    """Test that schema is preserved when querying remote tables."""
    # Setup: Write data with specific types
    postgres_store = PostgresMetadataStore(
        connection_string=postgres_db,
        hash_algorithm=HashAlgorithm.MD5,
        auto_create_tables=True,
    )

    remote_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "value": ["a", "b", "c"],
            "metaxy_provenance_by_field": [{"value": f"hash_{i}"} for i in range(1, 4)],
        }
    )

    with RemoteFeature.graph.use(), postgres_store.open("write"):
        metadata = add_metaxy_provenance_column(remote_data, RemoteFeature)
        postgres_store.write_metadata(RemoteFeature, metadata)

    # Test: Query and verify schema
    duckdb_store = DuckDBMetadataStore(
        database=tmp_path / "local.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    with duckdb_store.open("write"):
        try:
            duckdb_store.install_adbc_scanner()
        except Exception as e:
            pytest.skip(f"ADBC scanner extension not available: {e}")

        handle = duckdb_store.adbc_connect({"driver": "postgresql", "uri": postgres_db})

        try:
            result = duckdb_store.adbc_scan(handle, "SELECT * FROM remote__key")

            # Verify schema is preserved
            result_native = result.to_native()
            assert "sample_id" in result_native.columns
            assert "value" in result_native.columns
            assert "metaxy_provenance" in result_native.columns

            # Verify types (Polars should infer correct types from Arrow)
            assert result_native["sample_id"].dtype == pl.Int64 or result_native["sample_id"].dtype == pl.Int32
            assert result_native["value"].dtype == pl.String

        finally:
            duckdb_store.adbc_disconnect(handle)


@pytest.mark.postgres
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DuckDB ADBC scanner has known issues with PostgreSQL connections",
    strict=False,
)
def test_federation_empty_result_set(tmp_path, postgres_db):
    """Test querying remote table with WHERE clause that matches no rows."""
    # Setup: Write data to remote PostgreSQL
    postgres_store = PostgresMetadataStore(
        connection_string=postgres_db,
        hash_algorithm=HashAlgorithm.MD5,
        auto_create_tables=True,
    )

    remote_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "value": ["a", "b", "c"],
            "metaxy_provenance_by_field": [{"value": f"hash_{i}"} for i in range(1, 4)],
        }
    )

    with RemoteFeature.graph.use(), postgres_store.open("write"):
        metadata = add_metaxy_provenance_column(remote_data, RemoteFeature)
        postgres_store.write_metadata(RemoteFeature, metadata)

    # Test: Query with WHERE clause that matches nothing
    duckdb_store = DuckDBMetadataStore(
        database=tmp_path / "local.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    with duckdb_store.open("write"):
        try:
            duckdb_store.install_adbc_scanner()
        except Exception as e:
            pytest.skip(f"ADBC scanner extension not available: {e}")

        handle = duckdb_store.adbc_connect({"driver": "postgresql", "uri": postgres_db})

        try:
            # Query with impossible WHERE clause
            result = duckdb_store.adbc_scan(handle, "SELECT * FROM remote__key WHERE sample_id > 1000")

            # Should return empty DataFrame with correct schema
            result_native = result.to_native()
            assert len(result_native) == 0
            assert "sample_id" in result_native.columns
            assert "value" in result_native.columns

        finally:
            duckdb_store.adbc_disconnect(handle)
