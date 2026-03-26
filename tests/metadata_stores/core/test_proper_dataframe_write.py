"""Tests for DuckDB-specific SQL dialect detection."""

from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore


def test_sql_dialect_uses_connection(ibis_store: DuckDBMetadataStore) -> None:
    with ibis_store.open("w"):
        expected = ibis_store.conn.name
        assert ibis_store._sql_dialect == expected
