"""Tests for ADBC bulk ingestion API."""

import polars as pl
import pytest
from pytest_cases import fixture, parametrize_with_cases

import metaxy as mx
from metaxy import HashAlgorithm
from metaxy.metadata_store import MetadataStore


class SimpleFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="simple",
        id_columns=["sample_id"],
        fields=["value"],
    ),
):
    """Simple feature for bulk ingestion tests."""

    sample_id: int
    value: str


class ADBCStoreCases:
    """Test cases for ADBC stores only."""

    @pytest.mark.adbc
    @pytest.mark.native
    @pytest.mark.postgres
    def case_adbc_postgres(self, postgres_db: str) -> MetadataStore:
        from metaxy.metadata_store.adbc_postgres import ADBCPostgresMetadataStore

        return ADBCPostgresMetadataStore(
            connection_string=postgres_db,
            hash_algorithm=HashAlgorithm.MD5,
            max_connections=8,
        )

    @pytest.mark.adbc
    @pytest.mark.native
    @pytest.mark.duckdb
    def case_adbc_duckdb(self, tmp_path) -> MetadataStore:
        from metaxy.metadata_store.adbc_duckdb import ADBCDuckDBMetadataStore

        return ADBCDuckDBMetadataStore(
            database=tmp_path / "test_bulk.duckdb",
            hash_algorithm=HashAlgorithm.XXHASH64,
            max_connections=8,
        )

    @pytest.mark.adbc
    @pytest.mark.native
    def case_adbc_sqlite(self, tmp_path) -> MetadataStore:
        from metaxy.metadata_store.adbc_sqlite import ADBCSQLiteMetadataStore

        return ADBCSQLiteMetadataStore(
            database=tmp_path / "test_bulk.sqlite",
            hash_algorithm=HashAlgorithm.MD5,
            max_connections=8,
        )


@fixture
@parametrize_with_cases("store", cases=ADBCStoreCases)
def adbc_store(store: MetadataStore) -> MetadataStore:
    """Fixture for ADBC stores with connection pooling enabled."""
    return store


def test_adbc_store_has_bulk_api(adbc_store: MetadataStore):
    """Test that ADBC stores have write_metadata_bulk method."""
    assert hasattr(adbc_store, "write_metadata_bulk")
    assert callable(adbc_store.write_metadata_bulk)


def test_non_adbc_store_raises():
    """Test that non-ADBC stores raise NotImplementedError."""
    import tempfile
    from pathlib import Path

    from metaxy.metadata_store.delta import DeltaMetadataStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = DeltaMetadataStore(root_path=Path(tmpdir) / "delta")

        with pytest.raises(NotImplementedError, match="does not support bulk ingestion"):
            with store:
                store.write_metadata_bulk(SimpleFeature, pl.DataFrame(), concurrency=4)
