"""DuckDB + DuckLake metadata store tests."""

from pathlib import Path

import pytest

from metaxy import HashAlgorithm
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore
from metaxy.metadata_store import MetadataStore
from tests.metadata_stores.shared import (
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
)


@pytest.mark.ibis
@pytest.mark.native
@pytest.mark.duckdb
@pytest.mark.ducklake
class TestDuckDBDuckLake(
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
):
    @pytest.fixture
    def store(self, tmp_path: Path) -> MetadataStore:
        from metaxy.ext.metadata_stores.ducklake import DuckLakeConfig

        return DuckDBMetadataStore(
            database=tmp_path / "test_ducklake.duckdb",
            hash_algorithm=HashAlgorithm.XXHASH64,
            ducklake=DuckLakeConfig.model_validate(
                {
                    "alias": "integration_lake",
                    "catalog": {"type": "duckdb", "uri": str(tmp_path / "ducklake_catalog.duckdb")},
                    "storage": {"type": "local", "path": str(tmp_path / "ducklake_storage")},
                }
            ),
        )

    @pytest.fixture
    def named_store(self, tmp_path: Path) -> MetadataStore:
        from metaxy.ext.metadata_stores.ducklake import DuckLakeConfig

        return DuckDBMetadataStore(
            database=tmp_path / "test_ducklake.duckdb",
            hash_algorithm=HashAlgorithm.XXHASH64,
            name="ducklake-staging",
            ducklake=DuckLakeConfig.model_validate(
                {
                    "alias": "integration_lake",
                    "catalog": {"type": "duckdb", "uri": str(tmp_path / "ducklake_catalog.duckdb")},
                    "storage": {"type": "local", "path": str(tmp_path / "ducklake_storage")},
                }
            ),
        )
