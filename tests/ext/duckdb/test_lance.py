"""DuckDB + Lance storage handler tests."""

from pathlib import Path

import pytest

from metaxy import HashAlgorithm
from metaxy.ext.duckdb.engine import DuckDBEngine
from metaxy.ext.duckdb.handlers.lance import DuckDBLanceHandler
from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.storage_config import LanceStorageConfig
from tests.metadata_stores.shared import (
    CRUDTests,
    DisplayTests,
    FilterTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
)


@pytest.mark.ibis
@pytest.mark.native
@pytest.mark.duckdb
@pytest.mark.lance
class TestDuckDBLance(
    CRUDTests,
    DisplayTests,
    FilterTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
):
    @pytest.fixture
    def store(self, tmp_path: Path) -> MetadataStore:
        lance_path = tmp_path / "duckdb_lance_store"
        handler = DuckDBLanceHandler()
        engine = DuckDBEngine(
            database=":memory:",
            handlers=[handler],
            extensions=[*handler.required_extensions()],
        )
        return MetadataStore(
            engine=engine,
            storage=[LanceStorageConfig(format="lance", location=str(lance_path))],
            hash_algorithm=HashAlgorithm.XXHASH64,
        )
