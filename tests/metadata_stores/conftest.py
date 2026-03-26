"""Common fixtures for metadata store core tests."""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from metaxy_testing import HashAlgorithmCases
from pytest_cases import fixture, parametrize_with_cases

from metaxy import HashAlgorithm
from metaxy.config import MetaxyConfig
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore
from metaxy.metadata_store import MetadataStore


class BasicStoreCases:
    """Minimal store cases for backend-agnostic API tests."""

    def case_duckdb(self, tmp_path: Path) -> tuple[type[MetadataStore], dict[str, Any]]:
        db_path = tmp_path / "test.duckdb"
        return (DuckDBMetadataStore, {"database": db_path})


@fixture
@parametrize_with_cases("store_config", cases=BasicStoreCases)
def persistent_store(
    store_config: tuple[type[MetadataStore], dict[str, Any]],
) -> MetadataStore:
    """Parametrized persistent store."""
    store_type, config = store_config
    return store_type(**config)


@pytest.fixture
def ibis_store(tmp_path: Path) -> DuckDBMetadataStore:
    """Ibis store (DuckDB, xxhash64)."""
    from metaxy.versioning.types import HashAlgorithm

    return DuckDBMetadataStore(
        database=tmp_path / "test.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )


@pytest.fixture
def default_hash_algorithm() -> HashAlgorithm:
    """Single default hash algorithm for non-hash tests (xxhash64).

    Use this fixture when you need a hash algorithm but aren't testing
    hash algorithm behavior specifically.
    """
    return HashAlgorithm.XXHASH64


@fixture
@parametrize_with_cases("algo", cases=HashAlgorithmCases)
def hash_algorithm(algo: HashAlgorithm) -> HashAlgorithm:
    """Parametrized hash algorithm fixture for hash algorithm tests.

    This creates the Cartesian product with store fixtures that use it.
    """
    return algo


@pytest.fixture
def config_with_truncation(truncation_length: int | None) -> Generator[MetaxyConfig, None, None]:
    """Fixture that sets MetaxyConfig with hash_truncation_length.

    The test must be parametrized on truncation_length for this fixture to work.

    Usage:
        @pytest.mark.parametrize("truncation_length", [None, 8, 16, 32])
        def test_something(config_with_truncation):
            # Config is already set with the truncation length from the parameter
            pass
    """
    config = MetaxyConfig.get().model_copy(update={"hash_truncation_length": truncation_length})

    with config.use():
        yield config
