"""Common fixtures for metadata store tests."""

import os
import socket
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import boto3
import duckdb
import pytest
from metaxy_testing import HashAlgorithmCases
from moto.server import ThreadedMotoServer
from pytest_cases import fixture, parametrize_with_cases

from metaxy import HashAlgorithm
from metaxy.config import MetaxyConfig
from metaxy.ext.metadata_stores.clickhouse import ClickHouseMetadataStore
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore
from metaxy.ext.metadata_stores.lancedb import LanceDBMetadataStore
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    MetadataStore,
)
from tests.conftest import require_fixture


def find_free_port() -> int:
    """Find a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


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


# ============= FIXTURES FOR NON-HASH TESTS =============


@pytest.fixture
def default_store(tmp_path: Path) -> DeltaMetadataStore:
    """Default store (Delta, xxhash64)."""
    delta_path = tmp_path / "default_delta_store"
    return DeltaMetadataStore(
        root_path=delta_path,
        hash_algorithm=HashAlgorithm.XXHASH64,
    )


@pytest.fixture
def ibis_store(tmp_path: Path) -> DuckDBMetadataStore:
    """Ibis store (DuckDB, xxhash64)."""
    from metaxy.versioning.types import HashAlgorithm

    return DuckDBMetadataStore(
        database=tmp_path / "test.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
        extensions=["hashfuncs"],
    )


class AllStoresCases:
    """All store types (Delta, DuckDB, DuckDB+DuckLake, ClickHouse, LanceDB)."""

    @pytest.mark.delta
    @pytest.mark.polars
    def case_delta(self, tmp_path: Path) -> MetadataStore:
        delta_path = tmp_path / "delta_store"
        return DeltaMetadataStore(
            root_path=delta_path,
            hash_algorithm=HashAlgorithm.XXHASH64,
        )

    @pytest.mark.ibis
    @pytest.mark.native
    @pytest.mark.duckdb
    def case_duckdb(self, tmp_path: Path) -> MetadataStore:
        return DuckDBMetadataStore(
            database=tmp_path / "test.duckdb",
            hash_algorithm=HashAlgorithm.XXHASH64,
            extensions=["hashfuncs"],
        )

    @pytest.mark.ibis
    @pytest.mark.native
    @pytest.mark.duckdb
    @pytest.mark.ducklake
    def case_duckdb_ducklake(self, tmp_path: Path) -> MetadataStore:
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

    @pytest.mark.ibis
    @pytest.mark.native
    @pytest.mark.clickhouse
    def case_clickhouse(self, request) -> MetadataStore:
        return ClickHouseMetadataStore(
            connection_string=require_fixture(request, "clickhouse_db"),
            hash_algorithm=HashAlgorithm.XXHASH64,
        )

    @pytest.mark.lancedb
    @pytest.mark.polars
    def case_lancedb(self, tmp_path: Path) -> MetadataStore:
        lancedb_path = tmp_path / "lancedb_store"
        return LanceDBMetadataStore(
            uri=lancedb_path,
            hash_algorithm=HashAlgorithm.XXHASH64,
        )


@fixture
@parametrize_with_cases("store", cases=AllStoresCases)
def any_store(store: MetadataStore) -> MetadataStore:
    """Parametrized store (Delta + DuckDB + ClickHouse + LanceDB)."""
    return store


@pytest.fixture
def default_hash_algorithm():
    """Single default hash algorithm for non-hash tests (xxhash64).

    Use this fixture when you need a hash algorithm but aren't testing
    hash algorithm behavior specifically.
    """
    return HashAlgorithm.XXHASH64


@fixture
@parametrize_with_cases("algo", cases=HashAlgorithmCases)
def hash_algorithm(algo):
    """Parametrized hash algorithm fixture for hash algorithm tests.

    This creates the Cartesian product with store fixtures that use it.
    """
    return algo


@fixture
def store_with_hash_algo_native(any_store: MetadataStore, hash_algorithm: HashAlgorithm) -> MetadataStore:
    """Parametrized store with parametrized hash algorithm.

    Use with @parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
    to test all stores with all hash algorithms.
    """
    any_store._versioning_engine = "native"
    any_store.hash_algorithm = hash_algorithm
    try:
        any_store.validate_hash_algorithm()
    except HashAlgorithmNotSupportedError:
        pytest.skip(f"Hash algorithm {hash_algorithm} not supported by store {any_store.display()}")
    return any_store


@pytest.fixture
def config_with_truncation(truncation_length):
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


@pytest.fixture(scope="session")
def s3_endpoint_url() -> Generator[str, None, None]:
    """Start a moto S3 server on a random free port."""
    port = find_free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture(scope="function")
def s3_bucket_and_storage_options(
    s3_endpoint_url: str,
) -> tuple[str, dict[str, Any]]:
    """
    Creates a unique S3 bucket and provides storage_options
    """
    bucket_name = f"test-bucket-{uuid.uuid4().hex[:8]}"
    access_key = "testing"
    secret_key = "testing"
    region = "us-east-1"

    s3_resource: Any = boto3.resource(
        "s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3_resource.create_bucket(Bucket=bucket_name)

    storage_options = {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
        "AWS_ENDPOINT_URL": s3_endpoint_url,
        "AWS_REGION": region,
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }

    return (bucket_name, storage_options)


# ---------------------------------------------------------------------------
# MotherDuck fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def motherduck_token() -> str:
    import warnings

    token = os.environ.get("MOTHERDUCK_TOKEN", "")
    if not token:
        warnings.warn("MOTHERDUCK_TOKEN not set — skipping MotherDuck tests", stacklevel=1)
        pytest.skip("MOTHERDUCK_TOKEN not set")
    return token


@pytest.fixture(scope="session")
def motherduck_region() -> str | None:
    """MotherDuck organization AWS region (e.g. ``eu-central-1``).

    Reads ``MOTHERDUCK_REGION``; returns ``None`` when unset.  Required for
    DuckLake writes because DuckDB does not follow S3 301 redirects.
    """
    return os.environ.get("MOTHERDUCK_REGION") or None


@pytest.fixture(scope="session")
def motherduck_database(motherduck_token: str) -> Generator[str, None, None]:
    db_name = f"metaxy_tests_{uuid.uuid4().hex[:8]}"
    conn = duckdb.connect(f"md:?motherduck_token={motherduck_token}")
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    conn.close()
    yield db_name
    conn = duckdb.connect(f"md:?motherduck_token={motherduck_token}")
    conn.execute(f"DROP DATABASE IF EXISTS {db_name}")
    conn.close()


@pytest.fixture(scope="session")
def motherduck_ducklake_database(motherduck_token: str) -> Generator[str, None, None]:
    db_name = f"metaxy_tests_lake_{uuid.uuid4().hex[:8]}"
    conn = duckdb.connect(f"md:?motherduck_token={motherduck_token}")
    conn.install_extension("ducklake")
    conn.load_extension("ducklake")
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} (TYPE DUCKLAKE)")
    conn.close()
    yield db_name
    conn = duckdb.connect(f"md:?motherduck_token={motherduck_token}")
    conn.execute(f"DROP DATABASE IF EXISTS {db_name}")
    conn.close()
