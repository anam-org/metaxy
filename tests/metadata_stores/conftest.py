"""Common fixtures for metadata store tests."""

import os
import socket
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import boto3
import pytest
from moto.server import ThreadedMotoServer
from pytest_cases import fixture, parametrize_with_cases

from metaxy import HashAlgorithm
from metaxy._testing import HashAlgorithmCases
from metaxy.config import MetaxyConfig
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    MetadataStore,
)
from metaxy.metadata_store.clickhouse import ClickHouseMetadataStore
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.lancedb import LanceDBMetadataStore
from metaxy.metadata_store.postgresql import PostgreSQLMetadataStore
from metaxy.models.feature import FeatureGraph

# Note: clickhouse_server and clickhouse_db fixtures are defined in tests/conftest.py
# to be available across all test directories.

# ============= POSTGRESQL FIXTURES =============

# Force pytest-postgresql to use a short socket directory to avoid hitting the
# 103-character Unix socket limit enforced by PostgreSQL on macOS.
# Use system temp directory for cross-platform compatibility (Windows, Linux, macOS)
_PG_SOCKET_DIR = Path(tempfile.gettempdir()) / "metaxy-pg"
_PG_SOCKET_DIR.mkdir(parents=True, exist_ok=True)


# --- SETUP POSTGRES PATH ---
_nix_pg_bin = os.environ.get("PG_BIN")
_pg_available = False

if _nix_pg_bin:
    _pg_executable = str(Path(_nix_pg_bin) / "pg_ctl")
    _pg_available = Path(_pg_executable).exists()
else:
    # Try system PATH
    _pg_executable = "pg_ctl"
    _pg_available = True


# --- FIXTURE CONFIGURATION ---
if _pg_available:
    try:
        from pytest_postgresql import factories

        postgresql_proc = factories.postgresql_proc(
            executable=_pg_executable,
            unixsocketdir=str(_PG_SOCKET_DIR),
            postgres_options="-c fsync=off -c synchronous_commit=off -c full_page_writes=off",
            user="postgres",
            password=None,
        )

        # Session-scoped process, function-scoped database for test isolation
        @pytest.fixture(scope="function")
        def postgresql_db(postgresql_proc) -> str:  # type: ignore[no-untyped-def]
            """PostgreSQL database connection string fixture.

            Returns connection string for a test PostgreSQL instance.
            Creates a fresh database for each test to ensure isolation.
            """
            import psycopg

            # Use unique database name for each test
            test_db_name = f"metaxy_test_{uuid.uuid4().hex[:8]}"

            # Connect to postgres database to create test database
            admin_conn_str = (
                f"postgresql://{postgresql_proc.user}@{postgresql_proc.host}:{postgresql_proc.port}/postgres"
            )
            conn = psycopg.connect(admin_conn_str, autocommit=True)
            try:
                conn.execute(f"CREATE DATABASE {test_db_name}")  # ty: ignore[no-matching-overload]
            finally:
                conn.close()

            conn_str = (
                f"postgresql://{postgresql_proc.user}@{postgresql_proc.host}:{postgresql_proc.port}/{test_db_name}"
            )

            yield conn_str

            # Cleanup: force disconnect all connections and drop database
            conn = psycopg.connect(admin_conn_str, autocommit=True)
            try:
                # Terminate all connections to the test database
                conn.execute(  # ty: ignore[no-matching-overload]
                    f"""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = '{test_db_name}'
                      AND pid <> pg_backend_pid()
                    """
                )
                # Now drop the database
                conn.execute(f"DROP DATABASE IF EXISTS {test_db_name}")  # ty: ignore[no-matching-overload]
            finally:
                conn.close()

    except ImportError:
        # pytest-postgresql not installed, use environment variable
        @pytest.fixture(scope="function")
        def postgresql_db() -> str:
            """PostgreSQL connection string from environment (pytest-postgresql not available)."""
            conn_str = os.environ.get("POSTGRES_TEST_URL")
            if not conn_str:
                pytest.skip("pytest-postgresql not installed and POSTGRES_TEST_URL not set")
            assert conn_str is not None  # ty: Help type checker understand pytest.skip never returns
            return conn_str

else:
    # PostgreSQL not available, skip
    @pytest.fixture(scope="function")
    def postgresql_db() -> str:
        """PostgreSQL not available - skip tests."""
        pytest.skip("PostgreSQL is not available on this system. Set PG_BIN or install PostgreSQL.")


def find_free_port() -> int:
    """Find a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class StoreCases:
    """Store configuration cases for parametrization."""

    def case_duckdb(self, tmp_path: Path, test_graph: FeatureGraph) -> tuple[type[MetadataStore], dict[str, Any]]:
        db_path = tmp_path / "test.duckdb"
        return (DuckDBMetadataStore, {"database": db_path})

    def case_duckdb_ducklake(
        self, tmp_path: Path, test_graph: FeatureGraph
    ) -> tuple[type[MetadataStore], dict[str, Any]]:
        db_path = tmp_path / "test_ducklake.duckdb"
        metadata_path = tmp_path / "ducklake_catalog.duckdb"
        storage_dir = tmp_path / "ducklake_storage"

        ducklake_config = {
            "alias": "integration_lake",
            "metadata_backend": {"type": "duckdb", "path": str(metadata_path)},
            "storage_backend": {"type": "local", "path": str(storage_dir)},
        }

        config = {
            "database": db_path,
            "ducklake": ducklake_config,
            "extensions": ["json"],
        }
        return (DuckDBMetadataStore, config)

    def case_clickhouse(
        self, clickhouse_db: str, test_graph: FeatureGraph
    ) -> tuple[type[MetadataStore], dict[str, Any]]:
        return (ClickHouseMetadataStore, {"connection_string": clickhouse_db})


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


class AnyStoreCases:
    """Minimal store cases (Delta + DuckDB)."""

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


class AllStoresCases:
    """All store types (Delta, DuckDB, ClickHouse, LanceDB)."""

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
    @pytest.mark.clickhouse
    def case_clickhouse(self, clickhouse_db: str) -> MetadataStore:
        return ClickHouseMetadataStore(
            connection_string=clickhouse_db,
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

    @pytest.mark.ibis
    @pytest.mark.polars
    @pytest.mark.postgresql
    def case_postgresql(self, postgresql_db: str) -> MetadataStore:
        """PostgreSQL store case using pytest-postgresql or POSTGRES_TEST_URL."""
        return PostgreSQLMetadataStore(
            connection_string=postgresql_db,
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
