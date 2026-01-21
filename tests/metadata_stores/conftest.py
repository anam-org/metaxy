"""Common fixtures for metadata store tests."""

import logging
import os
import socket
import subprocess
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import boto3
import psycopg
import pytest
from metaxy._testing.duckdb_json_compat_store import DuckDBJsonCompatStore  # type: ignore[import-not-found]
from metaxy.metadata_store.postgres import PostgresMetadataStore  # type: ignore[import-not-found]
from moto.server import ThreadedMotoServer
from psycopg import conninfo as psycopg_conninfo
from pytest_cases import fixture, parametrize_with_cases
from pytest_postgresql import factories

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
from metaxy.models.feature import FeatureGraph

# Note: clickhouse_server and clickhouse_db fixtures are defined in tests/conftest.py
# to be available across all test directories.


def find_free_port() -> int:
    """Find a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


logger = logging.getLogger(__name__)

# Force pytest-postgresql to use a short socket directory to avoid hitting the
# 103-character Unix socket limit enforced by PostgreSQL on macOS.
_PG_SOCKET_DIR = Path("/tmp/metaxy-pg")
_PG_SOCKET_DIR.mkdir(parents=True, exist_ok=True)


# Use Nix-provided PostgreSQL if available, otherwise use system PostgreSQL
original_popen = subprocess.Popen


class SafePopen(original_popen):
    def __init__(self, args, **kwargs):
        if (
            isinstance(args, list)
            and len(args) > 0
            and any(x in str(args[0]) for x in ["pg_ctl", "initdb", "postgres"])
        ):
            env = kwargs.get("env", os.environ).copy()
            for var in [
                "LD_LIBRARY_PATH",
                "DYLD_LIBRARY_PATH",
                "DYLD_FALLBACK_LIBRARY_PATH",
            ]:
                if var in env:
                    del env[var]
            env["LANG"] = "C.UTF-8"
            env["LC_ALL"] = "C.UTF-8"
            kwargs["env"] = env
        super().__init__(args, **kwargs)


@pytest.fixture(scope="session", autouse=True)
def patch_subprocess():
    subprocess.Popen = SafePopen  # type: ignore
    yield
    subprocess.Popen = original_popen  # type: ignore


# --- SETUP POSTGRES PATH ---
_nix_pg_bin = os.environ.get("PG_BIN")
_pg_executable = None
_pg_available = False

if _nix_pg_bin:
    _pg_executable = str(Path(_nix_pg_bin) / "pg_ctl")
    _pg_available = Path(_pg_executable).exists()
else:
    # Try to find pg_ctl in system PATH
    import shutil

    pg_ctl_path = shutil.which("pg_ctl")
    if pg_ctl_path:
        _pg_executable = pg_ctl_path
        _pg_available = True

# --- FIXTURE CONFIGURATION ---
if _pg_available:
    postgresql_proc = factories.postgresql_proc(
        executable=_pg_executable,
        unixsocketdir="/tmp",
        postgres_options="-c fsync=off -c synchronous_commit=off -c full_page_writes=off",
        # Ensure default user matches what Nix Postgres expects (usually the current user or postgres)
        user="postgres",
        password=None,
    )
else:
    # Create a dummy fixture that will be skipped
    @pytest.fixture(scope="session")
    def postgresql_proc():
        pytest.skip("PostgreSQL is not available on this system")


@pytest.fixture(scope="session")
def postgres_server(postgresql_proc: Any):
    """Expose connection details from pytest-postgresql's server."""
    host = postgresql_proc.host
    port = postgresql_proc.port
    user = postgresql_proc.user
    password = postgresql_proc.password
    options = postgresql_proc.options

    admin_dbname = "postgres"

    admin_dsn = psycopg_conninfo.make_conninfo(
        host=host,
        port=str(port),
        user=user,
        password=password or None,
        dbname=admin_dbname,
        options=options or None,
    )

    return {
        "host": host,
        "port": port,
        "user": user,
        "dbname": admin_dbname,
        "password": password,
        "options": options,
        "dsn": admin_dsn,
        "psycopg": psycopg,
    }


@pytest.fixture
def postgres_db(postgres_server):
    """Create a clean PostgreSQL database for each test."""
    psycopg_mod = postgres_server["psycopg"]
    admin_dsn = postgres_server["dsn"]
    host = postgres_server["host"]
    port = postgres_server["port"]
    user = postgres_server["user"]
    password = postgres_server["password"]

    db_name = f"test_{uuid.uuid4().hex}"

    with psycopg_mod.connect(admin_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{db_name}"')

    if password:
        auth = f"{user}:{password}"
    else:
        auth = user
    test_conn_string = f"postgresql://{auth}@{host}:{port}/{db_name}"

    yield test_conn_string

    try:
        with psycopg_mod.connect(admin_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
    except psycopg.Error as exc:
        logger.warning("Failed to drop test database %s: %s", db_name, exc)


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
    """All store types (Delta, DuckDB, ClickHouse, LanceDB, ADBC variants)."""

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
    @pytest.mark.native
    @pytest.mark.postgres
    def case_postgres(self, postgres_db: str) -> MetadataStore:
        return PostgresMetadataStore(
            connection_string=postgres_db,
            hash_algorithm=HashAlgorithm.MD5,
            auto_create_tables=True,
        )

    @pytest.mark.ibis
    @pytest.mark.native
    @pytest.mark.duckdb
    def case_duckdb_json_compat(self, tmp_path: Path) -> MetadataStore:
        db_path = tmp_path / "test_json_compat.duckdb"
        return DuckDBJsonCompatStore(database=str(db_path))

    # ADBC store cases - use native ADBC drivers for high-performance bulk writes

    @pytest.mark.adbc
    @pytest.mark.native
    @pytest.mark.postgres
    def case_adbc_postgres(self, postgres_db: str) -> MetadataStore:
        from metaxy.metadata_store.adbc_postgres import ADBCPostgresMetadataStore

        return ADBCPostgresMetadataStore(
            connection_string=postgres_db,
            hash_algorithm=HashAlgorithm.MD5,
        )

    # TODO: DuckDB ADBC support requires different implementation
    # DuckDB's ADBC driver is built into the main duckdb package, not distributed separately
    # Will be implemented in a future PR using DuckDB's native ADBC integration
    # @pytest.mark.adbc
    # @pytest.mark.native
    # @pytest.mark.duckdb
    # def case_adbc_duckdb(self, tmp_path: Path) -> MetadataStore:
    #     from metaxy.metadata_store.adbc_duckdb import ADBCDuckDBMetadataStore
    #     return ADBCDuckDBMetadataStore(
    #         database=tmp_path / "test_adbc.duckdb",
    #         hash_algorithm=HashAlgorithm.XXHASH64,
    #     )

    @pytest.mark.adbc
    @pytest.mark.native
    def case_adbc_sqlite(self, tmp_path: Path) -> MetadataStore:
        from metaxy.metadata_store.adbc_sqlite import ADBCSQLiteMetadataStore

        return ADBCSQLiteMetadataStore(
            database=tmp_path / "test_adbc.sqlite",
            hash_algorithm=HashAlgorithm.MD5,
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
