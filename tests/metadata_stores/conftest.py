"""Common fixtures for metadata store tests."""

import logging
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import ibis
import ibis.backends.postgres
import psycopg
import pytest
from psycopg import conninfo as psycopg_conninfo
from pytest_cases import fixture, parametrize_with_cases

from metaxy._testing import HashAlgorithmCases
from metaxy.metadata_store import InMemoryMetadataStore, MetadataStore
from metaxy.metadata_store.clickhouse import ClickHouseMetadataStore
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.models.feature import FeatureGraph

assert HashAlgorithmCases is not None  # ensure the import is not removed

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _format_pg_connection_string(
    *,
    user: str,
    host: str,
    port: int,
    database: str,
    password: str | None = None,
    options: str | None = None,
) -> str:
    """Build a PostgreSQL connection URI with proper escaping."""
    auth = quote_plus(user)
    if password:
        auth = f"{quote_plus(user)}:{quote_plus(password)}"
    uri = f"postgresql://{auth}@{host}:{port}/{database}"
    if options:
        uri = f"{uri}?options={quote_plus(options)}"
    return uri


@pytest.fixture(scope="session")
def clickhouse_server(tmp_path_factory):
    """Start a ClickHouse server for testing (session-scoped).

    Uses clickhouse binary to start a local server.
    Cleans up the process after all tests complete.

    Yields connection params (host, port) if ClickHouse is available, otherwise skips tests.
    """
    # Check if clickhouse binary is available
    clickhouse_bin = shutil.which("clickhouse") or shutil.which("clickhouse-server")
    if not clickhouse_bin:
        pytest.skip("ClickHouse binary not found in PATH")

    port = _find_free_port()
    http_port = _find_free_port()

    # Create temporary directories for ClickHouse
    base_dir = tmp_path_factory.mktemp("clickhouse")
    data_dir = base_dir / "data"
    data_dir.mkdir()
    log_dir = base_dir / "log"
    log_dir.mkdir()
    tmp_dir = base_dir / "tmp"
    tmp_dir.mkdir()
    user_files_dir = base_dir / "user_files"
    user_files_dir.mkdir()
    format_schemas_dir = base_dir / "format_schemas"
    format_schemas_dir.mkdir()

    # Start ClickHouse server with all paths configured
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(  # type: ignore[call-overload]
            [
                clickhouse_bin,
                "server",
                "--",
                f"--tcp_port={port}",
                f"--http_port={http_port}",
                f"--path={data_dir}/",
                f"--tmp_path={tmp_dir}/",
                f"--user_files_path={user_files_dir}/",
                f"--format_schema_path={format_schemas_dir}/",
                "--logger.console=1",
                "--logger.level=warning",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        pytest.skip(f"Failed to start ClickHouse server: {e}")

    assert process is not None, "Process should be initialized"

    # Wait for ClickHouse to be ready (max 30 seconds)
    # First, wait for the TCP port to be accepting connections
    max_retries = 30
    ready = False
    last_error = None

    for i in range(max_retries):
        # Check if process is still running
        if process.poll() is not None:
            # Process died - get stderr output
            _, stderr = process.communicate()
            pytest.skip(
                f"ClickHouse server process terminated unexpectedly: {stderr.decode()[:500]}"
            )

        # Try to connect to the port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(("localhost", port))
                ready = True
                break
        except (TimeoutError, ConnectionRefusedError, OSError) as e:
            last_error = e
            time.sleep(1)

    if not ready:
        process.terminate()
        try:
            _, stderr = process.communicate(timeout=5)
            error_msg = stderr.decode()[:500]
        except Exception:
            error_msg = "Could not get error output"
        pytest.skip(
            f"ClickHouse server port not ready. Last error: {last_error}. Stderr: {error_msg}"
        )

    connection_string = f"clickhouse://localhost:{http_port}/default"
    try:
        conn: Any = ibis.connect(connection_string)  # type: ignore[assignment]
        conn.list_tables()
    except Exception as e:
        process.terminate()
        try:
            _, stderr = process.communicate(timeout=5)
            error_msg = stderr.decode()[:500]
        except Exception:
            error_msg = "Could not get error output"
        pytest.skip(f"ClickHouse Ibis connection failed: {e}. Stderr: {error_msg}")

    yield {"host": "localhost", "port": http_port}

    # Cleanup: terminate server
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest.fixture
def clickhouse_db(clickhouse_server):
    """Create a clean test database for each test (function-scoped).

    Creates a unique database, yields connection string, then drops the database.
    """

    host = clickhouse_server["host"]
    port = clickhouse_server["port"]

    # Generate unique database name
    db_name = f"test_{uuid.uuid4().hex[:8]}"

    # Connect to default database to create test database
    default_conn_string = f"clickhouse://{host}:{port}/default"
    conn: Any = ibis.connect(default_conn_string)  # type: ignore[assignment]

    # Create test database
    conn.raw_sql(f"CREATE DATABASE {db_name}")  # type: ignore[attr-defined]

    # Return connection string for test database
    test_conn_string = f"clickhouse://{host}:{port}/{db_name}"

    yield test_conn_string

    # Cleanup: drop test database
    try:
        conn.raw_sql(f"DROP DATABASE IF EXISTS {db_name}")  # type: ignore[attr-defined]
    except Exception:
        logger.warning("Failed to drop test database %s: %s", db_name)


@pytest.fixture(scope="session")
def postgres_server(postgresql_proc: Any):
    """Expose connection details for the pytest-postgresql server."""
    host = postgresql_proc.host
    port = postgresql_proc.port
    user = postgresql_proc.user
    password = postgresql_proc.password
    admin_db = postgresql_proc.dbname
    options = postgresql_proc.options

    def build_conninfo(dbname: str) -> str:
        return psycopg_conninfo.make_conninfo(
            host=host,
            port=str(port),
            user=user,
            password=password or None,
            dbname=dbname,
            options=options or None,
        )

    admin_dsn = ""
    last_error: Exception | None = None
    for candidate in filter(None, [admin_db, "postgres"]):
        try:
            admin_dsn = build_conninfo(candidate)
            with psycopg.connect(admin_dsn):
                pass
        except psycopg.OperationalError as exc:
            last_error = exc
            continue
        admin_db = candidate
        break
    else:
        raise RuntimeError(
            "Unable to connect to Postgres admin database"
        ) from last_error

    return {
        "host": host,
        "port": port,
        "user": user,
        "dbname": admin_db,
        "password": password,
        "options": options,
        "dsn": admin_dsn,
        "psycopg": psycopg,
    }


@pytest.fixture
def postgres_db(postgres_server):
    """Create a clean PostgreSQL database for each test (function-scoped)."""
    psycopg = postgres_server["psycopg"]
    admin_dsn = postgres_server["dsn"]
    host = postgres_server["host"]
    port = postgres_server["port"]
    user = postgres_server["user"]
    password = postgres_server["password"]
    options = postgres_server.get("options")

    db_name = f"test_{uuid.uuid4().hex}"

    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{db_name}"')

    test_conn_string = _format_pg_connection_string(
        user=user,
        host=host,
        port=port,
        database=db_name,
        password=password,
        options=options,
    )

    yield test_conn_string

    try:
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
    except psycopg.Error as exc:
        logger.warning("Failed to drop test database %s: %s", db_name, exc)


@pytest.fixture
def store_params(tmp_path: Path, clickhouse_db: str) -> dict[str, Any]:
    """Provide all store-specific parameters in a single dict.

    Combines all store-specific fixtures (tmp_path, clickhouse_db, etc.) into
    one dictionary that can be passed to create_store().

    Args:
        tmp_path: Temporary directory for file-based stores
        clickhouse_db: ClickHouse connection string

    Returns:
        Dictionary with all available store parameters
    """
    return {
        "tmp_path": tmp_path,
        "clickhouse_db": clickhouse_db,
    }


# Store case functions for pytest-cases


class StoreCases:
    """Store configuration cases for parametrization."""

    def case_inmemory(
        self, test_graph: FeatureGraph
    ) -> tuple[type[MetadataStore], dict[str, Any]]:
        """InMemory store case."""
        # Registry is accessed globally via FeatureGraph.get_active()
        return (InMemoryMetadataStore, {})

    def case_duckdb(
        self, tmp_path: Path, test_graph: FeatureGraph
    ) -> tuple[type[MetadataStore], dict[str, Any]]:
        """DuckDB store case."""
        db_path = tmp_path / "test.duckdb"
        # Registry is accessed globally via FeatureGraph.get_active()
        return (DuckDBMetadataStore, {"database": db_path})

    def case_duckdb_ducklake(
        self, tmp_path: Path, test_graph: FeatureGraph
    ) -> tuple[type[MetadataStore], dict[str, Any]]:
        """DuckDB store configured with DuckLake attachment."""

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
        """ClickHouse store case."""
        # Registry is accessed globally via FeatureGraph.get_active()
        # clickhouse_db provides a clean database connection string
        return (ClickHouseMetadataStore, {"connection_string": clickhouse_db})


@fixture
@parametrize_with_cases("store_config", cases=StoreCases)
def persistent_store(
    store_config: tuple[type[MetadataStore], dict[str, Any]],
) -> MetadataStore:
    """Parametrized persistent store fixture.

    This fixture runs tests for all persistent store implementations.
    Returns an unopened store - tests should use it with a context manager.

    Usage:
        def test_something(persistent_store, test_graph):
            with persistent_store as store:
                # Test code runs for all store types
                # Access feature classes via test_graph.UpstreamFeatureA, etc.
    """
    store_type, config = store_config
    return store_type(**config)  # type: ignore[abstract]
