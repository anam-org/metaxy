"""PostgreSQL test fixtures."""

import os
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import pytest

# Force pytest-postgresql to use a short socket directory to avoid hitting the
# 103-character Unix socket limit enforced by PostgreSQL on macOS.
# Use system temp directory for cross-platform compatibility (Windows, Linux, macOS)
_PG_SOCKET_DIR = Path(tempfile.gettempdir()) / "metaxy-pg"
_PG_SOCKET_DIR.mkdir(parents=True, exist_ok=True)


def _with_search_path(connection_url: str, schema_name: str) -> str:
    """Inject PostgreSQL search_path into a connection URL's query params."""
    parsed_url = urlparse(connection_url)
    query_pairs = parse_qsl(parsed_url.query, keep_blank_values=True)

    search_path_option = f"-csearch_path={schema_name}"
    existing_options = [value for key, value in query_pairs if key == "options" and value]
    options_value = f"{' '.join(existing_options)} {search_path_option}" if existing_options else search_path_option

    non_options_pairs = [(key, value) for key, value in query_pairs if key != "options"]
    updated_query = urlencode([*non_options_pairs, ("options", options_value)], doseq=True, quote_via=quote)
    return urlunparse(parsed_url._replace(query=updated_query))


# --- SETUP POSTGRES PATH ---
_nix_pg_bin = os.environ.get("PG_BIN")
_pg_available = False

if _nix_pg_bin:
    _pg_executable = str(Path(_nix_pg_bin) / "pg_ctl")
    _pg_available = Path(_pg_executable).exists()
else:
    import shutil

    _pg_executable = shutil.which("pg_ctl") or "pg_ctl"
    _pg_available = shutil.which("pg_ctl") is not None


# --- FIXTURE CONFIGURATION ---
if _pg_available:
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
    def postgresql_db(postgresql_proc) -> Generator[str]:
        """PostgreSQL database connection string fixture.

        Returns connection string for a test PostgreSQL instance.
        Creates a fresh database for each test to ensure isolation.
        """
        import psycopg
        from psycopg import sql

        # Use unique database name for each test
        test_db_name = f"metaxy_test_{uuid.uuid4().hex[:8]}"

        # Connect to postgres database to create test database
        admin_conn_str = f"postgresql://{postgresql_proc.user}@{postgresql_proc.host}:{postgresql_proc.port}/postgres"
        conn = psycopg.connect(admin_conn_str, autocommit=True)
        try:
            # Use sql.Identifier for safe database name quoting
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_db_name)))
        finally:
            conn.close()

        conn_str = f"postgresql://{postgresql_proc.user}@{postgresql_proc.host}:{postgresql_proc.port}/{test_db_name}"

        yield conn_str

        # Cleanup: force disconnect all connections and drop database
        conn = psycopg.connect(admin_conn_str, autocommit=True)
        try:
            conn.execute(
                sql.SQL(
                    """
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = {}
                      AND pid <> pg_backend_pid()
                    """
                ).format(sql.Literal(test_db_name))
            )
            conn.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(test_db_name)))
        finally:
            conn.close()

else:
    _postgres_test_url = os.environ.get("POSTGRES_TEST_URL")

    if _postgres_test_url:
        _pg_url: str = _postgres_test_url

        @pytest.fixture(scope="function")
        def postgresql_db() -> Generator[str, None, None]:
            """PostgreSQL connection string from POSTGRES_TEST_URL with per-test schema isolation."""
            import psycopg
            from psycopg import sql

            schema_name = f"metaxy_test_{uuid.uuid4().hex[:8]}"
            isolated_url = _with_search_path(_pg_url, schema_name)

            conn = psycopg.connect(_pg_url, autocommit=True)
            try:
                conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
            finally:
                conn.close()

            yield isolated_url

            conn = psycopg.connect(_pg_url, autocommit=True)
            try:
                conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema_name)))
            finally:
                conn.close()

    else:

        @pytest.fixture(scope="function")
        def postgresql_db() -> str:
            """PostgreSQL not available - skip tests."""
            pytest.skip("PostgreSQL not available. Set PG_BIN, install PostgreSQL, or set POSTGRES_TEST_URL.")
