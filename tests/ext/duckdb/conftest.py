"""DuckDB and MotherDuck test fixtures."""

import os
import uuid
from collections.abc import Generator

import duckdb
import pytest

# ============= MOTHERDUCK FIXTURES =============


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
