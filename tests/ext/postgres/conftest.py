"""PostgreSQL test fixtures."""

import uuid
from collections.abc import Generator

import pytest
from metaxy_testing.services import dagger_service, service_image


@pytest.fixture(scope="session")
def postgresql_server(pytestconfig: pytest.Config) -> Generator[str]:
    """Admin connection string of a PostgreSQL server for the test session, run as a Dagger service."""
    with dagger_service(
        service_image("postgres", pytestconfig.rootpath / "tests.toml"),
        [5432],
        env={"POSTGRES_HOST_AUTH_METHOD": "trust"},
        args=["postgres", "-c", "fsync=off", "-c", "synchronous_commit=off", "-c", "full_page_writes=off"],
    ) as ports:
        yield f"postgresql://postgres@127.0.0.1:{ports[5432]}"


@pytest.fixture(scope="function")
def postgresql_db(postgresql_server: str) -> Generator[str]:
    """Connection string of a fresh database, dropped after the test."""
    import psycopg
    from psycopg import sql

    test_db_name = f"metaxy_test_{uuid.uuid4().hex[:8]}"
    admin_conn_str = f"{postgresql_server}/postgres"

    with psycopg.connect(admin_conn_str, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_db_name)))

    yield f"{postgresql_server}/{test_db_name}"

    with psycopg.connect(admin_conn_str, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(test_db_name)))
