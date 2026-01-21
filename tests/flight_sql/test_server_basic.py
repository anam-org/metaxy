"""Basic tests for Flight SQL server functionality."""

import pytest

from metaxy import HashAlgorithm
from metaxy.flight_sql import MetaxyFlightSQLServer
from metaxy.metadata_store.duckdb import DuckDBMetadataStore


@pytest.fixture
def empty_store(tmp_path):
    """Create an empty DuckDB metadata store for testing."""
    return DuckDBMetadataStore(
        database=tmp_path / "test.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )


def test_server_creation(empty_store):
    """Test that Flight SQL server can be created."""
    server = MetaxyFlightSQLServer(
        location="grpc://localhost:0",
        store=empty_store,
    )

    assert server is not None
    assert server._store is empty_store


def test_server_context_manager(empty_store):
    """Test that server works as context manager."""
    server = MetaxyFlightSQLServer(
        location="grpc://localhost:0",
        store=empty_store,
    )

    # Store should not be open initially
    assert not empty_store._is_open

    with server:
        # Store should be opened by server
        assert empty_store._is_open

    # Store should be closed after context
    assert not empty_store._is_open


def test_server_location_parsing(empty_store):
    """Test that server correctly parses location strings."""
    server = MetaxyFlightSQLServer(
        location="grpc://localhost:8815",
        store=empty_store,
    )

    assert server.location is not None
    # Location should be a Flight Location object
    assert hasattr(server.location, "uri")
