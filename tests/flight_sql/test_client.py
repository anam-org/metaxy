"""Tests for Flight SQL client functionality."""

import polars as pl
import pytest

import metaxy as mx
from metaxy import HashAlgorithm
from metaxy._testing import add_metaxy_provenance_column
from metaxy.flight_sql import FlightSQLMetadataStore, MetaxyFlightSQLServer
from metaxy.metadata_store.duckdb import DuckDBMetadataStore


class SimpleFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="simple",
        id_columns=["sample_id"],
        fields=["value"],
    ),
):
    """Simple feature for testing."""

    sample_id: int
    value: str


@pytest.fixture
def backend_store(tmp_path):
    """Create a DuckDB backend store with test data."""
    store = DuckDBMetadataStore(
        database=tmp_path / "backend.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    # Write test data
    with SimpleFeature.graph.use(), store:
        test_data = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "value": ["a", "b", "c"],
                "metaxy_provenance_by_field": [
                    {"value": "h1"},
                    {"value": "h2"},
                    {"value": "h3"},
                ],
            }
        )
        metadata = add_metaxy_provenance_column(test_data, SimpleFeature)
        store.write_metadata(SimpleFeature, metadata)

    return store


@pytest.fixture
def flight_server(backend_store):
    """Create a Flight SQL server with backend store."""
    import socket

    # Find an available port
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    server = MetaxyFlightSQLServer(
        location=f"grpc://localhost:{port}",
        store=backend_store,
    )

    # Open the backend store and server
    with backend_store, server:
        yield server


def test_client_creation():
    """Test that Flight SQL client can be created."""
    client = FlightSQLMetadataStore(url="grpc://localhost:8815")

    assert client is not None
    assert client.url == "grpc://localhost:8815"
    assert not client._is_open


def test_client_context_manager():
    """Test that client works as context manager."""
    client = FlightSQLMetadataStore(url="grpc://localhost:8815")

    assert not client._is_open

    # This will fail to connect, but we're testing the context manager
    try:
        with client:
            assert client._is_open
    except Exception:
        pass  # Expected - no server running

    # Should be closed after context
    assert not client._is_open


def test_client_read_metadata_sql(flight_server):
    """Test that client can execute SQL queries via Flight."""
    # Create client pointing to server
    client = FlightSQLMetadataStore(url=str(flight_server.location.uri.decode()))

    with client:
        # Execute query (table names don't have __key suffix in DuckDB)
        result = client.read_metadata_sql("SELECT * FROM simple ORDER BY sample_id")

        # Verify results
        assert len(result) == 3
        assert result["sample_id"].to_list() == [1, 2, 3]
        assert result["value"].to_list() == ["a", "b", "c"]


def test_client_query_with_filter(flight_server):
    """Test that client can execute filtered queries."""
    client = FlightSQLMetadataStore(url=str(flight_server.location.uri.decode()))

    with client:
        result = client.read_metadata_sql("SELECT * FROM simple WHERE sample_id <= 2 ORDER BY sample_id")

        assert len(result) == 2
        assert result["sample_id"].to_list() == [1, 2]
        assert result["value"].to_list() == ["a", "b"]


def test_client_aggregation_query(flight_server):
    """Test that client can execute aggregation queries."""
    client = FlightSQLMetadataStore(url=str(flight_server.location.uri.decode()))

    with client:
        result = client.read_metadata_sql("SELECT COUNT(*) as count FROM simple")

        assert len(result) == 1
        assert result["count"][0] == 3


def test_client_requires_open():
    """Test that client requires being open before queries."""
    client = FlightSQLMetadataStore(url="grpc://localhost:8815")

    with pytest.raises(RuntimeError, match="not open"):
        client.read_metadata_sql("SELECT 1")


def test_client_config_model():
    """Test that client has correct config model."""
    from metaxy.flight_sql.client import FlightSQLClientConfig

    config_class = FlightSQLMetadataStore.config_model()
    assert config_class == FlightSQLClientConfig

    # Test config creation
    config = config_class(url="grpc://localhost:8815")
    assert config.url == "grpc://localhost:8815"
