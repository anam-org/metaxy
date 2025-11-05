"""PostgreSQL-specific tests that don't apply to other stores."""

from typing import cast

import pytest

pytest.importorskip("ibis")

try:
    import ibis.backends.postgres  # noqa: F401
except ImportError:
    pytest.skip("ibis-postgres not installed", allow_module_level=True)

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store.postgres import PostgresMetadataStore


def test_postgres_initialization_with_params() -> None:
    """Test initialization with explicit connection parameters."""
    store = PostgresMetadataStore(
        host="localhost",
        database="metaxy",
        user="ml",
        password="secret",
    )

    assert store.host == "localhost"
    assert store.port == 5432
    assert store.database == "metaxy"
    assert store.schema is None

    display = store.display()
    assert "PostgresMetadataStore" in display
    assert "database=metaxy" in display


def test_postgres_requires_configuration() -> None:
    """Test that configuration is required."""
    with pytest.raises(ValueError, match="Must provide either"):
        PostgresMetadataStore()


def test_postgres_respects_custom_port_and_schema() -> None:
    """Test that custom ports and schemas are preserved."""
    store = PostgresMetadataStore(
        host="localhost",
        port=5433,
        database="metaxy",
        schema="features",
    )

    assert store.port == 5433
    assert store.schema == "features"

    display = store.display()
    assert "port=5433" in display
    assert "schema=features" in display


def test_postgres_hash_sql_generators_include_sha256() -> None:
    """Test SHA256 hash SQL generation."""
    store = PostgresMetadataStore(host="localhost", database="metaxy")

    generators = store._get_hash_sql_generators()
    assert HashAlgorithm.SHA256 in generators

    class DummyTable:
        def compile(self) -> str:
            return "SELECT * FROM dummy"

    import ibis.expr.types as ir

    dummy_table = cast(ir.Table, cast(object, DummyTable()))
    sql = generators[HashAlgorithm.SHA256](
        dummy_table, {"field1": "col1", "field2": "col2"}
    )
    assert "DIGEST(col1, 'sha256')" in sql
    assert "DIGEST(col2, 'sha256')" in sql
    assert "__hash_field1" in sql
    assert "__hash_field2" in sql


def test_postgres_display_with_connection_string() -> None:
    """Test display output when initialized with connection string."""
    store = PostgresMetadataStore(
        "postgresql://user:pass@localhost:5432/metaxy",
    )

    display = store.display()
    assert "connection_string=postgresql://user:pass@localhost:5432/metaxy" in display
