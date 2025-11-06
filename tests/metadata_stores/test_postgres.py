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


def test_postgres_default_hash_algorithm_is_md5() -> None:
    """Test that default hash algorithm is MD5 (no extension required)."""
    store = PostgresMetadataStore(host="localhost", database="metaxy")

    assert store._get_default_hash_algorithm() == HashAlgorithm.MD5
    # Default hash algorithm should be MD5 unless explicitly overridden
    assert store.hash_algorithm == HashAlgorithm.MD5


def test_postgres_enable_pgcrypto_parameter() -> None:
    """Test that enable_pgcrypto parameter is stored correctly."""
    store_with_pgcrypto = PostgresMetadataStore(
        host="localhost", database="metaxy", enable_pgcrypto=True
    )
    store_without_pgcrypto = PostgresMetadataStore(
        host="localhost", database="metaxy", enable_pgcrypto=False
    )

    assert store_with_pgcrypto.enable_pgcrypto is True
    assert store_without_pgcrypto.enable_pgcrypto is False


def test_postgres_sha256_with_explicit_hash_algorithm() -> None:
    """Test that SHA256 can be explicitly set as hash algorithm."""
    store = PostgresMetadataStore(
        host="localhost",
        database="metaxy",
        hash_algorithm=HashAlgorithm.SHA256,
    )

    assert store.hash_algorithm == HashAlgorithm.SHA256
    # Should have SHA256 generator
    generators = store._get_hash_sql_generators()
    assert HashAlgorithm.SHA256 in generators
    # Should also have MD5 (inherited from base)
    assert HashAlgorithm.MD5 in generators
