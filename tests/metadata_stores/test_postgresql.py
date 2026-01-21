"""PostgreSQL-specific tests that don't apply to other stores.

These tests focus on PostgreSQL-specific functionality like connection handling,
struct ↔ JSONB serialization, and hash algorithms.

Main provenance and integration tests run automatically via test_provenance_golden_reference.py
when PostgreSQL is included in the `any_store` fixture.

Requirements:
    - pytest-postgresql installed (preferred for automatic test database setup)
    - OR set POSTGRES_TEST_URL environment variable
    - OR set PG_BIN environment variable to point to PostgreSQL bin directory
"""

import polars as pl
import pytest

# Skip all tests in this module if Ibis PostgreSQL backend not available
pytest.importorskip("ibis")

try:
    import ibis.backends.postgres  # noqa: F401
except ImportError:
    pytest.skip("ibis-postgres not installed", allow_module_level=True)

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.postgresql import PostgreSQLMetadataStore
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
from metaxy.versioning.types import HashAlgorithm


@pytest.fixture
def clean_postgres_db(postgresql_db: str, test_graph):
    """Provide a clean PostgreSQL database for each test."""
    with PostgreSQLMetadataStore(postgresql_db) as store:
        for table_name in store.conn.list_tables():
            if table_name.startswith("test_stores__"):
                store.conn.drop_table(table_name, force=True)
    yield postgresql_db


def test_postgresql_native_implementation(postgresql_db: str, test_graph) -> None:
    """Test that PostgreSQL store always uses Polars engine for versioning."""
    import narwhals as nw

    with PostgreSQLMetadataStore(postgresql_db) as store:
        assert store.native_implementation() == nw.Implementation.POLARS


def test_postgresql_connection_string_init(postgresql_db: str, test_graph) -> None:
    """Test initialization with connection string."""
    store = PostgreSQLMetadataStore(postgresql_db)
    assert not store._is_open

    with store.open():
        assert store._is_open
        assert store._conn is not None


def test_postgresql_connection_params_init() -> None:
    """Test initialization with connection params dict."""
    params = {
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_pass",
    }

    store = PostgreSQLMetadataStore(connection_params=params)
    assert not store._is_open


def test_postgresql_requires_connection_info() -> None:
    """Test that store requires either connection_string or connection_params."""
    with pytest.raises(ValueError, match="Must provide either connection_string or connection_params"):
        PostgreSQLMetadataStore()


def test_postgresql_table_naming(clean_postgres_db: str, test_graph, test_features: dict) -> None:
    """Test that feature keys are converted to table names correctly."""
    with PostgreSQLMetadataStore(clean_postgres_db) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], metadata)

        table_names = store.conn.list_tables()
        assert "test_stores__upstream_a" in table_names


def test_postgresql_struct_to_jsonb_roundtrip(clean_postgres_db: str, test_graph, test_features: dict) -> None:
    """Test that struct columns serialize to JSONB and parse back correctly.

    This is critical for PostgreSQL - verifies the struct ↔ JSONB conversion layer works.
    """
    with PostgreSQLMetadataStore(clean_postgres_db) as store:
        # Write data with struct columns
        original_metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "hash1", "audio": "hash2"},
                    {"frames": "hash3", "audio": "hash4"},
                    {"frames": "hash5", "audio": "hash6"},
                ],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], original_metadata)

        # Read data back
        result = collect_to_polars(store.read_metadata(test_features["UpstreamFeatureA"]))

        # Verify struct fields match
        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}

        # Check struct column type (JSONB doesn't preserve field order, so check fields exist)
        schema_type = result.schema[METAXY_PROVENANCE_BY_FIELD]
        assert isinstance(schema_type, pl.Struct)
        field_names = {field.name for field in schema_type.fields}
        assert field_names == {"audio", "frames"}
        assert all(field.dtype == pl.Utf8 for field in schema_type.fields)

        # Verify struct values (order may vary due to deduplication)
        prov_values = result[METAXY_PROVENANCE_BY_FIELD].to_list()
        expected_values = [
            {"frames": "hash1", "audio": "hash2"},
            {"frames": "hash3", "audio": "hash4"},
            {"frames": "hash5", "audio": "hash6"},
        ]
        assert set(map(lambda x: frozenset(x.items()), prov_values)) == set(
            map(lambda x: frozenset(x.items()), expected_values)
        )


def test_postgresql_persistence(clean_postgres_db: str, test_graph, test_features: dict) -> None:
    """Test that data persists across different store instances."""
    # Write data in first instance
    with PostgreSQLMetadataStore(clean_postgres_db) as store1:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                    {"frames": "h3", "audio": "h3"},
                ],
            }
        )
        store1.write_metadata(test_features["UpstreamFeatureA"], metadata)

    # Read data in second instance
    with PostgreSQLMetadataStore(clean_postgres_db) as store2:
        result = collect_to_polars(store2.read_metadata(test_features["UpstreamFeatureA"]))

        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_postgresql_uses_ibis_backend(postgresql_db: str, test_graph) -> None:
    """Test that PostgreSQL store uses Ibis backend."""
    with PostgreSQLMetadataStore(postgresql_db) as store:
        # Should have conn
        assert hasattr(store, "conn")
        # Backend should be postgres
        assert store._conn is not None


def test_postgresql_default_hash_algorithm(postgresql_db: str) -> None:
    """Test that PostgreSQL defaults to XXHASH64 (computed in Polars)."""
    with PostgreSQLMetadataStore(postgresql_db) as store:
        assert store._get_default_hash_algorithm() == HashAlgorithm.XXHASH64
