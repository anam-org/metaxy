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

import datetime

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.postgresql import _METAXY_STRUCT_COLUMNS, PostgreSQLMetadataStore
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
        store.write(test_features["UpstreamFeatureA"], metadata)

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
        store.write(test_features["UpstreamFeatureA"], original_metadata)

        # Read data back and verify roundtrip
        result = collect_to_polars(store.read(test_features["UpstreamFeatureA"]))
        expected = original_metadata.sort("sample_uid")
        result_sorted = result.select(expected.columns).sort("sample_uid")
        assert_frame_equal(result_sorted, expected)


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
        store1.write(test_features["UpstreamFeatureA"], metadata)

    # Read data in second instance
    with PostgreSQLMetadataStore(clean_postgres_db) as store2:
        result = collect_to_polars(store2.read(test_features["UpstreamFeatureA"]))

        assert len(result) == 3
        assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_postgresql_uses_ibis_backend(postgresql_db: str, test_graph) -> None:
    """Test that PostgreSQL store uses Ibis backend."""
    with PostgreSQLMetadataStore(postgresql_db) as store:
        # Should have conn
        assert hasattr(store, "conn")
        # Backend should be postgres
        assert store._conn is not None


def test_postgresql_auto_cast_false_only_converts_system_columns(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """With auto_cast=False, user struct columns stay as JSON strings on read.

    When auto_cast=False, transform_before_write only encodes system struct columns.
    User struct columns must be pre-encoded to JSON strings before write.
    On read, only system columns are decoded back to Structs.
    """
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=False) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
                # Pre-encode user struct column (auto_cast=False won't do it)
                "user_metadata": [
                    '{"model": "resnet", "version": "1"}',
                    '{"model": "vgg", "version": "2"}',
                ],
            }
        )
        store.write(test_features["UpstreamFeatureA"], metadata)

        result = collect_to_polars(store.read(test_features["UpstreamFeatureA"]))

        # System column should be decoded back to Struct
        assert isinstance(result.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
        # User column should remain as String (not decoded)
        assert result.schema["user_metadata"] == pl.Utf8


def test_postgresql_default_hash_algorithm(postgresql_db: str) -> None:
    """Test that PostgreSQL defaults to XXHASH32 (computed in Polars)."""
    with PostgreSQLMetadataStore(postgresql_db) as store:
        assert store._get_default_hash_algorithm() == HashAlgorithm.XXHASH32


@pytest.mark.parametrize(
    ("col_name", "col_dtype", "auto_cast", "expected"),
    [
        # Metaxy system columns: always converted when JSON or String
        (next(iter(_METAXY_STRUCT_COLUMNS)), dt.JSON(), True, True),
        (next(iter(_METAXY_STRUCT_COLUMNS)), dt.JSON(), False, True),
        (next(iter(_METAXY_STRUCT_COLUMNS)), dt.String(), True, True),
        (next(iter(_METAXY_STRUCT_COLUMNS)), dt.String(), False, True),
        (next(iter(_METAXY_STRUCT_COLUMNS)), dt.Int64(), True, False),
        # User columns: only dt.JSON when auto_cast=True
        ("user_col", dt.JSON(), True, True),
        ("user_col", dt.JSON(), False, False),
        ("user_col", dt.String(), True, False),
        ("user_col", dt.String(), False, False),
        ("user_col", dt.Int64(), True, False),
        ("user_col", dt.Float64(), False, False),
    ],
    ids=[
        "metaxy-json-auto_on",
        "metaxy-json-auto_off",
        "metaxy-string-auto_on",
        "metaxy-string-auto_off",
        "metaxy-int64-auto_on",
        "user-json-auto_on",
        "user-json-auto_off",
        "user-string-auto_on",
        "user-string-auto_off",
        "user-int64-auto_on",
        "user-float64-auto_off",
    ],
)
def test_get_json_columns_for_struct(col_name: str, col_dtype: dt.DataType, auto_cast: bool, expected: bool) -> None:
    """Verify _get_json_columns_for_struct correctly identifies columns by type and auto_cast setting."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    store.auto_cast_struct_for_jsonb = auto_cast
    schema = sch.Schema.from_tuples([(col_name, col_dtype)])
    result = store._get_json_columns_for_struct(schema)
    assert (col_name in result) == expected


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (42, pl.Int64()),
        (3.14, pl.Float64()),
        ("hello", pl.String()),
        (True, pl.Boolean()),
        (datetime.date(2024, 1, 15), pl.Date()),
    ],
    ids=["int64", "float64", "string", "boolean", "date"],
)
def test_postgresql_scalar_type_roundtrip(
    clean_postgres_db: str, test_graph, test_features: dict, value: object, dtype: pl.DataType
) -> None:
    """Scalar Polars types survive a write/read roundtrip through PostgreSQL."""
    with PostgreSQLMetadataStore(clean_postgres_db) as store:
        df = pl.DataFrame(
            {
                "sample_uid": [1],
                METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h1"}],
                "extra": pl.Series([value], dtype=dtype),
            }
        )
        store.write(test_features["UpstreamFeatureA"], df)
        result = collect_to_polars(store.read(test_features["UpstreamFeatureA"]))

        assert result["extra"][0] == value


def test_postgresql_struct_column_stored_as_text(clean_postgres_db: str, test_graph, test_features: dict) -> None:
    """User Struct columns are JSON-encoded on write but stored as TEXT, not JSONB.

    With the narrowed read path, only dt.JSON columns are auto-decoded for user data.
    Struct columns written via json_encode() become TEXT in PostgreSQL, so they
    remain as strings on read. Metaxy system columns (also TEXT) are still decoded.
    """
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=True) as store:
        df = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
                "user_struct": [
                    {"model": "resnet", "version": "1"},
                    {"model": "vgg", "version": "2"},
                ],
            }
        )
        store.write(test_features["UpstreamFeatureA"], df)
        result = collect_to_polars(store.read(test_features["UpstreamFeatureA"]))
        result_sorted = result.sort("sample_uid")

        # Metaxy system column is decoded back to Struct
        assert isinstance(result_sorted.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
        # User struct is stored as TEXT → remains String on read (not auto-decoded)
        assert result_sorted.schema["user_struct"] == pl.String
