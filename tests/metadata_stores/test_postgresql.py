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

from urllib.parse import parse_qsl, urlparse

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import narwhals as nw
import polars as pl
import polars.testing
import pytest

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError
from metaxy.metadata_store.postgresql import PostgreSQLMetadataStore
from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD
from metaxy.models.feature import current_graph
from metaxy.models.types import FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm
from tests.metadata_stores.conftest import _with_search_path


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


def test_with_search_path_adds_options_when_absent() -> None:
    """Search path option should be added while preserving existing query params."""
    base_url = "postgresql://postgres@localhost:5432/test_db?sslmode=require&connect_timeout=10"

    result = _with_search_path(base_url, "test_schema")
    parsed = urlparse(result)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))

    assert query_params["sslmode"] == "require"
    assert query_params["connect_timeout"] == "10"
    assert query_params["options"] == "-csearch_path=test_schema"


def test_with_search_path_appends_to_existing_options_with_percent20() -> None:
    """Existing options should keep a real space separator encoded as %20 (not +)."""
    base_url = "postgresql://postgres@localhost:5432/test_db?sslmode=require&options=-cstatement_timeout%3D5"

    result = _with_search_path(base_url, "isolated_schema")
    parsed = urlparse(result)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))

    assert query_params["sslmode"] == "require"
    assert query_params["options"] == "-cstatement_timeout=5 -csearch_path=isolated_schema"
    assert "%20-csearch_path%3Disolated_schema" in parsed.query
    assert "+-csearch_path=isolated_schema" not in parsed.query


def test_postgresql_struct_to_jsonb_roundtrip(clean_postgres_db: str, test_graph, test_features: dict) -> None:
    """Test that struct columns serialize to JSONB and parse back correctly.

    This is critical for PostgreSQL - verifies the struct ↔ JSONB conversion layer works.
    """
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="w") as store:
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

        # Read data back and verify full roundtrip of user-provided columns
        # JSONB does not preserve struct field order, so compare with check_dtypes=False
        result = collect_to_polars(store.read(test_features["UpstreamFeatureA"]))
        result_sorted = result.sort("sample_uid").select(original_metadata.columns)
        pl.testing.assert_frame_equal(result_sorted, original_metadata.sort("sample_uid"), check_dtypes=False)


def test_postgresql_uses_ibis_backend(postgresql_db: str, test_graph) -> None:
    """Test that PostgreSQL store uses Ibis backend."""
    with PostgreSQLMetadataStore(postgresql_db) as store:
        # Should have conn
        assert hasattr(store, "conn")
        # Backend should be postgres
        assert store._conn is not None


def test_postgresql_create_versioning_engine_uses_polars_without_hash_functions(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """PostgreSQL should construct PolarsVersioningEngine without hash_functions kwarg."""
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="w") as store:
        plan = store._resolve_feature_plan(test_features["UpstreamFeatureA"].key)
        with store._create_versioning_engine(plan) as engine:
            assert isinstance(engine, PolarsVersioningEngine)


def test_postgresql_auto_cast_false_only_converts_system_columns(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """With auto_cast=False, transform_before_write leaves user Struct columns unchanged."""
    feature = test_features["UpstreamFeatureA"]
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=False).open(mode="w") as store:
        metadata = pl.DataFrame(
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
        transformed = collect_to_polars(
            store.transform_before_write(nw.from_native(metadata), feature.key, store.get_table_name(feature.key))
        )

        assert transformed.schema[METAXY_PROVENANCE_BY_FIELD] == pl.Utf8
        assert isinstance(transformed.schema["user_struct"], pl.Struct)
        assert transformed["user_struct"].to_list() == [
            {"model": "resnet", "version": "1"},
            {"model": "vgg", "version": "2"},
        ]


def test_postgresql_auto_cast_false_only_decodes_system_columns_on_read(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """With auto_cast=False, user TEXT JSON stays as text while system columns still decode."""
    feature = test_features["UpstreamFeatureA"]

    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=False).open(mode="r") as store:
        table_name = store.get_table_name(feature.key)
        feature_version = current_graph().get_feature_version(feature.key)
        project_version = current_graph().project_version

        store.conn.raw_sql(f"DROP TABLE IF EXISTS {table_name}")  # ty: ignore[unresolved-attribute]
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                user_struct TEXT,
                {METAXY_PROVENANCE_BY_FIELD} TEXT,
                metaxy_provenance TEXT,
                metaxy_feature_version TEXT,
                metaxy_project_version TEXT,
                {METAXY_DATA_VERSION_BY_FIELD} TEXT,
                metaxy_data_version TEXT,
                metaxy_created_at TIMESTAMPTZ,
                metaxy_updated_at TIMESTAMPTZ,
                metaxy_deleted_at TIMESTAMPTZ,
                metaxy_materialization_id TEXT
            )
            """
        )
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            INSERT INTO {table_name} (
                sample_uid,
                user_struct,
                {METAXY_PROVENANCE_BY_FIELD},
                metaxy_provenance,
                metaxy_feature_version,
                metaxy_project_version,
                {METAXY_DATA_VERSION_BY_FIELD},
                metaxy_data_version,
                metaxy_created_at,
                metaxy_updated_at,
                metaxy_deleted_at,
                metaxy_materialization_id
            ) VALUES
            (
                1,
                '{{"model":"resnet","version":"1"}}',
                '{{"frames":"h1","audio":"h2"}}',
                'prov1',
                '{feature_version}',
                '{project_version}',
                '{{"frames":"v1","audio":"v1"}}',
                'dv1',
                now(),
                now(),
                NULL,
                'm1'
            ),
            (
                2,
                '{{"model":"vgg","version":"2"}}',
                '{{"frames":"h3","audio":"h4"}}',
                'prov2',
                '{feature_version}',
                '{project_version}',
                '{{"frames":"v2","audio":"v2"}}',
                'dv2',
                now(),
                now(),
                NULL,
                'm1'
            )
            """
        )

        result = collect_to_polars(store.read(feature)).sort("sample_uid")

        assert isinstance(result.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
        assert result.schema["user_struct"] == pl.Utf8
        assert result.sort("sample_uid")["user_struct"].to_list() == [
            '{"model":"resnet","version":"1"}',
            '{"model":"vgg","version":"2"}',
        ]


def test_postgresql_user_defined_jsonb_column(clean_postgres_db: str, test_graph, test_features: dict) -> None:
    """User-defined JSONB columns should decode back to Polars Struct on read."""
    feature = test_features["UpstreamFeatureA"]

    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=True).open(mode="r") as store:
        table_name = store.get_table_name(feature.key)
        feature_version = current_graph().get_feature_version(feature.key)
        project_version = current_graph().project_version

        store.conn.raw_sql(f"DROP TABLE IF EXISTS {table_name}")  # ty: ignore[unresolved-attribute]
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                user_metadata JSONB,
                {METAXY_PROVENANCE_BY_FIELD} JSONB,
                metaxy_provenance TEXT,
                metaxy_feature_version TEXT,
                metaxy_project_version TEXT,
                {METAXY_DATA_VERSION_BY_FIELD} JSONB,
                metaxy_data_version TEXT,
                metaxy_created_at TIMESTAMPTZ,
                metaxy_updated_at TIMESTAMPTZ,
                metaxy_deleted_at TIMESTAMPTZ,
                metaxy_materialization_id TEXT
            )
            """
        )
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            INSERT INTO {table_name} (
                sample_uid,
                user_metadata,
                {METAXY_PROVENANCE_BY_FIELD},
                metaxy_provenance,
                metaxy_feature_version,
                metaxy_project_version,
                {METAXY_DATA_VERSION_BY_FIELD},
                metaxy_data_version,
                metaxy_created_at,
                metaxy_updated_at,
                metaxy_deleted_at,
                metaxy_materialization_id
            ) VALUES
            (
                1,
                '{{"source": "camera1", "quality": "high"}}'::jsonb,
                '{{"frames": "hash1", "audio": "hash2"}}'::jsonb,
                'prov1',
                '{feature_version}',
                '{project_version}',
                '{{"frames": "v1", "audio": "v1"}}'::jsonb,
                'dv1',
                now(),
                now(),
                NULL,
                'm1'
            ),
            (
                2,
                '{{"source": "camera2", "resolution": "4k"}}'::jsonb,
                '{{"frames": "hash3", "audio": "hash4"}}'::jsonb,
                'prov2',
                '{feature_version}',
                '{project_version}',
                '{{"frames": "v2", "audio": "v2"}}'::jsonb,
                'dv2',
                now(),
                now(),
                NULL,
                'm1'
            )
            """
        )

        result = collect_to_polars(store.read(feature)).sort("sample_uid")

        assert isinstance(result.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
        assert isinstance(result.schema["user_metadata"], pl.Struct)
        assert result["user_metadata"].to_list() == [
            {"source": "camera1", "quality": "high", "resolution": None},
            {"source": "camera2", "quality": None, "resolution": "4k"},
        ]


@pytest.mark.parametrize(
    ("user_column_type", "expected_pg_type"),
    [("JSONB", "jsonb"), ("JSON", "json")],
    ids=["jsonb", "json"],
)
def test_postgresql_auto_cast_struct_for_json_columns(
    clean_postgres_db: str,
    test_graph,
    test_features: dict,
    user_column_type: str,
    expected_pg_type: str,
) -> None:
    """auto_cast_struct_for_jsonb=True should adapt Structs to the target PostgreSQL JSON type."""
    feature = test_features["UpstreamFeatureA"]

    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=True).open(mode="w") as store:
        table_name = store.get_table_name(feature.key)

        store.conn.raw_sql(f"DROP TABLE IF EXISTS {table_name}")  # ty: ignore[unresolved-attribute]
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                user_tags {user_column_type},
                {METAXY_PROVENANCE_BY_FIELD} JSONB,
                metaxy_provenance TEXT,
                metaxy_feature_version TEXT,
                metaxy_project_version TEXT,
                {METAXY_DATA_VERSION_BY_FIELD} JSONB,
                metaxy_data_version TEXT,
                metaxy_created_at TIMESTAMPTZ,
                metaxy_updated_at TIMESTAMPTZ,
                metaxy_deleted_at TIMESTAMPTZ,
                metaxy_materialization_id TEXT
            )
            """
        )

        df = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "user_tags": [
                    {"source": "camera1", "quality": "high"},
                    {"source": "camera2", "quality": "medium"},
                ],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
            }
        )
        store.write(feature, df)

        result = collect_to_polars(store.read(feature)).sort("sample_uid")
        raw_types = store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"SELECT pg_typeof(user_tags)::text FROM {table_name} ORDER BY sample_uid"
        ).fetchall()

        assert isinstance(result.schema["user_tags"], pl.Struct)
        assert result["user_tags"].to_list() == [
            {"source": "camera1", "quality": "high"},
            {"source": "camera2", "quality": "medium"},
        ]
        assert raw_types == [(expected_pg_type,), (expected_pg_type,)]


def test_postgresql_default_hash_algorithm(postgresql_db: str) -> None:
    """Test that PostgreSQL defaults to XXHASH32 (computed in Polars)."""
    with PostgreSQLMetadataStore(postgresql_db) as store:
        assert store._get_default_hash_algorithm() == HashAlgorithm.XXHASH32


def test_postgresql_validate_hash_algorithm_uses_polars_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation should use PolarsVersioningEngine.supported_hash_algorithms()."""
    store = PostgreSQLMetadataStore(
        connection_params={"host": "localhost", "database": "dummy"},
        hash_algorithm=HashAlgorithm.SHA256,
    )

    monkeypatch.setattr(
        PolarsVersioningEngine,
        "supported_hash_algorithms",
        classmethod(lambda cls: frozenset({HashAlgorithm.MD5})),
    )

    with pytest.raises(HashAlgorithmNotSupportedError, match="Supported algorithms: md5"):
        store._validate_hash_algorithm_support()


def test_postgresql_hard_delete_with_single_filter(clean_postgres_db: str, test_graph, test_features: dict) -> None:
    """Hard delete should remove rows matching a single Narwhals predicate."""
    feature = test_features["UpstreamFeatureA"]
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="w") as store:
        store.write(
            feature,
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"frames": "h1", "audio": "h1"},
                        {"frames": "h2", "audio": "h2"},
                        {"frames": "h3", "audio": "h3"},
                    ],
                }
            ),
        )

        store.delete(
            feature,
            filters=[nw.col("sample_uid") == 2],
            soft=False,
            with_feature_history=True,
        )

        result = collect_to_polars(
            store.read(
                feature,
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )
        ).sort("sample_uid")
        assert result["sample_uid"].to_list() == [1, 3]


def test_postgresql_hard_delete_with_multiple_filters(clean_postgres_db: str, test_graph, test_features: dict) -> None:
    """Hard delete should combine multiple Narwhals predicates with AND semantics."""
    feature = test_features["UpstreamFeatureA"]
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="w") as store:
        store.write(
            feature,
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3, 4],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"frames": "h1", "audio": "h1"},
                        {"frames": "h2", "audio": "h2"},
                        {"frames": "h3", "audio": "h3"},
                        {"frames": "h4", "audio": "h4"},
                    ],
                }
            ),
        )

        store.delete(
            feature,
            filters=[nw.col("sample_uid") >= 2, nw.col("sample_uid") < 4],
            soft=False,
            with_feature_history=True,
        )

        result = collect_to_polars(
            store.read(
                feature,
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )
        ).sort("sample_uid")
        assert result["sample_uid"].to_list() == [1, 4]


def test_postgresql_hard_delete_with_empty_filter_list_deletes_all(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """Hard delete with an empty filter list should delete all rows without errors."""
    feature = test_features["UpstreamFeatureA"]
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="w") as store:
        store.write(
            feature,
            pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"frames": "h1", "audio": "h1"},
                        {"frames": "h2", "audio": "h2"},
                    ],
                }
            ),
        )

        store.delete(
            feature,
            filters=[],
            soft=False,
            with_feature_history=True,
        )

        result = collect_to_polars(
            store.read(
                feature,
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )
        )
        assert result.is_empty()


@pytest.mark.parametrize(
    ("filters", "expected_remaining"),
    [
        ([(nw.col("sample_uid") == 1) | (nw.col("sample_uid") == 4)], [2, 3, 5]),
        ([nw.col("sample_uid").is_in([1, 3])], [2, 4, 5]),
        (
            [((nw.col("sample_uid") >= 2) & (nw.col("sample_uid") <= 4)) | (nw.col("sample_uid") == 1)],
            [5],
        ),
    ],
    ids=["or_predicate", "in_predicate", "nested_boolean_predicate"],
)
def test_postgresql_hard_delete_with_complex_predicates(
    clean_postgres_db: str,
    test_graph,
    test_features: dict,
    filters: list[nw.Expr],
    expected_remaining: list[int],
) -> None:
    """Hard delete should support complex predicate shapes compiled through SQL WHERE extraction."""
    feature = test_features["UpstreamFeatureA"]
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="w") as store:
        store.write(
            feature,
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3, 4, 5],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"frames": "h1", "audio": "h1"},
                        {"frames": "h2", "audio": "h2"},
                        {"frames": "h3", "audio": "h3"},
                        {"frames": "h4", "audio": "h4"},
                        {"frames": "h5", "audio": "h5"},
                    ],
                }
            ),
        )

        store.delete(
            feature,
            filters=filters,
            soft=False,
            with_feature_history=True,
        )

        result = collect_to_polars(
            store.read(
                feature,
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )
        ).sort("sample_uid")
        assert result["sample_uid"].to_list() == expected_remaining


@pytest.mark.parametrize(
    ("col_name", "col_dtype", "auto_cast", "expected"),
    [
        # Metaxy system columns: always converted when JSON or String
        (METAXY_PROVENANCE_BY_FIELD, dt.JSON(), True, True),
        (METAXY_PROVENANCE_BY_FIELD, dt.JSON(), False, True),
        (METAXY_PROVENANCE_BY_FIELD, dt.String(), True, True),
        (METAXY_PROVENANCE_BY_FIELD, dt.String(), False, True),
        (METAXY_PROVENANCE_BY_FIELD, dt.Int64(), True, False),
        (METAXY_DATA_VERSION_BY_FIELD, dt.JSON(), True, True),
        (METAXY_DATA_VERSION_BY_FIELD, dt.JSON(), False, True),
        (METAXY_DATA_VERSION_BY_FIELD, dt.String(), True, True),
        (METAXY_DATA_VERSION_BY_FIELD, dt.String(), False, True),
        (METAXY_DATA_VERSION_BY_FIELD, dt.Int64(), True, False),
        # User columns: only dt.JSON is considered when auto_cast=True
        ("user_col", dt.JSON(), True, True),
        ("user_col", dt.JSON(), False, False),
        ("user_col", dt.String(), True, False),
        ("user_col", dt.String(), False, False),
        ("user_col", dt.Int64(), True, False),
        ("user_col", dt.Float64(), False, False),
    ],
    ids=[
        "metaxy-provenance-json-auto_on",
        "metaxy-provenance-json-auto_off",
        "metaxy-provenance-string-auto_on",
        "metaxy-provenance-string-auto_off",
        "metaxy-provenance-int64-auto_on",
        "metaxy-data-version-json-auto_on",
        "metaxy-data-version-json-auto_off",
        "metaxy-data-version-string-auto_on",
        "metaxy-data-version-string-auto_off",
        "metaxy-data-version-int64-auto_on",
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


def test_postgresql_auto_create_struct_columns_use_jsonb(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """Auto-created Struct columns should use JSONB instead of TEXT."""
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=True).open(mode="w") as store:
        feature = test_features["UpstreamFeatureA"]
        table_name = store.get_table_name(feature.key)
        df = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
                "user_struct": pl.Series(
                    name="user_struct",
                    values=[
                        {"model": "resnet"},
                        {"model": "vgg"},
                    ],
                    dtype=pl.Struct(
                        [
                            pl.Field("model", pl.Utf8),
                        ]
                    ),
                ),
            }
        )
        store.write(feature, df)
        result = collect_to_polars(store.read(feature))
        result_sorted = result.sort("sample_uid")
        raw_types = store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"SELECT pg_typeof(user_struct)::text FROM {table_name} ORDER BY sample_uid"
        ).fetchall()

        assert isinstance(result_sorted.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
        assert result_sorted[METAXY_PROVENANCE_BY_FIELD].to_list() == [
            {"frames": "h1", "audio": "h2"},
            {"frames": "h3", "audio": "h4"},
        ]
        assert isinstance(result_sorted.schema["user_struct"], pl.Struct)
        assert result_sorted["user_struct"].to_list() == [{"model": "resnet"}, {"model": "vgg"}]
        assert raw_types == [("jsonb",), ("jsonb",)]


def test_postgresql_auto_cast_struct_for_jsonb_all_null_values(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """All-NULL user Struct values stored as JSONB remain nullable strings on read."""
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=True).open(mode="w") as store:
        feature = test_features["UpstreamFeatureA"]
        table_name = store.get_table_name(feature.key)
        df = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
                METAXY_DATA_VERSION_BY_FIELD: [
                    {"frames": "v1", "audio": "v2"},
                    {"frames": "v3", "audio": "v4"},
                ],
                "user_struct": pl.Series(
                    name="user_struct",
                    values=[None, None],
                    dtype=pl.Struct(
                        [
                            pl.Field("model", pl.Utf8),
                            pl.Field("version", pl.Utf8),
                        ]
                    ),
                ),
            }
        )
        store.write(feature, df)
        result = collect_to_polars(store.read(feature)).sort("sample_uid")
        raw_types = store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"SELECT pg_typeof(user_struct)::text FROM {table_name} ORDER BY sample_uid"
        ).fetchall()

        # Metaxy system columns should decode back to Struct with expected fields
        for system_column in (METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD):
            system_dtype = result.schema[system_column]
            assert isinstance(system_dtype, pl.Struct)
            assert {field.name for field in system_dtype.fields} == {"frames", "audio"}

        assert result.schema["user_struct"] == pl.Utf8
        assert result["user_struct"].to_list() == ["null", "null"]
        assert raw_types == [("jsonb",), ("jsonb",)]


@pytest.mark.parametrize("auto_cast", [True, False], ids=["auto_cast_on", "auto_cast_off"])
def test_postgresql_system_struct_columns_all_null_roundtrip(
    clean_postgres_db: str, test_graph, test_features: dict, auto_cast: bool
) -> None:
    """All-null system JSON payloads should roundtrip without forcing a Struct schema."""
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=auto_cast).open(mode="w") as store:
        null_struct = pl.Series(
            name=METAXY_PROVENANCE_BY_FIELD,
            values=[None, None],
            dtype=pl.Struct([pl.Field("frames", pl.Utf8), pl.Field("audio", pl.Utf8)]),
        )
        df = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: null_struct,
                METAXY_DATA_VERSION_BY_FIELD: null_struct.rename(METAXY_DATA_VERSION_BY_FIELD),
            }
        )

        store.write(test_features["UpstreamFeatureA"], df)
        result = collect_to_polars(store.read(test_features["UpstreamFeatureA"])).sort("sample_uid")

        for system_column in (METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD):
            assert result.schema[system_column] == pl.String
            assert result[system_column].to_list() == ["null", "null"]


def test_postgresql_auto_cast_struct_for_jsonb_user_jsonb_all_null_values(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """User JSONB columns with all SQL NULL values should roundtrip as nullable strings."""
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=True).open(mode="w") as store:
        feature = test_features["UpstreamFeatureA"]
        table_name = store.get_table_name(feature.key)

        df = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
                "user_struct": pl.Series(
                    name="user_struct",
                    values=[
                        {"model": "resnet", "version": "1"},
                        {"model": "vgg", "version": "2"},
                    ],
                    dtype=pl.Struct(
                        [
                            pl.Field("model", pl.Utf8),
                            pl.Field("version", pl.Utf8),
                        ]
                    ),
                ),
            }
        )
        store.write(feature, df)
        raw_types = store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"SELECT pg_typeof(user_struct)::text FROM {table_name} ORDER BY sample_uid"
        ).fetchall()
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"UPDATE {table_name} SET user_struct = NULL"
        )

        result = collect_to_polars(store.read(feature)).sort("sample_uid")

        assert isinstance(result.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
        assert result.schema["user_struct"] == pl.Utf8
        assert result["user_struct"].to_list() == [None, None]
        assert raw_types == [("jsonb",), ("jsonb",)]


def test_postgresql_auto_cast_struct_for_jsonb_false_all_null_values(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """With auto_cast=False, all-NULL user JSON strings stay strings."""
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=False).open(mode="w") as store:
        df = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
                METAXY_DATA_VERSION_BY_FIELD: [
                    {"frames": "v1", "audio": "v2"},
                    {"frames": "v3", "audio": "v4"},
                ],
                # auto_cast=False requires user data to be pre-encoded (or already string)
                "user_json": pl.Series(name="user_json", values=[None, None], dtype=pl.Utf8),
            }
        )
        store.write(test_features["UpstreamFeatureA"], df)
        result = collect_to_polars(store.read(test_features["UpstreamFeatureA"])).sort("sample_uid")

        for system_column in (METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD):
            system_dtype = result.schema[system_column]
            assert isinstance(system_dtype, pl.Struct)
            assert {field.name for field in system_dtype.fields} == {"frames", "audio"}

        assert result.schema["user_json"] == pl.Utf8
        assert result["user_json"].to_list() == [None, None]


def test_postgresql_parse_json_user_column_all_null_values(postgresql_db: str, test_graph, test_features: dict) -> None:
    """All-NULL user JSON columns infer pl.Null when input is created without dtype hints."""
    with PostgreSQLMetadataStore(postgresql_db, auto_cast_struct_for_jsonb=True).open(mode="w") as store:
        feature_key = test_features["UpstreamFeatureA"].key
        pl_df = pl.DataFrame({"sample_uid": [1, 2], "user_json": [None, None]})

        result = store._parse_json_to_struct_columns(pl_df, feature_key, ["user_json"])

        assert result.schema["user_json"] == pl.Null
        assert result["user_json"].null_count() == len(result)


def test_postgresql_parse_json_user_utf8_column_all_null_values(test_graph, test_features: dict) -> None:
    """All-NULL nullable UTF8 user columns should remain UTF8 (not Struct) when parsing candidates."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    feature_key = test_features["UpstreamFeatureA"].key
    pl_df = pl.DataFrame(
        {
            "sample_uid": [1, 2],
            "user_text": pl.Series(name="user_text", values=[None, None], dtype=pl.Utf8),
        }
    )

    result = store._parse_json_to_struct_columns(pl_df, feature_key, ["user_text"])

    assert result.schema["user_text"] == pl.Utf8
    assert result["user_text"].to_list() == [None, None]


def test_postgresql_parse_json_user_utf8_literal_null_values(test_graph, test_features: dict) -> None:
    """Literal \"null\" values in UTF8 user columns should remain UTF8 strings."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    feature_key = test_features["UpstreamFeatureA"].key
    pl_df = pl.DataFrame(
        {
            "sample_uid": [1, 2],
            "user_text": pl.Series(name="user_text", values=["null", None], dtype=pl.Utf8),
        }
    )

    result = store._parse_json_to_struct_columns(pl_df, feature_key, ["user_text"])

    assert result.schema["user_text"] == pl.Utf8
    assert result["user_text"].to_list() == ["null", None]


def test_postgresql_plain_nullable_text_columns_remain_utf8_after_read(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """Nullable plain TEXT columns should not be auto-converted to Struct on read."""
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=True).open(mode="w") as store:
        df = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
                METAXY_DATA_VERSION_BY_FIELD: [
                    {"frames": "v1", "audio": "v2"},
                    {"frames": "v3", "audio": "v4"},
                ],
                "plain_text_all_null": pl.Series(
                    name="plain_text_all_null",
                    values=[None, None],
                    dtype=pl.Utf8,
                ),
                "plain_text_with_literal_null": pl.Series(
                    name="plain_text_with_literal_null",
                    values=["null", None],
                    dtype=pl.Utf8,
                ),
                "plain_text_with_object_json": pl.Series(
                    name="plain_text_with_object_json",
                    values=['{"looks":"json"}', None],
                    dtype=pl.Utf8,
                ),
            }
        )
        store.write(test_features["UpstreamFeatureA"], df)
        result = collect_to_polars(store.read(test_features["UpstreamFeatureA"])).sort("sample_uid")

        assert result.schema["plain_text_all_null"] == pl.Utf8
        assert result.schema["plain_text_with_literal_null"] == pl.Utf8
        assert result.schema["plain_text_with_object_json"] == pl.Utf8
        assert result["plain_text_all_null"].to_list() == [None, None]
        assert result["plain_text_with_literal_null"].to_list() == ["null", None]
        assert result["plain_text_with_object_json"].to_list() == ['{"looks":"json"}', None]


def test_parse_json_to_struct_columns_missing_feature_in_graph_decodes_system_structs(test_graph) -> None:
    """Missing feature definitions should still decode Metaxy system columns."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    original_value = '{"frames":"h1","audio":"h2"}'
    df = pl.DataFrame(
        {
            METAXY_PROVENANCE_BY_FIELD: [
                original_value,
            ]
        }
    )
    missing_feature_key = FeatureKey(["test_stores", "missing_feature"])

    result = store._parse_json_to_struct_columns(
        df,
        missing_feature_key,
        [METAXY_PROVENANCE_BY_FIELD],
    )
    assert isinstance(result.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
    assert result[METAXY_PROVENANCE_BY_FIELD].to_list() == [{"frames": "h1", "audio": "h2"}]


def test_postgresql_validate_required_system_columns_without_feature_schema(test_graph) -> None:
    """Required system columns should validate after inference-based decode despite key-order differences."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
    df = pl.DataFrame(
        {
            METAXY_PROVENANCE_BY_FIELD: ['{"frames":"h1","audio":"h2"}'],
            METAXY_DATA_VERSION_BY_FIELD: ['{"audio":"v2","frames":"v1"}'],
        }
    )
    parsed = store._parse_json_to_struct_columns(
        df,
        missing_feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )
    store._validate_required_system_struct_columns(
        parsed,
        missing_feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )
    assert isinstance(parsed.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
    assert isinstance(parsed.schema[METAXY_DATA_VERSION_BY_FIELD], pl.Struct)


def test_postgresql_read_missing_feature_schema_decodes_system_columns(clean_postgres_db: str, test_graph) -> None:
    """Public read() should decode required system columns even when JSON key order differs."""
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="r") as store:
        missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
        table_name = store.get_table_name(missing_feature_key)

        store.conn.raw_sql(f"DROP TABLE IF EXISTS {table_name}")  # ty: ignore[unresolved-attribute]
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                {METAXY_PROVENANCE_BY_FIELD} TEXT,
                {METAXY_DATA_VERSION_BY_FIELD} TEXT
            )
            """
        )
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            INSERT INTO {table_name} (sample_uid, {METAXY_PROVENANCE_BY_FIELD}, {METAXY_DATA_VERSION_BY_FIELD})
            VALUES
                (1, '{{"frames":"h1","audio":"h2"}}', '{{"audio":"v2","frames":"v1"}}'),
                (2, '{{"audio":"h4","frames":"h3"}}', '{{"frames":"v3","audio":"v4"}}')
            """
        )

        result = collect_to_polars(
            store.read(
                missing_feature_key,
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )
        ).sort("sample_uid")

        assert isinstance(result.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
        assert isinstance(result.schema[METAXY_DATA_VERSION_BY_FIELD], pl.Struct)
        assert result[METAXY_PROVENANCE_BY_FIELD].to_list() == [
            {"frames": "h1", "audio": "h2"},
            {"frames": "h3", "audio": "h4"},
        ]
        assert result[METAXY_DATA_VERSION_BY_FIELD].to_list() == [
            {"frames": "v1", "audio": "v2"},
            {"frames": "v3", "audio": "v4"},
        ]


def test_postgresql_read_missing_feature_schema_fails_fast_on_invalid_system_json(
    clean_postgres_db: str, test_graph
) -> None:
    """Public read() should fail early when required system columns are not decoded to Struct."""
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="r") as store:
        missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
        table_name = store.get_table_name(missing_feature_key)

        store.conn.raw_sql(f"DROP TABLE IF EXISTS {table_name}")  # ty: ignore[unresolved-attribute]
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                {METAXY_PROVENANCE_BY_FIELD} TEXT,
                {METAXY_DATA_VERSION_BY_FIELD} TEXT
            )
            """
        )
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            INSERT INTO {table_name} (sample_uid, {METAXY_PROVENANCE_BY_FIELD}, {METAXY_DATA_VERSION_BY_FIELD})
            VALUES
                (1, '[1,2,3]', '{{"frames":"v1","audio":"v2"}}')
            """
        )

        with pytest.raises(ValueError, match="Failed to decode or validate required Metaxy system JSON columns"):
            store.read(
                missing_feature_key,
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )


def test_postgresql_read_missing_feature_schema_allows_all_null_system_json(clean_postgres_db: str, test_graph) -> None:
    """All-NULL required system columns should be treated as no payload and not fail."""
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="r") as store:
        missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
        table_name = store.get_table_name(missing_feature_key)

        store.conn.raw_sql(f"DROP TABLE IF EXISTS {table_name}")  # ty: ignore[unresolved-attribute]
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                {METAXY_PROVENANCE_BY_FIELD} TEXT,
                {METAXY_DATA_VERSION_BY_FIELD} TEXT
            )
            """
        )
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            INSERT INTO {table_name} (sample_uid, {METAXY_PROVENANCE_BY_FIELD}, {METAXY_DATA_VERSION_BY_FIELD})
            VALUES
                (1, NULL, NULL),
                (2, NULL, NULL)
            """
        )

        result = collect_to_polars(
            store.read(
                missing_feature_key,
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )
        ).sort("sample_uid")

        assert result["sample_uid"].to_list() == [1, 2]
        assert result[METAXY_PROVENANCE_BY_FIELD].to_list() == [None, None]
        assert result[METAXY_DATA_VERSION_BY_FIELD].to_list() == [None, None]


@pytest.mark.parametrize("auto_cast", [True, False], ids=["auto_cast_on", "auto_cast_off"])
def test_postgresql_read_missing_feature_schema_all_null_system_json_auto_cast_modes(
    clean_postgres_db: str, test_graph, auto_cast: bool
) -> None:
    """All-NULL required system columns should not fail regardless of auto_cast mode."""
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=auto_cast).open(mode="r") as store:
        missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
        table_name = store.get_table_name(missing_feature_key)

        store.conn.raw_sql(f"DROP TABLE IF EXISTS {table_name}")  # ty: ignore[unresolved-attribute]
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                {METAXY_PROVENANCE_BY_FIELD} TEXT,
                {METAXY_DATA_VERSION_BY_FIELD} TEXT
            )
            """
        )
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            INSERT INTO {table_name} (sample_uid, {METAXY_PROVENANCE_BY_FIELD}, {METAXY_DATA_VERSION_BY_FIELD})
            VALUES
                (1, NULL, NULL),
                (2, NULL, NULL)
            """
        )

        result = collect_to_polars(
            store.read(
                missing_feature_key,
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )
        ).sort("sample_uid")
        assert result["sample_uid"].to_list() == [1, 2]
        assert result[METAXY_PROVENANCE_BY_FIELD].to_list() == [None, None]
        assert result[METAXY_DATA_VERSION_BY_FIELD].to_list() == [None, None]


def test_postgresql_validate_required_system_columns_allows_all_null_without_feature_schema(test_graph) -> None:
    """Validation should allow all-NULL required system columns without feature schema context."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
    parsed = pl.DataFrame(
        {
            METAXY_PROVENANCE_BY_FIELD: pl.Series(
                name=METAXY_PROVENANCE_BY_FIELD,
                values=[None, None],
                dtype=pl.Utf8,
            ),
            METAXY_DATA_VERSION_BY_FIELD: pl.Series(
                name=METAXY_DATA_VERSION_BY_FIELD,
                values=[None, None],
                dtype=pl.Utf8,
            ),
        }
    )

    store._validate_required_system_struct_columns(
        parsed,
        missing_feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )


def test_postgresql_known_feature_empty_result_preserves_system_struct_schema(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """Empty reads for known features should preserve Metaxy struct schemas."""
    feature = test_features["UpstreamFeatureA"]

    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="w") as store:
        store.write(
            feature,
            pl.DataFrame(
                {
                    "sample_uid": [1],
                    METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h2"}],
                }
            ),
        )

        result = collect_to_polars(
            store.read(
                feature,
                filters=[nw.col("sample_uid") == 999],
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )
        )

        assert result.is_empty()
        assert isinstance(result.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
        assert isinstance(result.schema[METAXY_DATA_VERSION_BY_FIELD], pl.Struct)


def test_postgresql_read_missing_feature_schema_empty_result_returns_empty(clean_postgres_db: str, test_graph) -> None:
    """Missing-feature reads with zero rows should return empty results, not validation errors."""
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="r") as store:
        missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
        table_name = store.get_table_name(missing_feature_key)

        store.conn.raw_sql(f"DROP TABLE IF EXISTS {table_name}")  # ty: ignore[unresolved-attribute]
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                {METAXY_PROVENANCE_BY_FIELD} TEXT,
                {METAXY_DATA_VERSION_BY_FIELD} TEXT
            )
            """
        )

        result = collect_to_polars(
            store.read(
                missing_feature_key,
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )
        )
        assert result.is_empty()


def test_postgresql_read_missing_feature_schema_filter_to_zero_rows_returns_empty(
    clean_postgres_db: str, test_graph
) -> None:
    """Missing-feature reads filtered to zero rows should return empty results, not validation errors."""
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="r") as store:
        missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
        table_name = store.get_table_name(missing_feature_key)

        store.conn.raw_sql(f"DROP TABLE IF EXISTS {table_name}")  # ty: ignore[unresolved-attribute]
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                {METAXY_PROVENANCE_BY_FIELD} TEXT,
                {METAXY_DATA_VERSION_BY_FIELD} TEXT
            )
            """
        )
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            INSERT INTO {table_name} (sample_uid, {METAXY_PROVENANCE_BY_FIELD}, {METAXY_DATA_VERSION_BY_FIELD})
            VALUES
                (1, '{{"frames":"h1","audio":"h2"}}', '{{"frames":"v1","audio":"v2"}}')
            """
        )

        result = collect_to_polars(
            store.read(
                missing_feature_key,
                filters=[nw.col("sample_uid") == 999],
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )
        )
        assert result.is_empty()


def test_postgresql_read_missing_required_system_columns_raises_schema_error(
    clean_postgres_db: str, test_graph
) -> None:
    """Missing required Metaxy system columns should raise a clear schema error on full reads."""
    with PostgreSQLMetadataStore(clean_postgres_db).open(mode="r") as store:
        missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
        table_name = store.get_table_name(missing_feature_key)

        store.conn.raw_sql(f"DROP TABLE IF EXISTS {table_name}")  # ty: ignore[unresolved-attribute]
        store.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"""
            CREATE TABLE {table_name} (
                sample_uid BIGINT,
                {METAXY_PROVENANCE_BY_FIELD} TEXT
            )
            """
        )

        with pytest.raises(ValueError, match="Required system columns are missing from the result set"):
            store.read(
                missing_feature_key,
                with_feature_history=True,
                with_sample_history=True,
                include_soft_deleted=True,
            )


def test_parse_json_system_columns_all_null_remain_null_without_forced_schema(test_graph, test_features: dict) -> None:
    """All-NULL required system columns should not force a Struct schema."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    feature_key = test_features["UpstreamFeatureA"].key
    df = pl.DataFrame(
        {
            METAXY_PROVENANCE_BY_FIELD: [None, None],
            METAXY_DATA_VERSION_BY_FIELD: [None, None],
        }
    )

    parsed = store._parse_json_to_struct_columns(
        df,
        feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )
    store._validate_required_system_struct_columns(
        parsed,
        feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )

    assert parsed.schema[METAXY_PROVENANCE_BY_FIELD] in (pl.Null, pl.String)
    assert parsed.schema[METAXY_DATA_VERSION_BY_FIELD] in (pl.Null, pl.String)
    assert parsed[METAXY_PROVENANCE_BY_FIELD].to_list() == [None, None]
    assert parsed[METAXY_DATA_VERSION_BY_FIELD].to_list() == [None, None]


def test_validate_required_system_columns_allow_inferred_sparse_fields(test_graph, test_features: dict) -> None:
    """Required system columns should rely on inferred Struct fields without backfilling missing keys."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    feature_key = test_features["UpstreamFeatureA"].key
    df = pl.DataFrame(
        {
            METAXY_PROVENANCE_BY_FIELD: ['{"frames":"h1"}'],
            METAXY_DATA_VERSION_BY_FIELD: ['{"frames":"v1"}'],
        }
    )
    parsed = store._parse_json_to_struct_columns(
        df,
        feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )
    store._validate_required_system_struct_columns(
        parsed,
        feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )

    assert isinstance(parsed.schema[METAXY_PROVENANCE_BY_FIELD], pl.Struct)
    assert isinstance(parsed.schema[METAXY_DATA_VERSION_BY_FIELD], pl.Struct)
    assert parsed[METAXY_PROVENANCE_BY_FIELD].to_list() == [{"frames": "h1"}]
    assert parsed[METAXY_DATA_VERSION_BY_FIELD].to_list() == [{"frames": "v1"}]


def test_validate_required_system_columns_ignores_field_order_with_feature_schema(
    test_graph, test_features: dict
) -> None:
    """Field-name order should not matter for inferred Struct validation."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    feature_key = test_features["UpstreamFeatureA"].key
    parsed = pl.DataFrame(
        {
            METAXY_PROVENANCE_BY_FIELD: [{"audio": "h2", "frames": "h1"}],
            METAXY_DATA_VERSION_BY_FIELD: [{"frames": "v1", "audio": "v2"}],
        }
    )

    store._validate_required_system_struct_columns(
        parsed,
        feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )


def test_validate_required_system_columns_allows_extra_fields_with_feature_schema(
    test_graph, test_features: dict
) -> None:
    """Extra inferred fields should remain valid when system JSON decodes to Struct."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    feature_key = test_features["UpstreamFeatureA"].key
    parsed = pl.DataFrame(
        {
            METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h2", "extra": "x"}],
            METAXY_DATA_VERSION_BY_FIELD: [{"frames": "v1", "audio": "v2", "extra": "x"}],
        }
    )

    store._validate_required_system_struct_columns(
        parsed,
        feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )


def test_validate_required_system_columns_without_feature_schema_allows_different_field_sets(test_graph) -> None:
    """Required system columns should not need matching inferred field sets."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
    parsed = pl.DataFrame(
        {
            METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h2"}],
            METAXY_DATA_VERSION_BY_FIELD: [{"frames": "v1"}],
        }
    )

    store._validate_required_system_struct_columns(
        parsed,
        missing_feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )


def test_validate_required_system_columns_without_feature_schema_ignores_field_order(test_graph) -> None:
    """Without feature schema lookup, decoded Struct validation ignores key order."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
    parsed = pl.DataFrame(
        {
            METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h2"}],
            METAXY_DATA_VERSION_BY_FIELD: [{"audio": "v2", "frames": "v1"}],
        }
    )

    store._validate_required_system_struct_columns(
        parsed,
        missing_feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )


def test_validate_required_system_columns_without_feature_schema_allows_consistent_extra_fields(test_graph) -> None:
    """Without feature schema lookup, extra inferred fields should remain valid."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    missing_feature_key = FeatureKey(["test_stores", "missing_feature"])
    parsed = pl.DataFrame(
        {
            METAXY_PROVENANCE_BY_FIELD: [{"frames": "h1", "audio": "h2", "extra": "x"}],
            METAXY_DATA_VERSION_BY_FIELD: [{"audio": "v2", "extra": "x", "frames": "v1"}],
        }
    )

    store._validate_required_system_struct_columns(
        parsed,
        missing_feature_key,
        [METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD],
    )


def test_parse_json_user_columns_do_not_require_feature_in_graph(test_graph) -> None:
    """User JSON decode should not require feature definition lookup."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    df = pl.DataFrame({"user_json": ['{"model":"resnet","version":"1"}']})
    missing_feature_key = FeatureKey(["test_stores", "missing_feature"])

    result = store._parse_json_to_struct_columns(df, missing_feature_key, ["user_json"])

    assert isinstance(result.schema["user_json"], pl.Struct)
    assert result["user_json"].to_list() == [{"model": "resnet", "version": "1"}]


def test_parse_json_user_columns_infers_from_full_column(test_graph) -> None:
    """Schema inference for user JSON should include sparse keys from all rows."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    df = pl.DataFrame({"user_json": ['{"a":1}', '{"a":2,"b":"x"}']})
    missing_feature_key = FeatureKey(["test_stores", "missing_feature"])

    result = store._parse_json_to_struct_columns(df, missing_feature_key, ["user_json"])

    assert isinstance(result.schema["user_json"], pl.Struct)
    assert result["user_json"].to_list() == [{"a": 1, "b": None}, {"a": 2, "b": "x"}]


def test_parse_json_user_columns_with_incompatible_shapes_fallback_to_strings(test_graph) -> None:
    """Incompatible user JSON rows should remain strings instead of failing decode."""
    store = PostgreSQLMetadataStore(connection_params={"host": "localhost", "database": "dummy"})
    original_values = ['{"a":1}', '{"a":{"nested":2}}']
    df = pl.DataFrame({"user_json": original_values})
    missing_feature_key = FeatureKey(["test_stores", "missing_feature"])

    result = store._parse_json_to_struct_columns(df, missing_feature_key, ["user_json"])

    assert result.schema["user_json"] == pl.Utf8
    assert result["user_json"].to_list() == original_values


def test_postgresql_user_jsonb_infers_schema_from_all_rows(
    clean_postgres_db: str, test_graph, test_features: dict
) -> None:
    """User JSONB decode should infer schema across all rows without dropping sparse fields."""
    with PostgreSQLMetadataStore(clean_postgres_db, auto_cast_struct_for_jsonb=True).open(mode="w") as store:
        feature = test_features["UpstreamFeatureA"]

        df = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
                "user_json": pl.Series(
                    name="user_json",
                    values=[
                        {"a": 1},
                        {"a": 2, "b": "x"},
                    ],
                    dtype=pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Utf8)]),
                ),
            }
        )
        store.write(feature, df)

        result = collect_to_polars(store.read(feature)).sort("sample_uid")
        assert isinstance(result.schema["user_json"], pl.Struct)
        assert result["user_json"].to_list() == [{"a": 1, "b": None}, {"a": 2, "b": "x"}]
