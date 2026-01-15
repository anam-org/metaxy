"""PostgreSQL-specific tests that don't apply to other stores."""

from __future__ import annotations

from typing import Any, cast

import ibis
import narwhals as nw
import polars as pl
import polars.testing as pl_testing
import pytest

from metaxy._testing import add_metaxy_provenance_column
from metaxy._testing.models import SampleFeatureSpec
from metaxy._utils import switch_implementation_to_polars
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.postgres import PostgresMetadataStore
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.feature import BaseFeature
from metaxy.models.types import FeatureKey
from metaxy.versioning.types import HashAlgorithm

pytest.importorskip("ibis")

try:
    import ibis.backends.postgres  # noqa: F401
except ImportError:
    pytest.skip("ibis-postgres not installed", allow_module_level=True)


def test_postgres_initialization_with_params(postgres_server: dict[str, Any]) -> None:
    """Test initialization with explicit connection parameters."""
    dbname = postgres_server["dbname"]
    store_host = postgres_server["host"]
    store_port = postgres_server["port"]
    store_user = postgres_server["user"]
    store_password = postgres_server["password"]

    store = PostgresMetadataStore(
        host=store_host,
        port=store_port,
        database=dbname,
        user=store_user,
        password=store_password,
    )

    assert store.host == store_host
    assert store.port == store_port
    assert store.database == dbname
    assert store.schema is None

    display = store.display()
    assert "PostgresMetadataStore" in display
    assert f"database={dbname}" in display


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


def test_postgres_hash_functions_include_sha256() -> None:
    """Test SHA256 hash function is available."""
    store = PostgresMetadataStore(host="localhost", database="metaxy")

    hash_functions = store._create_hash_functions()
    assert HashAlgorithm.SHA256 in hash_functions
    assert HashAlgorithm.MD5 in hash_functions

    assert callable(hash_functions[HashAlgorithm.SHA256])
    assert callable(hash_functions[HashAlgorithm.MD5])


def test_postgres_display_with_connection_string() -> None:
    """Test display output when initialized with connection string."""
    connection_string = (
        "postgresql://user:pass@localhost:5432/metaxy?password=secret&sslmode=require"
    )
    store = PostgresMetadataStore(connection_string)

    display = store.display()
    assert (
        display == "PostgresMetadataStore("
        "connection_string=postgresql://user:***@localhost:5432/metaxy"
        "?password=***&sslmode=require)"
    )


def test_postgres_default_hash_algorithm_is_md5() -> None:
    """Test that default hash algorithm is MD5 (no extension required)."""
    store = PostgresMetadataStore(host="localhost", database="metaxy")

    assert store._get_default_hash_algorithm() == HashAlgorithm.MD5
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
    hash_functions = store._create_hash_functions()
    assert HashAlgorithm.SHA256 in hash_functions
    assert HashAlgorithm.MD5 in hash_functions


def test_postgres_jsonb_pack_unpack_round_trip(postgres_db: str, graph) -> None:
    """Write flattened provenance, ensure JSONB storage and unpacking on read."""

    class SimpleFeature(
        BaseFeature,
        spec=SampleFeatureSpec(key=FeatureKey(["pg", "jsonb_test"]), fields=["value"]),
    ):
        pass

    feature_key = SimpleFeature.spec().key

    df = pl.DataFrame(
        {
            "sample_uid": [1, 2],
            "value": ["a", "b"],
            "metaxy_provenance_by_field__value": ["p1", "p2"],
            "metaxy_data_version_by_field__value": ["p1", "p2"],
        }
    )

    with (
        graph.use(),
        PostgresMetadataStore(
            connection_string=postgres_db,
            hash_algorithm=HashAlgorithm.MD5,
        ) as store,
    ):
        raw_conn = cast(Any, store.conn).con
        table_name = store.get_table_name(feature_key)
        with raw_conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE "{table_name}" (
                    sample_uid BIGINT,
                    value TEXT,
                    {METAXY_PROVENANCE_BY_FIELD} JSONB,
                    {METAXY_DATA_VERSION_BY_FIELD} JSONB
                )
                """
            )
        raw_conn.commit()

        store.write_metadata_to_store(feature_key, nw.from_native(df))

        schema = store.schema or "public"
        with raw_conn.cursor() as cur:
            cur.execute(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s AND column_name = %s
                """,
                (schema, table_name, METAXY_PROVENANCE_BY_FIELD),
            )
            provenance_type = cur.fetchone()[0]

        assert provenance_type.lower() == "jsonb"

        lazy = store.read_metadata_in_store(feature_key)
        assert lazy is not None
        result_pl = switch_implementation_to_polars(lazy).collect().to_native()
        assert "metaxy_provenance_by_field__value" in result_pl.columns
        assert "metaxy_data_version_by_field__value" in result_pl.columns
        if METAXY_PROVENANCE_BY_FIELD in result_pl.columns:
            assert result_pl.schema[METAXY_PROVENANCE_BY_FIELD] == pl.Struct
        if METAXY_DATA_VERSION_BY_FIELD in result_pl.columns:
            assert result_pl.schema[METAXY_DATA_VERSION_BY_FIELD] == pl.Struct
        assert result_pl["metaxy_provenance_by_field__value"].to_list() == ["p1", "p2"]
        assert result_pl["metaxy_data_version_by_field__value"].to_list() == [
            "p1",
            "p2",
        ]


def test_postgres_resolve_update_matches_duckdb(
    postgres_db: str, tmp_path, graph
) -> None:
    """PostgreSQL resolve_update should match DuckDB provenance hashes."""

    class RootFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["pg", "resolve_update_root"]),
            fields=["value"],
        ),
    ):
        pass

    samples = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "value": [10, 20, 30],
            METAXY_PROVENANCE_BY_FIELD: [
                {"value": "h1"},
                {"value": "h2"},
                {"value": "h3"},
            ],
        }
    )
    samples = add_metaxy_provenance_column(
        samples, RootFeature, hash_algorithm=HashAlgorithm.MD5
    )
    ibis_samples = ibis.memtable(samples.to_dicts())

    duck_store = DuckDBMetadataStore(
        database=tmp_path / "duckdb_pg_parity.duckdb",
        auto_create_tables=True,
        hash_algorithm=HashAlgorithm.MD5,
    )
    pg_store = PostgresMetadataStore(
        connection_string=postgres_db,
        auto_create_tables=True,
        hash_algorithm=HashAlgorithm.MD5,
    )

    with graph.use():
        with duck_store:
            duck_inc = duck_store.resolve_update(
                RootFeature,
                samples=nw.from_native(ibis_samples, eager_only=False),
            )
            duck_added = duck_inc.added.lazy().collect().to_polars().sort("sample_uid")

        with pg_store:
            pg_inc = pg_store.resolve_update(
                RootFeature,
                samples=nw.from_native(ibis_samples, eager_only=False),
            )
            pg_added = pg_inc.added.lazy().collect().to_polars().sort("sample_uid")

    duck_field_hashes = duck_added.select(
        pl.col(METAXY_PROVENANCE_BY_FIELD).struct.field("value").alias("field_hash"),
        "metaxy_provenance",
    )
    if "metaxy_provenance_by_field__value" in pg_added.columns:
        pg_field_hashes = pg_added.select(
            pl.col("metaxy_provenance_by_field__value").alias("field_hash"),
            "metaxy_provenance",
        )
    else:
        pg_field_hashes = pg_added.select(
            pl.col(METAXY_PROVENANCE_BY_FIELD)
            .struct.field("value")
            .alias("field_hash"),
            "metaxy_provenance",
        )

    pl_testing.assert_frame_equal(
        duck_field_hashes, pg_field_hashes, check_row_order=True
    )
