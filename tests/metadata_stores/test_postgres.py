"""PostgreSQL-specific tests that don't apply to other stores."""

from typing import Any, cast

import ibis
import ibis.backends.postgres
import pytest

from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.postgres import PostgresMetadataStore
from metaxy.models.feature import BaseFeature
from metaxy.versioning.types import HashAlgorithm


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
    """Test SHA256 hash function via UDF approach."""
    store = PostgresMetadataStore(host="localhost", database="metaxy")

    # Hash functions are created via UDF approach (like DuckDB/ClickHouse)
    hash_functions = store._create_hash_functions()
    assert HashAlgorithm.SHA256 in hash_functions
    assert HashAlgorithm.MD5 in hash_functions

    # Verify the hash functions are callable
    sha256_fn = hash_functions[HashAlgorithm.SHA256]
    md5_fn = hash_functions[HashAlgorithm.MD5]
    assert callable(sha256_fn)
    assert callable(md5_fn)


def test_postgres_display_with_connection_string() -> None:
    """Test display output when initialized with connection string."""
    store = PostgresMetadataStore(
        "postgresql://user:pass@localhost:5432/metaxy",
    )

    display = store.display()
    assert "connection_string=postgresql://user:pass@localhost:5432/metaxy" in display


def test_postgres_schema_detection_decodes_bytes() -> None:
    """Ensure raw psycopg cursors returning bytes still produce schema names."""
    store = PostgresMetadataStore(host="localhost", database="metaxy")

    class DummyCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, query: str) -> None:
            assert query == "SHOW search_path"

        def fetchone(self) -> tuple[bytes]:
            return (b'"$user", "analytics"',)

    class DummyConn:
        def cursor(self):
            return DummyCursor()

    assert store._get_current_schema(DummyConn()) == "analytics"


def test_postgres_table_listing_decodes_bytes() -> None:
    """Ensure robust table listing handles byte outputs without the global shim."""
    store = PostgresMetadataStore(host="localhost", database="metaxy")
    store.schema = "public"

    class DummyCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, query: str, params: tuple[str, ...]) -> None:
            assert "pg_tables" in query
            assert params == ("public",)

        def fetchall(self) -> list[tuple[bytes]]:
            return [(b"foo",), (b"bar",)]

    class DummyRawConn:
        def cursor(self):
            return DummyCursor()

    class DummyBackend:
        def __init__(self) -> None:
            self.con = DummyRawConn()

        def list_tables(self) -> list[str]:
            return ["fallback"]

    store._conn = cast(Any, DummyBackend())

    assert store._list_tables_robustly() == ["foo", "bar"]


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
    # Should have SHA256 hash function via UDF approach
    hash_functions = store._create_hash_functions()
    assert HashAlgorithm.SHA256 in hash_functions
    # Should also have MD5
    assert HashAlgorithm.MD5 in hash_functions


def test_postgres_pgcrypto_enabled_during_native_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure pgcrypto extension is enabled lazily when native resolve runs."""
    store = PostgresMetadataStore(
        host="localhost",
        database="metaxy",
        enable_pgcrypto=True,
        hash_algorithm=HashAlgorithm.SHA256,
    )
    dummy_backend = cast(ibis.BaseBackend, object())
    store._conn = dummy_backend  # Pretend connection is open

    enable_calls: list[bool] = []

    def fake_enable(self: PostgresMetadataStore) -> None:
        enable_calls.append(True)

    monkeypatch.setattr(
        PostgresMetadataStore,
        "_ensure_pgcrypto_extension",
        fake_enable,
    )

    def fake_resolve(
        self: MetadataStore,
        feature: type[BaseFeature],
        *,
        filters=None,
        lazy=False,
    ) -> str:
        return "ok"

    monkeypatch.setattr(MetadataStore, "_resolve_update_native", fake_resolve)

    dummy_feature = cast(type[BaseFeature], object())
    first = store._resolve_update_native(dummy_feature)
    second = store._resolve_update_native(dummy_feature)

    assert first == "ok"
    assert second == "ok"
    assert enable_calls == [True]


def test_postgres_pgcrypto_not_enabled_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure pgcrypto is skipped when enable_pgcrypto flag is False."""

    store = PostgresMetadataStore(
        host="localhost",
        database="metaxy",
        enable_pgcrypto=False,
        hash_algorithm=HashAlgorithm.SHA256,
    )
    dummy_backend = cast(ibis.BaseBackend, object())
    store._conn = dummy_backend

    def fake_enable(
        self: PostgresMetadataStore,
    ) -> None:  # pragma: no cover - should not run
        raise AssertionError("pgcrypto should not be enabled when flag is False")

    monkeypatch.setattr(
        PostgresMetadataStore,
        "_ensure_pgcrypto_extension",
        fake_enable,
    )

    def fake_resolve(
        self: MetadataStore,
        feature: type[BaseFeature],
        *,
        filters=None,
        lazy=False,
    ) -> str:
        return "ok"

    monkeypatch.setattr(MetadataStore, "_resolve_update_native", fake_resolve)

    dummy_feature = cast(type[BaseFeature], object())
    result = store._resolve_update_native(dummy_feature)

    assert result == "ok"
