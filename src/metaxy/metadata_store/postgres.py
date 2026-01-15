"""PostgreSQL metadata store using JSONB-backed provenance columns.

Implements IbisJsonCompatStore for PostgreSQL, packing struct-like columns into
JSONB while keeping queries lazy in Ibis.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import ibis
import ibis.expr.datatypes as dt
import narwhals as nw
import polars as pl
import pyarrow as pa
from narwhals.typing import Frame
from psycopg import sql
from psycopg.types.json import Jsonb
from pydantic import ConfigDict, Field
from typing_extensions import Self

from metaxy.metadata_store.exceptions import (
    HashAlgorithmNotSupportedError,
    TableNotFoundError,
)
from metaxy.metadata_store.ibis import IbisMetadataStoreConfig
from metaxy.metadata_store.ibis_json_compat import IbisJsonCompatStore
from metaxy.metadata_store.types import AccessMode
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.types import CoercibleToFeatureKey
from metaxy.versioning.postgres import PostgresVersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

logger = logging.getLogger(__name__)


class PostgresMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for PostgresMetadataStore.

    Inherits connection_string, connection_params, table_prefix, auto_create_tables from
    IbisMetadataStoreConfig.

    Example:
        ```python
        config = PostgresMetadataStoreConfig(
            connection_string="postgresql://user:pass@host:5432/db",
            schema="public",
            enable_pgcrypto=False,
        )
        ```
    """

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    host: str | None = Field(default=None, description="PostgreSQL server host.")
    port: int | None = Field(default=None, description="PostgreSQL server port.")
    user: str | None = Field(default=None, description="PostgreSQL username.")
    password: str | None = Field(default=None, description="PostgreSQL password.")
    database: str | None = Field(default=None, description="PostgreSQL database name.")
    db_schema: str | None = Field(
        default=None,
        alias="schema",
        description="PostgreSQL schema (defaults to search_path).",
    )
    enable_pgcrypto: bool = Field(
        default=False,
        description=(
            "If True, attempt to enable pgcrypto for SHA256 hashing on open. "
            "pgcrypto is required for SHA256 hashing on PostgreSQL."
        ),
    )


class PostgresMetadataStore(IbisJsonCompatStore):
    """
    [PostgreSQL](https://www.postgresql.org/) metadata store using JSONB storage.

    Provides production-grade metadata storage using PostgreSQL with:
    - Full ACID compliance
    - JSONB storage for struct-like columns via Ibis JSON packing
    - Extension support (pgcrypto for SHA256)

    This store overrides writes to insert JSONB via psycopg, since Ibis inserts
    currently cast JSONB expressions to VARCHAR.

    Note: For production deployments, PostgreSQL tables should be created manually using
    database migration tools (e.g., Alembic) or via the SQLModel integration.
    Auto-table creation (auto_create_tables=True) is supported for testing and development
    but not recommended for production use.

    Example:
        ```python
        store = PostgresMetadataStore(
            connection_string="postgresql://user:pass@host:5432/db",
            hash_algorithm=HashAlgorithm.MD5,
        )
        ```
    """

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        schema: str | None = None,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        enable_pgcrypto: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize PostgreSQL metadata store.

        Args:
            connection_string: Connection string like
                "postgresql://user:pass@host:5432/db".
            host: PostgreSQL server host.
            port: PostgreSQL server port (defaults to 5432 when using params).
            user: PostgreSQL username.
            password: PostgreSQL password.
            database: PostgreSQL database name.
            schema: PostgreSQL schema (defaults to search_path).
            connection_params: Additional Ibis connection parameters.
            fallback_stores: Ordered list of read-only fallback stores.
            enable_pgcrypto: If True, attempt to enable pgcrypto on open.
                pgcrypto is required for SHA256 hashing on PostgreSQL.
            **kwargs: Passed to IbisMetadataStore.
        """
        params: dict[str, Any] = dict(connection_params or {})
        explicit_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "schema": schema,
        }
        for key, value in explicit_params.items():
            if value is not None:
                params.setdefault(key, value)

        if connection_string is None and not params:
            raise ValueError(
                "Must provide either connection_string or connection parameters. "
                "Example: connection_string='postgresql://user:pass@localhost:5432/db' "
                "or host='localhost', database='db'."
            )

        if connection_string is None and "port" not in params:
            params["port"] = 5432

        self.host = params.get("host")
        self.port = params.get("port")
        self.database = params.get("database")
        self.schema = params.get("schema")
        self.enable_pgcrypto = enable_pgcrypto
        self._pgcrypto_checked = False

        super().__init__(
            connection_string=connection_string,
            backend="postgres" if connection_string is None else None,
            connection_params=params if connection_string is None else params or None,
            fallback_stores=fallback_stores,
            **kwargs,
        )
        self.versioning_engine_cls = PostgresVersioningEngine

        supported_algorithms = {HashAlgorithm.MD5, HashAlgorithm.SHA256}
        if self.hash_algorithm not in supported_algorithms:
            raise HashAlgorithmNotSupportedError(
                "PostgresMetadataStore supports only MD5 and SHA256 hash algorithms. "
                f"Requested: {self.hash_algorithm}"
            )

    def _create_hash_functions(self) -> dict[HashAlgorithm, Any]:
        """Create PostgreSQL hash functions for Ibis expressions."""
        hash_functions: dict[HashAlgorithm, Any] = {}

        @ibis.udf.scalar.builtin
        def md5(_value: str) -> str:  # ty: ignore[invalid-return-type]
            ...

        def md5_hash(expr):
            return md5(expr.cast(str))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        @ibis.udf.scalar.builtin
        def digest(_value: str, _algorithm: str) -> bytes:  # ty: ignore[invalid-return-type]
            ...

        @ibis.udf.scalar.builtin
        def encode(_value: bytes, _format: str) -> str:  # ty: ignore[invalid-return-type]
            ...

        @ibis.udf.scalar.builtin
        def lower(_value: str) -> str:  # ty: ignore[invalid-return-type]
            ...

        def sha256_hash(expr):
            digest_expr = digest(expr.cast(str), ibis.literal("sha256"))
            encoded = encode(digest_expr, ibis.literal("hex"))
            return lower(encoded)

        hash_functions[HashAlgorithm.SHA256] = sha256_hash

        return hash_functions

    def _build_table_schema(
        self,
        df: pl.DataFrame,
        *,
        json_columns: set[str],
    ) -> ibis.Schema:
        """Build an Ibis schema, forcing JSONB for JSON/struct columns."""
        arrow_schema = cast(pa.Schema, df.to_arrow().schema)
        inferred = ibis.schema(arrow_schema)
        overrides = dict(inferred.items())
        for name, dtype in inferred.items():
            if name in json_columns or dtype.is_struct():
                overrides[name] = dt.jsonb
        return ibis.schema(overrides)

    def _get_json_unpack_exprs(
        self,
        json_column: str,
        field_names: list[str],
    ) -> dict[str, Any]:
        """Return Ibis expressions that extract JSONB fields into flat columns."""

        @ibis.udf.scalar.builtin
        def jsonb_extract_path_text(_data, *paths) -> str:  # ty: ignore[invalid-return-type]
            ...

        exprs: dict[str, Any] = {}
        table = ibis._
        for field_name in field_names:
            flattened_name = f"{json_column}__{field_name}"
            exprs[flattened_name] = jsonb_extract_path_text(
                table[json_column].cast(dt.jsonb),
                ibis.literal(field_name),
            )
        return exprs

    def _get_json_pack_expr(
        self,
        struct_name: str,
        field_columns: Mapping[str, str],
    ) -> Any:
        """Return an Ibis expression that packs flat columns into JSONB."""

        @ibis.udf.scalar.builtin(output_type=dt.jsonb)
        def jsonb_object(_keys: list[str], _values: list[str]) -> str:  # ty: ignore[invalid-return-type]
            ...

        table = ibis._
        keys: list[Any] = []
        values: list[Any] = []
        for field_name, column_name in sorted(field_columns.items()):
            keys.append(ibis.literal(field_name))
            values.append(table[column_name].cast(dt.string))
        if not keys:
            return ibis.literal("{}").cast(dt.jsonb)
        return jsonb_object(ibis.array(keys), ibis.array(values)).cast(dt.jsonb)

    def write_metadata_to_store(
        self,
        feature_key: CoercibleToFeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Write metadata, inserting JSONB columns via psycopg.

        Uses a psycopg insert to ensure JSONB columns stay typed as JSONB,
        since Ibis inserts currently cast JSONB to VARCHAR.
        """
        from metaxy._utils import collect_to_polars

        resolved_key = self._resolve_feature_key(feature_key)
        if self._is_system_table(resolved_key):
            return super().write_metadata_to_store(resolved_key, df, **kwargs)

        # Convert to Polars for proper JSONB handling via psycopg
        # (Ibis inserts cast JSONB to VARCHAR, causing type errors)
        if df.implementation != nw.Implementation.POLARS:
            polars_eager = collect_to_polars(df)
        else:
            polars_df = df.to_native()
            if isinstance(polars_df, pl.LazyFrame):
                polars_df = polars_df.collect()
            polars_eager = cast(pl.DataFrame, polars_df)

        plan = self._resolve_feature_plan(resolved_key)
        polars_eager = self._prepare_polars_json_write_frame(
            polars_eager,
            plan=plan,
        )

        json_columns = {
            METAXY_PROVENANCE_BY_FIELD,
            METAXY_DATA_VERSION_BY_FIELD,
        }

        table_name = self.get_table_name(resolved_key)
        if table_name not in self.conn.list_tables():
            if self.auto_create_tables:
                schema = self._build_table_schema(
                    polars_eager, json_columns=json_columns
                )
                self.conn.create_table(table_name, schema=schema)
            else:
                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist for feature {resolved_key.to_string()}."
                )

        columns = list(polars_eager.columns)
        rows: list[tuple[Any, ...]] = []
        for row in polars_eager.iter_rows(named=True):
            values: list[Any] = []
            for col in columns:
                value = row[col]
                if col in json_columns and value is not None:
                    values.append(Jsonb(value))
                else:
                    values.append(value)
            rows.append(tuple(values))

        raw_conn = cast(Any, self.conn).con
        placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(columns))
        column_sql = sql.SQL(", ").join([sql.Identifier(col) for col in columns])
        query = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({values})").format(
            table=sql.Identifier(table_name),
            columns=column_sql,
            values=placeholders,
        )
        with raw_conn.cursor() as cursor:
            cursor.executemany(query, rows)
        raw_conn.commit()

    def _ensure_pgcrypto_extension(self) -> None:
        """Enable pgcrypto if requested and not already checked."""
        if self._pgcrypto_checked or not self.enable_pgcrypto:
            return

        self._pgcrypto_checked = True
        try:
            raw_conn = cast(Any, self.conn).con
        except AttributeError:
            logger.warning("Could not access raw PostgreSQL connection for pgcrypto.")
            return

        try:
            with raw_conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            raw_conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not enable pgcrypto extension: %s. "
                "If using SHA256, ensure pgcrypto is enabled.",
                exc,
            )

    @contextmanager
    def open(self, mode: AccessMode = "read") -> Iterator[Self]:
        """Open the store and initialize PostgreSQL helpers."""
        with super().open(mode):
            self._ensure_pgcrypto_extension()
            yield self

    def display(self) -> str:
        """Display string for this store."""
        from metaxy.metadata_store.utils import sanitize_uri

        details: list[str] = []
        if self.database:
            details.append(f"database={self.database}")
        if self.schema:
            details.append(f"schema={self.schema}")
        if self.host:
            details.append(f"host={self.host}")
        if self.port:
            details.append(f"port={self.port}")
        detail_str = ", ".join(details)
        if detail_str:
            return f"PostgresMetadataStore({detail_str})"
        if self.connection_string:
            return f"PostgresMetadataStore(connection_string={sanitize_uri(self.connection_string)})"
        return "PostgresMetadataStore()"

    @classmethod
    def config_model(cls) -> type[PostgresMetadataStoreConfig]:
        """Return the configuration model for this store."""
        return PostgresMetadataStoreConfig
