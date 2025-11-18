from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import narwhals as nw
import polars as pl
from polars.datatypes import DataType as PolarsDataType
from polars.datatypes import DataTypeClass as PolarsDataTypeClass
from psycopg import Error as _PsycopgError
from typing_extensions import Self

from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError
from metaxy.metadata_store.ibis_json_compat import IbisJsonCompatStore
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import sanitize_uri
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.types import CoercibleToFeatureKey
from metaxy.versioning.types import HashAlgorithm

# Alias for backwards compatibility
PROVENANCE_BY_FIELD_COL = METAXY_PROVENANCE_BY_FIELD

if TYPE_CHECKING:
    from narwhals.typing import Frame

    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.types import FeatureKey


_PGCRYPTO_ERROR_TYPES: tuple[type[Exception], ...] = (_PsycopgError,)

logger = logging.getLogger(__name__)
SchemaMapping = Mapping[str, PolarsDataType | PolarsDataTypeClass]
SchemaMapping = Mapping[str, PolarsDataType | PolarsDataTypeClass]


def _decode_pg_text(value: Any) -> str:
    """Convert PostgreSQL driver outputs to text consistently."""
    if value is None:
        return ""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


class PostgresMetadataStore(IbisJsonCompatStore):
    """
    [PostgreSQL](https://www.postgresql.org/) metadata store using JSONB storage.

    Provides production-grade metadata storage using PostgreSQL with:
    - Full ACID compliance
    - JSONB storage for struct-like columns via Ibis JSON packing
    - Extension support (pgcrypto for SHA256)

    Note: For production deployments, PostgreSQL tables should be created manually using
    database migration tools (e.g., Alembic) or via the SQLModel integration.
    Auto-table creation (auto_create_tables=True) is supported for testing and development
    but not recommended for production use.
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
    ):
        """
        Initialize [PostgreSQL](https://www.postgresql.org/) metadata store.

        Args:
            connection_string: PostgreSQL connection string.
                Format: `postgresql://user:pass@host:port/database`.
                Supports additional parameters: `?options=-c%20statement_timeout=30s`.
            host: Server host (used when connection_string not provided).
            port: Server port (defaults to 5432 when omitted).
            user: Database user.
            password: Database password.
            database: Database name.
            schema: Target schema for table isolation (defaults to search_path when omitted).
                Recommended for production deployments.
            connection_params: Additional Ibis PostgreSQL connection parameters
                (e.g., `{"sslmode": "require", "connect_timeout": 10}`).
            fallback_stores: Ordered list of read-only fallback stores for branch deployments.
            enable_pgcrypto: Whether to auto-enable pgcrypto extension before native SHA256 hashing runs (default: False).
                Set to True if you want Metaxy to manage pgcrypto automatically; leave False if pgcrypto is already enabled
                or your database user lacks CREATE EXTENSION privileges.
            **kwargs: Passed to [metaxy.metadata_store.ibis.IbisMetadataStore][]
                (e.g., `hash_algorithm`, `auto_create_tables`).

        Raises:
            ValueError: If neither connection_string nor connection parameters provided.

        Note:
            When using SHA256 hash algorithm, pgcrypto extension is required.
            The store attempts to enable it automatically the first time
            SHA256 hashing runs inside `resolve_update()` unless
            `enable_pgcrypto=False`.
        """
        kwargs.setdefault("versioning_engine", "polars")
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
        self._pgcrypto_extension_checked = False

        # IbisJsonCompatStore handles dict-based versioning engine setup
        super().__init__(
            connection_string=connection_string,
            backend="postgres" if connection_string is None else None,
            connection_params=params if connection_string is None else params or None,
            fallback_stores=fallback_stores,
            **kwargs,
        )

        supported_algorithms = {HashAlgorithm.MD5, HashAlgorithm.SHA256}
        if self.hash_algorithm not in supported_algorithms:
            raise HashAlgorithmNotSupportedError(
                f"PostgresMetadataStore supports only MD5 and SHA256 hash algorithms. "
                f"Requested: {self.hash_algorithm}"
            )

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for PostgreSQL stores.

        Uses MD5 as it's built-in and requires no extensions.
        For SHA256 support, explicitly set hash_algorithm=HashAlgorithm.SHA256
        and ensure pgcrypto extension is enabled.
        For production you must self-install the necessary posgres extensions.
        """
        return HashAlgorithm.MD5

    def _create_hash_functions(self):
        """Create PostgreSQL-specific hash functions for Ibis expressions."""
        import ibis

        hash_functions = {}

        @ibis.udf.scalar.builtin
        def MD5(_x: str) -> str:  # noqa: N802
            """PostgreSQL MD5() function."""
            ...

        def md5_hash(col_expr):
            return MD5(col_expr.cast(str))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        @ibis.udf.scalar.builtin
        def digest(_value: str, _algorithm: str) -> bytes:
            """pgcrypto digest() function."""
            ...

        @ibis.udf.scalar.builtin
        def encode(_value: bytes, _fmt: str) -> str:
            """PostgreSQL encode() function."""
            ...

        @ibis.udf.scalar.builtin
        def lower(_value: str) -> str:
            """PostgreSQL lower() function."""
            ...

        def sha256_hash(col_expr):
            digest_expr = digest(col_expr.cast(str), ibis.literal("sha256"))
            encoded = encode(digest_expr, ibis.literal("hex"))
            return lower(encoded)

        hash_functions[HashAlgorithm.SHA256] = sha256_hash

        return hash_functions

    @contextmanager
    def open(self, mode: AccessMode = "read") -> Iterator[Self]:
        """Open connection to PostgreSQL and perform capability checks.

        Args:
            mode: Access mode (READ or WRITE). Defaults to READ.

        Yields:
            Self: The store instance with connection open

        Raises:
            ImportError: If psycopg driver not installed.
            Various database errors: If connection fails.
        """
        # Call parent context manager to establish connection
        with super().open(mode):
            try:
                # Reset pgcrypto check for the new connection
                self._pgcrypto_extension_checked = False

                yield self
            finally:
                # Cleanup is handled by parent's finally block
                pass

    def _ensure_pgcrypto_ready_for_native_provenance(self) -> None:
        """Enable pgcrypto before running native SHA256 provenance tracking.

        Note: pgcrypto is needed because we still use native SHA256 hashing
        via SQL DIGEST() function.
        """
        if self._pgcrypto_extension_checked:
            return

        # Only enable pgcrypto if:
        # 1. enable_pgcrypto flag is True
        # 2. Using SHA256 hash algorithm (which requires pgcrypto)
        if not self.enable_pgcrypto or self.hash_algorithm != HashAlgorithm.SHA256:
            return

        try:
            self._ensure_pgcrypto_extension()
        finally:
            self._pgcrypto_extension_checked = True

    def _ensure_pgcrypto_extension(self) -> None:
        """Ensure pgcrypto extension is enabled for SHA256 support.

        Attempts to create the extension if it doesn't exist. Logs a warning
        if the user lacks privileges rather than failing, as the extension
        might already be enabled.
        """
        try:
            # Use underlying connection to execute DDL (Ibis doesn't expose CREATE EXTENSION)
            raw_conn = cast(Any, self.conn).con
            with raw_conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            raw_conn.commit()
            logger.debug("pgcrypto extension enabled successfully")
        except _PGCRYPTO_ERROR_TYPES as err:
            self._log_pgcrypto_warning(err)
        except AttributeError as err:
            self._log_pgcrypto_warning(err)

    def _get_current_schema(self, raw_conn: Any) -> str:
        """Get the current schema from the search_path."""
        try:
            with raw_conn.cursor() as cursor:
                cursor.execute("SHOW search_path")
                result = cursor.fetchone()
        except Exception:
            logger.debug(
                "Could not determine current schema; defaulting to public",
                exc_info=True,
            )
            return "public"

        search_path = _decode_pg_text(result[0]) if result else ""
        if not search_path:
            return "public"

        for raw_path in search_path.split(","):
            trimmed = raw_path.strip()
            if not trimmed:
                continue
            unquoted = trimmed.strip('"')
            if unquoted == "$user" or not unquoted:
                continue
            return unquoted
        return "public"

    @staticmethod
    def _log_pgcrypto_warning(error: Exception) -> None:
        logger.warning(
            "Could not enable pgcrypto extension: %s. "
            "If using SHA256 hash algorithm, ensure pgcrypto is enabled "
            "or have a DBA run: CREATE EXTENSION IF NOT EXISTS pgcrypto;",
            error,
        )

    @contextmanager
    def _create_versioning_engine(self, plan):
        """Create provenance engine for PostgreSQL.

        Ensures pgcrypto extension is ready before creating tracker for SHA256 hashing.
        Delegates to IbisJsonCompatStore for JSON packing/unpacking.
        """
        # Ensure pgcrypto is ready before native provenance tracking
        self._ensure_pgcrypto_ready_for_native_provenance()

        # Use parent implementation from JsonCompatStore
        with super()._create_versioning_engine(plan) as engine:
            yield engine

    def _get_json_unpack_exprs(
        self, json_column: str, field_names: list[str]
    ) -> dict[str, Any]:
        """Extract flattened columns from a JSONB column using PostgreSQL functions."""
        import ibis

        @ibis.udf.scalar.builtin
        def jsonb_extract_path_text(_data, *paths) -> str: ...

        exprs: dict[str, Any] = {}
        table = ibis._  # placeholder for the current table context
        for field_name in field_names:
            flattened_name = f"{json_column}__{field_name}"
            exprs[flattened_name] = jsonb_extract_path_text(
                table[json_column].cast("jsonb"),
                ibis.literal(field_name),
            )
        return exprs

    def _get_json_pack_expr(
        self, struct_name: str, field_columns: Mapping[str, str]
    ) -> Any:
        """Pack flattened provenance columns into a JSONB object."""
        import ibis
        import ibis.expr.datatypes as dt

        @ibis.udf.scalar.builtin(output_type=dt.jsonb)
        def jsonb_object(_keys: list[str], _values: list[str]) -> str: ...

        keys: list[Any] = []
        values: list[Any] = []
        table = ibis._
        for field_name, col_name in sorted(field_columns.items()):
            keys.append(ibis.literal(field_name))
            values.append(table[col_name].cast(dt.string))
        return jsonb_object(ibis.array(keys), ibis.array(values))

    @staticmethod
    def _get_postgres_type(dtype: PolarsDataType | PolarsDataTypeClass) -> str:
        """Convert Polars dtype to PostgreSQL type.

        Struct types are mapped to JSONB for native PostgreSQL JSON support.
        """
        if dtype == pl.String:
            return "VARCHAR"
        elif dtype == pl.Int64:
            return "BIGINT"
        elif dtype == pl.Int32:
            return "INTEGER"
        elif dtype == pl.Float64:
            return "DOUBLE PRECISION"
        elif dtype == pl.Boolean:
            return "BOOLEAN"
        elif dtype == pl.Datetime:
            return "TIMESTAMP(6)"
        elif dtype == pl.Date:
            return "DATE"
        elif isinstance(dtype, pl.Struct):
            # Use JSONB for struct types (native PostgreSQL JSON with indexing)
            return "JSONB"
        else:
            return "VARCHAR"

    def _create_table_ddl(self, table_name: str, schema: SchemaMapping) -> str:
        """Generate CREATE TABLE IF NOT EXISTS DDL from Polars schema."""
        columns = [
            f'"{col_name}" {self._get_postgres_type(dtype)}'
            for col_name, dtype in schema.items()
        ]
        columns_sql = ", ".join(columns)
        return f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql})'

    def _execute_ddl(self, ddl: str):
        """Execute DDL with proper connection handling."""
        raw_conn = cast(Any, self.conn).con
        original_autocommit = raw_conn.autocommit
        try:
            raw_conn.autocommit = True
            with raw_conn.cursor() as cursor:
                cursor.execute(ddl)
        finally:
            raw_conn.autocommit = original_autocommit

    def _create_table_from_dataframe(self, table_name: str, df: pl.DataFrame) -> None:
        """Create table from DataFrame schema using raw DDL."""
        # Convert DataFrame schema to SchemaMapping
        schema: dict[str, PolarsDataType | PolarsDataTypeClass] = {}
        for col_name in df.columns:
            dtype = df.schema[col_name]
            # Convert pl.Null to pl.Utf8 for table creation
            if dtype == pl.Null:
                schema[col_name] = pl.Utf8
            else:
                schema[col_name] = dtype

        ddl = self._create_table_ddl(table_name, cast(SchemaMapping, schema))
        self._execute_ddl(ddl)

    def _create_system_tables(self) -> None:
        """
        Atomically create system tables using `CREATE TABLE IF NOT EXISTS`.
        Uses raw SQL to avoid Ibis limitations and ensure proper schema generation.
        """
        from metaxy.metadata_store.system import (
            EVENTS_KEY,
            FEATURE_VERSIONS_KEY,
            FEATURE_VERSIONS_SCHEMA,
        )
        from metaxy.metadata_store.system.events import EVENTS_SCHEMA

        ddl_fv = self._create_table_ddl(
            FEATURE_VERSIONS_KEY.table_name,
            cast(SchemaMapping, FEATURE_VERSIONS_SCHEMA),
        )
        self._execute_ddl(ddl_fv)

        ddl_me = self._create_table_ddl(
            EVENTS_KEY.table_name,
            cast(SchemaMapping, EVENTS_SCHEMA),
        )
        self._execute_ddl(ddl_me)

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Cast materialization_id and NULL columns to string before write."""
        from metaxy.models.constants import METAXY_MATERIALIZATION_ID

        if METAXY_MATERIALIZATION_ID in df.columns:
            df = df.with_columns(nw.col(METAXY_MATERIALIZATION_ID).cast(nw.String))

        # Postgres rejects NULL-typed columns; cast Null columns to string for Polars inputs
        if df.implementation == nw.Implementation.POLARS:
            null_cols = [
                col
                for col, dtype in df.schema.items()
                if dtype == pl.Null  # type: ignore[attr-defined]
            ]
            if null_cols:
                df = df.with_columns(
                    [nw.col(col).cast(nw.String).alias(col) for col in null_cols]
                )

        super().write_metadata_to_store(feature_key, df, **kwargs)

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop the table for a feature.

        Args:
            feature_key: Feature key to drop metadata for
        """
        table_name = self.get_table_name(feature_key)
        if table_name in self.conn.list_tables():
            self.conn.drop_table(table_name)

    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata, deserializing provenance structs when needed."""
        from metaxy.models.constants import (
            METAXY_DATA_VERSION,
            METAXY_PROVENANCE,
        )

        # Use robust table checking
        feature_key = self._resolve_feature_key(feature)
        table_name = feature_key.table_name

        if table_name not in self._list_tables_robustly():
            return None

        table = self.conn.table(table_name)
        lazy_frame: nw.LazyFrame[Any] = nw.from_native(table, eager_only=False)

        if feature_version is not None:
            lazy_frame = lazy_frame.filter(
                nw.col("metaxy_feature_version") == feature_version  # ty: ignore[invalid-argument-type]
            )

        if filters is not None:
            for filter_expr in filters:
                lazy_frame = lazy_frame.filter(filter_expr)  # ty: ignore[invalid-argument-type]

        if columns is not None:
            lazy_frame = lazy_frame.select(columns)

        # Ensure flattened provenance/data_version columns exist for this feature's fields
        plan = self._resolve_feature_plan(feature_key)

        expected_fields = [field.key.to_struct_key() for field in plan.feature.fields]
        for field_name in expected_fields:
            prov_flat = f"{METAXY_PROVENANCE_BY_FIELD}__{field_name}"
            if prov_flat not in lazy_frame.columns:
                if METAXY_PROVENANCE in lazy_frame.columns:
                    lazy_frame = lazy_frame.with_columns(
                        nw.col(METAXY_PROVENANCE).alias(prov_flat)
                    )
                else:
                    lazy_frame = lazy_frame.with_columns(
                        nw.lit(None, dtype=nw.String).alias(prov_flat)
                    )

            data_flat = f"{METAXY_DATA_VERSION_BY_FIELD}__{field_name}"
            if data_flat not in lazy_frame.columns:
                if METAXY_DATA_VERSION in lazy_frame.columns:
                    lazy_frame = lazy_frame.with_columns(
                        nw.col(METAXY_DATA_VERSION).alias(data_flat)
                    )
                else:
                    lazy_frame = lazy_frame.with_columns(
                        nw.lit(None, dtype=nw.String).alias(data_flat)
                    )

        return lazy_frame

    def write_metadata_to_store(self, feature_key: FeatureKey, df, **kwargs):
        """Ensure string-typed materialization_id before delegating write."""
        import polars as pl

        from metaxy.models.constants import METAXY_MATERIALIZATION_ID

        if METAXY_MATERIALIZATION_ID in df.columns:
            df = df.with_columns(nw.col(METAXY_MATERIALIZATION_ID).cast(nw.String))

        # Postgres rejects NULL-typed columns; cast Null columns to string for Polars inputs
        if df.implementation == nw.Implementation.POLARS:
            null_cols = [
                col
                for col, dtype in df.schema.items()
                if dtype == pl.Null  # type: ignore[attr-defined]
            ]
            if null_cols:
                df = df.with_columns(
                    [nw.col(col).cast(nw.String).alias(col) for col in null_cols]
                )

        super().write_metadata_to_store(feature_key, df, **kwargs)

    def write_metadata_to_store(self, feature_key: FeatureKey, df, **kwargs):
        """Ensure string-typed materialization_id before delegating write."""

        import polars as pl

        from metaxy.models.constants import METAXY_MATERIALIZATION_ID

        if METAXY_MATERIALIZATION_ID in df.columns:
            df = df.with_columns(nw.col(METAXY_MATERIALIZATION_ID).cast(nw.String))

        # Postgres rejects NULL-typed columns; cast Null columns to string for Polars inputs
        if df.implementation == nw.Implementation.POLARS:
            null_cols = [
                col
                for col, dtype in df.schema.items()
                if dtype == pl.Null  # type: ignore[attr-defined]
            ]
            if null_cols:
                df = df.with_columns(
                    [nw.col(col).cast(nw.String).alias(col) for col in null_cols]
                )

        super().write_metadata_to_store(feature_key, df, **kwargs)

    def read_metadata(
        self,
        feature: CoercibleToFeatureKey,
        *,
        feature_version=None,
        filters=None,
        columns=None,
        allow_fallback=True,
        current_only=True,
        latest_only=True,
    ):
        """Read metadata using the flattened JSON-compatible layout."""
        return super().read_metadata(
            feature,
            feature_version=feature_version,
            filters=filters,
            columns=columns,
            allow_fallback=allow_fallback,
            current_only=current_only,
            latest_only=latest_only,
        )

    def display(self) -> str:
        """Display string for this store."""
        details: list[str] = []
        if self.database:
            details.append(f"database={self.database}")
        if self.schema:
            details.append(f"schema={self.schema}")
        if self.host:
            details.append(f"host={self.host}")
        if self.port:
            details.append(f"port={self.port}")

        if self._is_open:
            details.append(f"features={len(self._list_features_local())}")  # ty: ignore[unresolved-attribute]
        detail_str = ", ".join(details)
        if detail_str:
            return f"PostgresMetadataStore({detail_str})"
        if self.connection_string:
            sanitized = sanitize_uri(self.connection_string)
            return f"PostgresMetadataStore(connection_string={sanitized})"
        return "PostgresMetadataStore()"
