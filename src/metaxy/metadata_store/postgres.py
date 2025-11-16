from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, cast, overload
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import narwhals as nw
import polars as pl
from narwhals.typing import FrameT
from polars.datatypes import DataType as PolarsDataType
from polars.datatypes import DataTypeClass as PolarsDataTypeClass
from psycopg import Error as _PsycopgError
from typing_extensions import Self

from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError
from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.metadata_store.types import AccessMode
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.utils.hashing import get_hash_truncation_length
from metaxy.versioning.ibis import IbisVersioningEngine
from metaxy.models.types import CoercibleToFeatureKey
from metaxy.versioning.flat_engine import IbisFlatVersioningEngine
from metaxy.versioning.types import HashAlgorithm

# Alias for backwards compatibility
PROVENANCE_BY_FIELD_COL = METAXY_PROVENANCE_BY_FIELD

if TYPE_CHECKING:
    from narwhals.typing import Frame

    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import BaseFeature
    from metaxy.models.types import FeatureKey


_PGCRYPTO_ERROR_TYPES: tuple[type[Exception], ...] = (_PsycopgError,)
SchemaMapping = Mapping[str, PolarsDataType | PolarsDataTypeClass]

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


def _sanitize_connection_string(value: str) -> str:
    """Mask passwords in connection strings for safe display."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<redacted>"

    sanitized_netloc = parsed.netloc
    if "@" in sanitized_netloc:
        userinfo, hostinfo = sanitized_netloc.rsplit("@", 1)
        if ":" in userinfo:
            username, _, _ = userinfo.partition(":")
            userinfo = f"{username}:***"
        sanitized_netloc = f"{userinfo}@{hostinfo}"

    sanitized_query = parsed.query
    if parsed.query:
        query_params = parse_qsl(parsed.query, keep_blank_values=True)
        masked_params = []
        changed = False
        for key, val in query_params:
            if key.lower() in {"password", "pwd", "pass"}:
                masked_params.append((key, "***"))
                changed = True
            else:
                masked_params.append((key, val))
        if changed:
            sanitized_query = urlencode(masked_params, doseq=True)

    if sanitized_netloc == parsed.netloc and sanitized_query == parsed.query:
        return value

    sanitized_parts = parsed._replace(netloc=sanitized_netloc, query=sanitized_query)
    return urlunsplit(sanitized_parts)


class PostgresMetadataStore(IbisMetadataStore):
    """
    [PostgreSQL](https://www.postgresql.org/) metadata store - thin wrapper around IbisMetadataStore.

    Provides production-grade metadata storage using PostgreSQL with:
    - Full ACID compliance
    - Extension support (pgcrypto for SHA256)
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
            ImportError: If Ibis or psycopg driver not installed.

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
    def open(self, mode: AccessMode = AccessMode.READ) -> Iterator[Self]:
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

                if not self._has_native_struct_support():
                    self._struct_compat_mode = True
                    message = (
                        "!!! Metaxy WARNING: PostgreSQL backend lacks native STRUCT type support. "
                        "Falling back to JSON serialization compatibility mode. !!!"
                    )
                    logger.warning(
                        message.replace("!!! Metaxy WARNING: ", "").replace("!!!", "")
                    )
                else:
                    self._struct_compat_mode = False
                    message = "!!! Metaxy INFO: PostgreSQL backend has native STRUCT type support. Normal operation. !!!"
                    logger.info(
                        message.replace("!!! Metaxy INFO: ", "").replace("!!!", "")
                    )

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
        # 3. Not using prefer_native=False (which would use Polars components)
        if (
            not self.enable_pgcrypto
            or self.hash_algorithm != HashAlgorithm.SHA256
            or not self._prefer_native
        ):
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

    def _list_tables_robustly(self) -> list[str]:
        """
        Robustly list tables using a raw DBAPI query to avoid bytes vs string issues.
        """
        raw_conn = getattr(self.conn, "con", None)
        if raw_conn is None:
            return self.conn.list_tables()

        try:
            schema_to_query = (self.schema or self._get_current_schema(raw_conn)).strip(
                '"'
            )
            if not schema_to_query:
                schema_to_query = "public"

            with raw_conn.cursor() as cursor:
                cursor.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = %s",
                    (schema_to_query,),
                )
                tables = [_decode_pg_text(row[0]) for row in cursor.fetchall()]
                return tables
        except Exception:
            logger.warning(
                "Robust table listing failed, falling back to Ibis.list_tables(). "
                "This may cause bytes vs. string errors in some environments.",
                exc_info=True,
            )
            return self.conn.list_tables()

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

        Uses PostgresProvenanceTracker when in struct compatibility mode,
        which builds JSON directly in SQL instead of native structs.

        Ensures pgcrypto extension is ready before creating tracker for SHA256 hashing.
        """
        # Ensure pgcrypto is ready before native provenance tracking
        self._ensure_pgcrypto_ready_for_native_provenance()

        if not self._struct_compat_mode:
            # Use standard Ibis tracker (native structs)
            with super()._create_versioning_engine(plan) as engine:
                yield engine
        else:
            # Use PostgreSQL-specific tracker (JSON serialization in SQL)
            if self._conn is None:
                raise RuntimeError(
                    "Cannot create provenance tracker: store is not open. "
                    "Ensure store is used as context manager."
                )

            # Create hash functions for Ibis expressions
            hash_functions = self._create_hash_functions()

            # Create PostgreSQL tracker (uses JSON instead of structs)
            tracker = PostgresProvenanceTracker(
                plan=plan,
                hash_functions=hash_functions,
            )

            try:
                yield tracker
            finally:
                # No cleanup needed
                pass

    @overload
    def resolve_update(
        self,
        feature: type[BaseFeature],
        *,
        samples: Frame | None = None,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: Literal[False] = False,
        **kwargs: Any,
    ) -> Increment: ...

    @overload
    def resolve_update(
        self,
        feature: type[BaseFeature],
        *,
        samples: Frame | None = None,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: Literal[True],
        **kwargs: Any,
    ) -> LazyIncrement: ...

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
        """Override to ensure table existence checks use robust listing."""
        table_name = feature_key.table_name
        if table_name in self._list_tables_robustly():
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
            sanitized = _sanitize_connection_string(self.connection_string)
            return f"PostgresMetadataStore(connection_string={sanitized})"
        return "PostgresMetadataStore()"

    def _has_native_struct_support(self) -> bool:
        """
        NOTE: Native struct support for PostgreSQL in the Ibis/sqlglot stack has proven
        unreliable across different environments. We are temporarily disabling it
        to enforce the more stable JSON serialization path.
        """
        return False

    @staticmethod
    def _serialize_struct_column(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
        """Convert a struct column to canonical JSON string for storage.

        Args:
            df: Polars DataFrame containing the struct column
            column_name: Name of the struct column to serialize

        Returns:
            DataFrame with the struct column replaced by JSON strings
        """
        if column_name not in df.columns:
            return df

        column = df.get_column(column_name)
        serialized = [
            json.dumps(value, sort_keys=True) if value is not None else None
            for value in column.to_list()
        ]
        return df.with_columns(
            pl.Series(name=column_name, values=serialized, dtype=pl.String)
        )

    @staticmethod
    def _serialize_struct_columns(df: pl.DataFrame) -> pl.DataFrame:
        """Convert all struct columns to canonical JSON strings for storage.

        Serializes both metaxy_provenance_by_field and metaxy_data_version_by_field
        columns if they exist in the DataFrame.

        Args:
            df: Polars DataFrame potentially containing struct columns

        Returns:
            DataFrame with all struct columns replaced by JSON strings
        """
        from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD

        # Serialize provenance_by_field
        df = PostgresMetadataStore._serialize_struct_column(
            df, METAXY_PROVENANCE_BY_FIELD
        )

        # Serialize data_version_by_field
        df = PostgresMetadataStore._serialize_struct_column(
            df, METAXY_DATA_VERSION_BY_FIELD
        )

        return df

    @staticmethod
    def _deserialize_provenance_column(df: pl.DataFrame) -> pl.DataFrame:
        """Restore struct provenance columns from JSON text or keep dicts.

        Handles both cases:
        - JSON strings from serialization: Parse with json.loads()
        - Python dicts from Ibis JSONB deserialization: Keep as-is

        Processes all columns starting with METAXY_PROVENANCE_BY_FIELD prefix,
        including renamed parent columns like "metaxy_provenance_by_field__parent",
        and METAXY_DATA_VERSION_BY_FIELD columns.
        """
        from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD

        # Find all provenance columns (main + renamed parent columns after join)
        provenance_cols = [
            col for col in df.columns if col.startswith(METAXY_PROVENANCE_BY_FIELD)
        ]

        # Also find data_version_by_field columns
        data_version_cols = [
            col for col in df.columns if col.startswith(METAXY_DATA_VERSION_BY_FIELD)
        ]

        # Combine both types of struct columns
        struct_cols = provenance_cols + data_version_cols

        if not struct_cols:
            return df

        # Deserialize each struct column
        for col_name in struct_cols:
            column = df.get_column(col_name)
            decoded = [
                # If value is already a dict (from Ibis JSONB), keep it
                # If value is a string (from serialization), parse it
                json.loads(value) if isinstance(value, str) else value
                for value in column.to_list()
            ]
            df = df.with_columns(pl.Series(name=col_name, values=decoded))

        return df

    @staticmethod
    def _ensure_utc_created_at(df: pl.DataFrame) -> pl.DataFrame:
        """Normalize metaxy_created_at columns to UTC-aware timestamps."""
        from polars.datatypes import Datetime

        from metaxy.models.constants import METAXY_CREATED_AT

        updates: list[pl.Expr] = []
        for col_name, dtype in df.schema.items():
            if (
                col_name.startswith(METAXY_CREATED_AT)
                and isinstance(dtype, Datetime)
                and dtype.time_zone is None
            ):
                updates.append(
                    pl.col(col_name).dt.replace_time_zone("UTC").alias(col_name)
                )

        if not updates:
            return df

        return df.with_columns(updates)


class PostgresProvenanceTracker(IbisVersioningEngine):  # pyright: ignore[reportIncompatibleMethodOverride]
    """Provenance tracker for PostgreSQL that uses JSON serialization for structs.

    PostgreSQL doesn't reliably support native STRUCT/ROW types across all versions.
    This tracker works directly with Ibis tables and uses PostgreSQL's JSON text operations,
    keeping everything in the database without materializing to Polars.

    Uses IbisVersioningEngine methods via manual initialization.
    """

    def __init__(self, plan, hash_functions):
        """Initialize PostgreSQL provenance tracker.

        Args:
            plan: Feature plan
            hash_functions: Dict mapping HashAlgorithm to Ibis hash functions
        """
        super().__init__(plan=plan, hash_functions=hash_functions)

    @staticmethod
    def _ensure_ibis_frame(
        df: nw.DataFrame[Any] | nw.LazyFrame[Any],
    ) -> tuple[Any, bool]:
        """Return the underlying Ibis table and whether the frame is lazy."""
        import ibis.expr.types

        if df.implementation != nw.Implementation.IBIS:
            raise TypeError("PostgreSQL tracker only works with Ibis-backed frames.")

        table = cast(ibis.expr.types.Table, df.to_native())
        return table, isinstance(df, nw.LazyFrame)

    @staticmethod
    def _wrap_like(
        df: FrameT,
        table: Any,
        *,
        is_lazy: bool,
    ) -> FrameT:
        """Wrap an Ibis table back into a Narwhals frame matching the input."""
        if df.implementation != nw.Implementation.IBIS:
            raise TypeError("PostgreSQL tracker only works with Ibis-backed frames.")

        wrapped = cast(nw.LazyFrame[Any], nw.from_native(table, eager_only=False))
        if is_lazy:
            return cast(FrameT, wrapped)
        return cast(FrameT, wrapped.collect())

    # Methods below access IbisVersioningEngine attributes set by __init__
    # pyright doesn't see them, so we use ignore comments where needed

    @staticmethod
    def _concat_ibis_parts(parts: Sequence[Any]):
        """Concatenate a list of Ibis string expressions without ibis.concat."""
        parts_list = list(parts)
        if not parts_list:
            raise ValueError("Cannot concatenate zero expressions")

        expr = parts_list[0]
        for part in parts_list[1:]:
            expr = expr.concat(part)
        return expr

    @staticmethod
    def _concat_with_separator(parts: Sequence[Any]):
        """Concatenate expressions with a '|' separator between each component."""
        import ibis

        parts_list = list(parts)
        if not parts_list:
            raise ValueError("Cannot concatenate zero expressions")

        interleaved = [parts_list[0]]
        for part in parts_list[1:]:
            interleaved.append(ibis.literal("|"))
            interleaved.append(part)

        return PostgresProvenanceTracker._concat_ibis_parts(interleaved)

    @staticmethod
    def build_struct_column(
        df: FrameT, struct_name: str, field_columns: dict[str, str]
    ) -> FrameT:
        """Build a JSON column from existing columns using PostgreSQL string concatenation.

        Args:
            df: Narwhals DataFrame backed by Ibis
            struct_name: Name for the new JSON column
            field_columns: Mapping from struct field names to column names

        Returns:
            Narwhals DataFrame with new JSON column added, backed by Ibis.
        """

        import ibis

        table, is_lazy = PostgresProvenanceTracker._ensure_ibis_frame(df)

        # Build JSON object using string concatenation
        # Format: {"field1": "value1", "field2": "value2"}
        concat_parts = [ibis.literal("{")]
        first = True
        for field_name, col_name in sorted(field_columns.items()):
            if not first:
                concat_parts.append(ibis.literal(","))
            first = False

            # Add "fieldname":"value"
            concat_parts.append(ibis.literal(f'"{field_name}":"'))
            # Cast to text (hash values are plain strings)
            col_value = table[col_name].cast(str)
            concat_parts.append(col_value)  # pyright: ignore[reportArgumentType]
            concat_parts.append(ibis.literal('"'))

        concat_parts.append(ibis.literal("}"))

        # Use concatenation helper and cast to JSONB to match struct semantics
        json_expr = PostgresProvenanceTracker._concat_ibis_parts(concat_parts).cast(
            "jsonb"
        )

        # Add JSON column
        result_table = table.mutate(**{struct_name: json_expr})
        return PostgresProvenanceTracker._wrap_like(df, result_table, is_lazy=is_lazy)

    def load_upstream_with_provenance(self, upstream, hash_algo, filters):
        """Load upstream with provenance, extracting JSON fields using PostgreSQL operators.

        This is the key method that handles JSON field extraction instead of struct field access.
        """
        import ibis
        import narwhals as nw

        logger.warning("[POSTGRES DEBUG] load_upstream_with_provenance called!")

        hash_length = get_hash_truncation_length()

        # Prepare upstream (rename, filter, join)
        df = self.prepare_upstream(upstream, filters)  # pyright: ignore[reportAttributeAccessIssue]

        # Convert to Ibis table for JSON operations
        ibis_table, _ = self._ensure_ibis_frame(df)

        # Build concatenation columns for each field, extracting from JSON
        temp_concat_cols: dict[str, str] = {}

        for field_spec in self.plan.feature.fields:  # pyright: ignore[reportAttributeAccessIssue]
            field_key_str = field_spec.key.to_struct_key()
            temp_col_name = f"__concat_{field_key_str}"
            temp_concat_cols[field_key_str] = temp_col_name

            # Build concatenation: field_key + code_version + parent_provenances
            components = [
                ibis.literal(field_spec.key.to_string()),
                ibis.literal(str(field_spec.code_version)),
            ]

            # Extract parent field provenances from JSONB
            # Sort parent fields for deterministic order (matches base tracker)
            parent_fields = self.plan.get_parent_fields_for_field(field_spec.key)  # pyright: ignore[reportAttributeAccessIssue]
            for fq_field_key in sorted(parent_fields.keys()):
                parent_field_spec = parent_fields[fq_field_key]
                # Use fq_field_key.to_string() to match base tracker format
                components.append(ibis.literal(fq_field_key.to_string()))

                # Extract JSONB field using Ibis's [] operator
                # With JSONB type, Ibis supports: table[jsonb_column][key]
                provenance_col = self.get_renamed_METAXY_PROVENANCE_BY_FIELD(  # pyright: ignore[reportAttributeAccessIssue]
                    fq_field_key.feature
                )
                parent_key_str = parent_field_spec.key.to_struct_key()

                # Ibis supports JSONB field extraction natively
                # Use JSON unwrap to get unquoted text value (->> operator)
                json_value = ibis_table[provenance_col][parent_key_str].str
                components.append(json_value)

            concat_expr = self._concat_with_separator(components)
            ibis_table = ibis_table.mutate(**{temp_col_name: concat_expr})

        # Convert back to Narwhals for hashing
        df = nw.from_native(ibis_table)

        # Hash each concatenation column
        temp_hash_cols: dict[str, str] = {}
        for field_key_str, concat_col in temp_concat_cols.items():
            hash_col_name = f"__hash_{field_key_str}"
            temp_hash_cols[field_key_str] = hash_col_name

            # Hash the concatenated string column
            df = self.hash_string_column(  # pyright: ignore[reportAttributeAccessIssue]
                df, concat_col, hash_col_name, hash_algo
            ).with_columns(nw.col(hash_col_name).str.slice(0, hash_length))

        # Build struct from hash columns
        df = self.build_struct_column(df, METAXY_PROVENANCE_BY_FIELD, temp_hash_cols)

        # Recompute table reference for JSON extraction
        ibis_table, _ = self._ensure_ibis_frame(df)

        # Add sample-level provenance
        field_names = sorted([f.key.to_struct_key() for f in self.plan.feature.fields])  # pyright: ignore[reportAttributeAccessIssue]

        sample_components = [
            ibis_table[METAXY_PROVENANCE_BY_FIELD][field_name].str
            for field_name in field_names
        ]
        sample_concat_expr = self._concat_with_separator(sample_components)
        ibis_table = ibis_table.mutate(__sample_concat=sample_concat_expr)

        # Hash sample-level provenance
        df = nw.from_native(ibis_table)
        df = self.hash_string_column(  # pyright: ignore[reportAttributeAccessIssue]
            df, "__sample_concat", "metaxy_provenance", hash_algo
        ).with_columns(nw.col("metaxy_provenance").str.slice(0, hash_length))
        df = df.drop("__sample_concat")

        # Drop temporary columns
        df = df.drop(*list(temp_concat_cols.values()))
        df = df.drop(*list(temp_hash_cols.values()))

        # Drop version columns if present
        version_columns = ["metaxy_feature_version", "metaxy_snapshot_version"]
        current_columns = df.collect_schema().names()
        columns_to_drop = [col for col in version_columns if col in current_columns]
        if columns_to_drop:
            df = df.drop(*columns_to_drop)

        # Mirror base tracker behavior: default data version columns to provenance values.
        from metaxy.models.constants import (
            METAXY_DATA_VERSION,
            METAXY_DATA_VERSION_BY_FIELD,
        )

        df = df.with_columns(
            nw.col("metaxy_provenance").alias(METAXY_DATA_VERSION),
            nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD),
        )

        return df
