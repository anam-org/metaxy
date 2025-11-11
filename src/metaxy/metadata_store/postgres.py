"""PostgreSQL metadata store - thin wrapper around IbisMetadataStore.

Provides production-grade metadata storage using PostgreSQL with:
- Full ACID compliance
- Schema isolation
- Extension support (pgcrypto for SHA256)
- Connection pooling support
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, cast

import narwhals as nw
import polars as pl
from polars.datatypes import DataType as PolarsDataType
from polars.datatypes import DataTypeClass as PolarsDataTypeClass

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import BaseFeature
    from metaxy.models.types import FeatureKey


from psycopg import Error as _PsycopgError
from psycopg.cursor import Cursor as PsycopgCursor

from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError
from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD
from metaxy.models.types import CoercibleToFeatureKey
from metaxy.versioning.flat_engine import IbisFlatVersioningEngine
from metaxy.versioning.types import HashAlgorithm

# Alias for backwards compatibility
PROVENANCE_BY_FIELD_COL = METAXY_PROVENANCE_BY_FIELD

logger = logging.getLogger(__name__)
_PGCRYPTO_ERROR_TYPES: tuple[type[Exception], ...] = (_PsycopgError,)
SchemaMapping = Mapping[str, PolarsDataType | PolarsDataTypeClass]

_original_execute = PsycopgCursor.execute
_original_iter = PsycopgCursor.__iter__


def _decoder(byte_value):
    return byte_value.decode("utf-8", "replace")


def _patched_execute(self, query, params=None, *args, **kwargs):
    """
    Patched `execute` to sanitize parameters on the way IN to the database.
    This fixes the `name = bytea` error.
    """
    sanitized_params = params
    if params:
        if isinstance(params, dict):
            sanitized_params = {}
            for key, value in params.items():
                if isinstance(value, list) and value and isinstance(value[0], bytes):
                    sanitized_params[key] = [_decoder(item) for item in value]
                elif isinstance(value, bytes):
                    sanitized_params[key] = _decoder(value)
                else:
                    sanitized_params[key] = value
        elif isinstance(params, (tuple, list)):
            sanitized_params = tuple(
                _decoder(p) if isinstance(p, bytes) else p for p in params
            )

    return _original_execute(self, query, sanitized_params, *args, **kwargs)


def _patched_iter(self):
    """
    Patched `__iter__` to sanitize results on the way OUT of the database.
    This fixes the `TypeError: startswith` error.
    """
    # Wrap the original iterator in a new generator that decodes each row.
    for row in _original_iter(self):
        yield tuple(_decoder(item) if isinstance(item, bytes) else item for item in row)


# 3. Apply the patches to the PsycopgCursor class.
PsycopgCursor.execute = _patched_execute
PsycopgCursor.__iter__ = _patched_iter

logger.info(
    "Applied final, global, brute-force patch to psycopg.Cursor at module load time."
)


class PostgresMetadataStore(IbisMetadataStore):
    """
    [PostgreSQL](https://www.postgresql.org/) metadata store using [Ibis](https://ibis-project.org/) backend.

    Provides production-grade metadata storage with full ACID compliance, schema isolation,
    and support for advanced hash algorithms via PostgreSQL extensions.

    **Hash Algorithm Support:**

    - **MD5** (default): Built-in, no extensions required
    - **SHA256**: Requires `pgcrypto` extension (auto-enabled before native hashing runs)

    The store automatically enables `pgcrypto` (when configured) the first time
    it needs to compute SHA256 hashes during `resolve_update()`. This avoids
    running `CREATE EXTENSION` during store initialization.

    **Schema Isolation:**

    Use the `schema` parameter to isolate metadata in a dedicated schema.
    If not specified, tables are created in the user's search_path (typically `public`).

    Example: Connection String
        ```py
        store = PostgresMetadataStore("postgresql://user:pass@localhost:5432/metadata")
        ```

    Example: Connection Parameters
        ```py
        store = PostgresMetadataStore(
            host="localhost",
            port=5432,
            user="ml",
            password="secret",
            database="metaxy",
            schema="features",  # Optional schema isolation
        )
        ```

    Example: SHA256 Hash Algorithm
        ```py
        # pgcrypto extension auto-enabled during resolve_update()
        store = PostgresMetadataStore(
            "postgresql://user:pass@localhost:5432/metadata",
            hash_algorithm=HashAlgorithm.SHA256,
        )
        ```

    Example: With Fallback Stores
        ```py
        # Read from prod, write to dev
        prod_store = PostgresMetadataStore(
            "postgresql://user:pass@prod:5432/metadata",
        )
        dev_store = PostgresMetadataStore(
            "postgresql://user:pass@dev:5432/metadata",
            fallback_stores=[prod_store],
        )
        ```

    Warning:
        SHA256 requires the `pgcrypto` extension. The store will attempt to enable it
        automatically the first time native SHA256 hashing runs. Ensure your database user has
        CREATE EXTENSION privileges or have a DBA pre-create the extension.
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
            ImportError: If Ibis or psycopg2 driver not installed.

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
        # if "schema" in params and params["schema"] is not None:
        # import sys

        # print("*********************", file=sys.stderr)
        # print(params["schema"], file=sys.stderr)
        # params["schema"] = self._ensure_string_identifier(params["schema"])
        # print(params["schema"], file=sys.stderr)
        # print("*********************", file=sys.stderr)

        # import sys

        # print("*********************", file=sys.stderr)
        # print(explicit_params["schema"], file=sys.stderr)
        # explicit_params["schema"] = self._ensure_string_identifier(explicit_params["schema"])
        # print(explicit_params["schema"], file=sys.stderr)
        # print("*********************", file=sys.stderr)
        try:
            import locale

            current_locale = locale.getlocale()
            logger.info(f"Locale diagnosis: current_locale={current_locale}, ")
        except Exception as e:
            logger.warning(f"Could not perform locale diagnosis: {e}")

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

    def open(self) -> None:  # ty: ignore[invalid-method-override]
        """Open connection to PostgreSQL and perform capability checks.

        Raises:
            ImportError: If psycopg2 driver not installed.
            Various database errors: If connection fails.
        """
        super().open()
        # Reset pgcrypto check for the new connection
        self._pgcrypto_extension_checked = False
        import sys

        if not self._has_native_struct_support():
            self._struct_compat_mode = True
            message = (
                "!!! Metaxy WARNING: PostgreSQL backend lacks native STRUCT type support. "
                "Falling back to JSON serialization compatibility mode. !!!"
            )
            print(message, file=sys.stderr)
            logger.warning(
                message.replace("!!! Metaxy WARNING: ", "").replace("!!!", "")
            )
        else:
            self._struct_compat_mode = False
            message = "!!! Metaxy INFO: PostgreSQL backend has native STRUCT type support. Normal operation. !!!"
            print(message, file=sys.stderr)
            logger.info(message.replace("!!! Metaxy INFO: ", "").replace("!!!", ""))

    def _ensure_pgcrypto_ready_for_native_provenance(self) -> None:
        """Enable pgcrypto before running native SHA256 provenance tracking."""
        if self._pgcrypto_extension_checked:
            return

        if (
            not self.enable_pgcrypto
            or self.hash_algorithm != HashAlgorithm.SHA256
            or self._struct_compat_mode
            or not self._supports_native_components()
        ):
            return

        try:
            self._ensure_pgcrypto_extension()
        finally:
            # Avoid repeated attempts (CREATE EXTENSION IF NOT EXISTS is idempotent,
            # but we only need to try once per connection)
            self._pgcrypto_extension_checked = True

    def _ensure_pgcrypto_extension(self) -> None:
        """Ensure pgcrypto extension is enabled for SHA256 support.

        Attempts to create the extension if it doesn't exist. Logs a warning
        if the user lacks privileges rather than failing, as the extension
        might already be enabled.
        """
        try:
            # Use underlying connection to execute DDL (Ibis doesn't expose CREATE EXTENSION)
            # For PostgreSQL backend, conn.con is the psycopg2 connection
            raw_conn = self.conn.con  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
            with raw_conn.cursor() as cursor:  # pyright: ignore[reportAttributeAccessIssue]
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                raw_conn.commit()  # pyright: ignore[reportAttributeAccessIssue]
            logger.debug("pgcrypto extension enabled successfully")
        except _PGCRYPTO_ERROR_TYPES as err:
            self._log_pgcrypto_warning(err)
        except AttributeError as err:
            # Ibis backend may not expose .con or cursor()
            self._log_pgcrypto_warning(err)

    def _get_current_schema(self, raw_conn: Any) -> str:
        """Get the current schema from the search_path."""
        try:
            with raw_conn.cursor() as cursor:  # pyright: ignore[reportAttributeAccessIssue]
                cursor.execute("SHOW search_path")
                result = cursor.fetchone()
            logger.info(
                "Schema detected from search_path: %s",
                result,
                exc_info=True,
            )
        except Exception:
            logger.debug(
                "Could not determine current schema; defaulting to public",
                exc_info=True,
            )
            return "public"

        search_path_raw = result[0] if result else ""
        if isinstance(search_path_raw, bytes):
            try:
                # Try to decode using UTF-8, which is almost always correct for search_path.
                search_path = search_path_raw.decode("utf-8")
            except UnicodeDecodeError:
                # If decoding fails, fall back to a safe default.
                logger.warning(
                    "Could not decode search_path from bytes; defaulting to public."
                )
                return "public"
        else:
            search_path = str(search_path_raw)
        if not search_path:
            return "public"

        for raw_path in str(search_path).split(","):
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
        raw_conn = getattr(self.conn, "con", None)  # pyright: ignore[reportAttributeAccessIssue]
        if raw_conn is None:
            return self.conn.list_tables()

        try:
            schema_to_query = (self.schema or self._get_current_schema(raw_conn)).strip(
                '"'
            )
            if not schema_to_query:
                schema_to_query = "public"

            with raw_conn.cursor() as cursor:  # pyright: ignore[reportAttributeAccessIssue]
                cursor.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = %s",
                    (schema_to_query,),
                )
                tables = [
                    row[0].decode() if isinstance(row[0], bytes) else str(row[0])
                    for row in cursor.fetchall()
                ]
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

    def _resolve_update_native(
        self,
        feature: type[BaseFeature],
        *,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: bool = False,
    ) -> Increment | LazyIncrement:
        """Ensure pgcrypto is available before delegating to base implementation."""
        self._ensure_pgcrypto_ready_for_native_provenance()
        return super()._resolve_update_native(  # ty: ignore[unresolved-attribute]
            feature,
            filters=filters,
            lazy=lazy,
        )

    @staticmethod
    def _ensure_string_identifier(value: str | bytes) -> str:
        """Ensure identifier is a string, not bytes.

        PostgreSQL identifiers must be strings. Hash functions may return bytes
        on some platforms, so we normalize them here.

        Args:
            value: Identifier value (string or bytes)

        Returns:
            String identifier safe for use in SQL
        """
        if isinstance(value, bytes):
            return value.hex()
        return str(value)

    def _get_hash_sql_generators(self) -> dict[HashAlgorithm, Any]:
        """Get hash SQL generators for PostgreSQL."""
        generators = super()._get_hash_sql_generators()  # ty: ignore[unresolved-attribute]

        # Store reference to the static method for use in closure
        ensure_str = self._ensure_string_identifier

        def sha256_generator(table, concat_columns: dict[str, str]) -> str:
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                # Ensure field_key is a string, not bytes
                field_key_str = ensure_str(field_key)
                concat_col_str = ensure_str(concat_col)

                hash_col = f"__hash_{field_key_str}"
                hash_expr = f"ENCODE(DIGEST({concat_col_str}, 'sha256'), 'hex')"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        generators[HashAlgorithm.SHA256] = sha256_generator
        return generators

    def _supports_native_components(self) -> bool:
        """Disable native components when running in struct-compatibility mode."""
        return super()._supports_native_components() and not self._struct_compat_mode  # ty: ignore[unresolved-attribute]
        return super()._supports_native_components() and not self._struct_compat_mode  # ty: ignore[unresolved-attribute]

    def _create_system_tables(self) -> None:
        """
        Atomically create system tables using `CREATE TABLE IF NOT EXISTS`.
        Uses raw SQL to avoid Ibis limitations and ensure proper schema generation.
        """
        from metaxy.metadata_store.system_tables import (
            FEATURE_VERSIONS_KEY,
            FEATURE_VERSIONS_SCHEMA,
            MIGRATION_EVENTS_KEY,
            MIGRATION_EVENTS_SCHEMA,
        )

        def get_postgres_type(
            dtype: PolarsDataType | PolarsDataTypeClass,
        ) -> str:
            """Convert Polars dtype to PostgreSQL type."""

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
            else:
                # Fallback for complex types
                return "VARCHAR"

        def create_table_ddl(table_name: str, schema: SchemaMapping) -> str:
            """Generate CREATE TABLE IF NOT EXISTS DDL from Polars schema."""
            columns = []
            for col_name, dtype in schema.items():
                pg_type = get_postgres_type(dtype)
                # Properly quote column names that might be keywords or contain special chars
                columns.append(f'"{col_name}" {pg_type}')

            columns_sql = ", ".join(columns)
            # Properly quote table name
            return f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql})'

        def execute_ddl(ddl: str):
            """Execute DDL with proper connection handling."""
            raw_conn = self.conn.con  # pyright: ignore[reportAttributeAccessIssue]
            original_autocommit = raw_conn.autocommit
            try:
                raw_conn.autocommit = True
                with raw_conn.cursor() as cursor:  # pyright: ignore[reportAttributeAccessIssue]
                    cursor.execute(ddl)
            finally:
                raw_conn.autocommit = original_autocommit

        # Create feature_versions table
        ddl_fv = create_table_ddl(
            FEATURE_VERSIONS_KEY.table_name,
            cast(SchemaMapping, FEATURE_VERSIONS_SCHEMA),
        )
        execute_ddl(ddl_fv)

        # Create migration_events table
        ddl_me = create_table_ddl(
            MIGRATION_EVENTS_KEY.table_name,
            cast(SchemaMapping, MIGRATION_EVENTS_SCHEMA),
        )
        execute_ddl(ddl_me)

    def _write_metadata_impl(self, feature_key: FeatureKey, df: pl.DataFrame) -> None:
        table_name = feature_key.table_name
        df_to_write = df
        if self._struct_compat_mode and PROVENANCE_BY_FIELD_COL in df.columns:
            df_to_write = self._serialize_provenance_column(df_to_write)

        conn = cast(Any, self.conn)

        if table_name not in self._list_tables_robustly():
            if not self.auto_create_tables:
                from metaxy.metadata_store.exceptions import TableNotFoundError

                raise TableNotFoundError(f"Table '{table_name}' does not exist.")

            df_for_creation = df_to_write
            if self._struct_compat_mode:
                sanitized_schema = {
                    k: (pl.String if isinstance(v, (pl.Struct, pl.List)) else v)
                    for k, v in df_for_creation.schema.items()
                }
                df_for_creation = pl.DataFrame(schema=sanitized_schema)

            for col in df_for_creation.columns:
                if df_for_creation[col].dtype == pl.Null:
                    df_for_creation = df_for_creation.with_columns(
                        pl.col(col).cast(pl.Utf8)
                    )

            conn.create_table(table_name, obj=df_for_creation)
            # After creating the table with a safe schema, insert the data (which is already serialized)
            if len(df_to_write) > 0:
                conn.insert(table_name, obj=df_to_write)
        else:
            conn.insert(table_name, obj=df_to_write)
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
            return f"PostgresMetadataStore(connection_string={self.connection_string})"
        return "PostgresMetadataStore()"

    # def _has_native_struct_support(self) -> bool:
    #     """Detect whether current Ibis/sqlglot stack can compile STRUCT types."""
    #     dialect = getattr(self.conn, "dialect", None)
    #     type_mapping = getattr(dialect, "TYPE_TO_EXPRESSIONS", None)
    #     if not type_mapping:
    #         return False

    #     data_type_cls = getattr(exp, "DataType", None)
    #     type_enum = getattr(data_type_cls, "Type", None) if data_type_cls else None
    #     struct_type = getattr(type_enum, "STRUCT", None) if type_enum else None
    #     if struct_type is None:
    #         return False
    #     return struct_type in type_mapping
    def _has_native_struct_support(self) -> bool:
        """
        Detect whether current Ibis/sqlglot stack can compile STRUCT types.

        NOTE: Native struct support for PostgreSQL in the Ibis/sqlglot stack has proven
        unreliable across different environments. We are temporarily disabling it
        to enforce the more stable JSON serialization path.
        """
        return False

    @staticmethod
    def _serialize_provenance_column(df: pl.DataFrame) -> pl.DataFrame:
        """Convert struct provenance column to canonical JSON for storage."""
        column = df.get_column(PROVENANCE_BY_FIELD_COL)
        serialized = [
            json.dumps(value, sort_keys=True) if value is not None else None
            for value in column.to_list()
        ]
        return df.with_columns(
            pl.Series(name=PROVENANCE_BY_FIELD_COL, values=serialized, dtype=pl.String)
        )

    @staticmethod
    def _deserialize_provenance_column(df: pl.DataFrame) -> pl.DataFrame:
        """Restore struct provenance column from JSON text."""
        if PROVENANCE_BY_FIELD_COL not in df.columns:
            return df
        column = df.get_column(PROVENANCE_BY_FIELD_COL)
        decoded = [
            json.loads(value) if value is not None else None
            for value in column.to_list()
        ]
        return df.with_columns(pl.Series(name=PROVENANCE_BY_FIELD_COL, values=decoded))
