from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import ibis
import ibis.expr.types as ibis_types
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
from metaxy.versioning.flat_engine import IbisFlatVersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from narwhals.typing import Frame

    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.types import FeatureKey

# Define FrameT for use in type annotations (runtime and type checking)
FrameT = nw.DataFrame[Any] | nw.LazyFrame[Any]


_PGCRYPTO_ERROR_TYPES: tuple[type[Exception], ...] = (_PsycopgError,)

logger = logging.getLogger(__name__)
SchemaMapping = Mapping[str, PolarsDataType | PolarsDataTypeClass]
SchemaMapping = Mapping[str, PolarsDataType | PolarsDataTypeClass]


class PostgresVersioningEngine(IbisFlatVersioningEngine):
    """PostgreSQL-specific versioning engine that uses any_value() instead of arbitrary().

    PostgreSQL 16+ ships any_value() but not first(), while Ibis implements
    arbitrary() using first(). We override aggregation helpers to call the
    native any_value() aggregate directly and rely on a compatibility first()
    aggregate installed by the store for other Ibis operations that still
    emit FIRST().
    """

    @staticmethod
    def aggregate_with_string_concat(
        df: FrameT,
        group_by_columns: list[str],
        concat_column: str,
        concat_separator: str,
        exclude_columns: list[str],
    ) -> FrameT:
        """Aggregate DataFrame using any_value() instead of arbitrary()."""
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis_types.Table = cast(ibis_types.Table, df.to_native())

        agg_exprs = {}
        agg_exprs[concat_column] = ibis_table[concat_column].group_concat(
            concat_separator
        )

        all_columns = set(ibis_table.columns)
        columns_to_aggregate = (
            all_columns - set(group_by_columns) - {concat_column} - set(exclude_columns)
        )

        def make_any_value(expr: ibis.Expr):
            dtype = expr.type()

            @ibis.udf.agg.builtin(signature=((dtype,), dtype))
            def any_value(x): ...  # noqa: ARG001

            return any_value(expr)

        # Use any_value() for all other columns
        for col in columns_to_aggregate:
            agg_exprs[col] = make_any_value(ibis_table[col])

        result_table = ibis_table.group_by(group_by_columns).aggregate(**agg_exprs)
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_column: str,
    ) -> FrameT:
        """Keep only the latest row per group using argmax to stay row-aligned."""
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )

        columns = df.collect_schema().names()
        if timestamp_column not in columns:
            raise ValueError(
                f"Timestamp column '{timestamp_column}' not found in DataFrame. "
                f"Available columns: {columns}"
            )

        ibis_table: ibis_types.Table = cast(ibis_types.Table, df.to_native())

        all_columns = set(ibis_table.columns)
        non_group_columns = all_columns - set(group_columns)

        agg_exprs = {
            col: ibis_table[col].argmax(ibis_table[timestamp_column])
            for col in non_group_columns
        }

        result_table = ibis_table.group_by(group_columns).aggregate(**agg_exprs)
        return cast(FrameT, nw.from_native(result_table))


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
        """Initialize PostgreSQL metadata store with JSONB storage.

        Args:
            connection_string: Connection string (postgresql://user:pass@host:port/database)
            host: Server host
            port: Server port (default: 5432)
            user: Database user
            password: Database password
            database: Database name
            schema: Target schema (default: search_path)
            connection_params: Additional Ibis connection parameters
            fallback_stores: Read-only fallback stores for branch deployments
            enable_pgcrypto: Auto-enable pgcrypto for SHA256 (default: False)
            **kwargs: Passed to IbisMetadataStore (hash_algorithm, auto_create_tables, etc.)
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
        # Use PostgreSQL-specific versioning engine that prefers any_value()
        self.versioning_engine_cls = PostgresVersioningEngine

        supported_algorithms = {HashAlgorithm.MD5, HashAlgorithm.SHA256}
        if self.hash_algorithm not in supported_algorithms:
            raise HashAlgorithmNotSupportedError(
                f"PostgresMetadataStore supports only MD5 and SHA256 hash algorithms. "
                f"Requested: {self.hash_algorithm}"
            )

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Default to MD5 (built-in, no extensions needed)."""
        return HashAlgorithm.MD5

    def _create_hash_functions(self):
        """Create hash functions (MD5 built-in, SHA256 via pgcrypto)."""

        hash_functions = {}

        @ibis.udf.scalar.builtin
        def MD5(_x: str) -> str:  # noqa: N802  # ty: ignore[invalid-return-type]
            """PostgreSQL MD5() function."""
            ...

        def md5_hash(col_expr):
            return MD5(col_expr.cast(str))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        @ibis.udf.scalar.builtin
        def digest(_value: str, _algorithm: str) -> bytes:  # ty: ignore[invalid-return-type]
            """pgcrypto digest() function."""
            ...

        @ibis.udf.scalar.builtin
        def encode(_value: bytes, _fmt: str) -> str:  # ty: ignore[invalid-return-type]
            """PostgreSQL encode() function."""
            ...

        @ibis.udf.scalar.builtin
        def lower(_value: str) -> str:  # ty: ignore[invalid-return-type]
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
        """Open connection and reset pgcrypto extension check."""
        with super().open(mode):
            self._pgcrypto_extension_checked = False
            self._ensure_first_aggregate()
            yield self

    def _ensure_first_aggregate(self) -> None:
        """Create lightweight first() aggregate for Ibis compatibility.

        Some Ibis operations still compile to FIRST(), which PostgreSQL lacks.
        We create a simple compatibility aggregate that mirrors any_value()
        semantics (returns the first non-null value).
        """
        try:
            raw_conn = cast(Any, self.conn).con
            with raw_conn.cursor() as cursor:
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION _first_sfunc(anyelement, anyelement)
                    RETURNS anyelement
                    LANGUAGE SQL IMMUTABLE STRICT AS $$SELECT $1$$;
                """)
                cursor.execute("""
                    CREATE OR REPLACE AGGREGATE first(anyelement) (
                        SFUNC = _first_sfunc,
                        STYPE = anyelement
                    );
                """)
            raw_conn.commit()
            logger.debug("first() aggregate function created successfully")
        except _PGCRYPTO_ERROR_TYPES as err:
            logger.warning(
                "Could not create first() aggregate function: %s. "
                "This is needed for provenance tracking with aggregation/expansion lineage.",
                err,
            )
        except AttributeError as err:
            logger.warning(
                "Could not create first() aggregate function: %s. "
                "This is needed for provenance tracking with aggregation/expansion lineage.",
                err,
            )

    def _ensure_pgcrypto_ready_for_native_provenance(self) -> None:
        """Enable pgcrypto for SHA256 hashing if needed."""
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
        """Enable pgcrypto extension. Logs warning if insufficient privileges."""
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
        """Ensure pgcrypto ready, then create versioning engine."""
        # Ensure pgcrypto is ready before native provenance tracking
        self._ensure_pgcrypto_ready_for_native_provenance()

        # Use parent implementation from IbisJsonCompatStore
        with super()._create_versioning_engine(plan) as engine:
            yield engine

    def _get_json_unpack_exprs(
        self, json_column: str, field_names: list[str]
    ) -> dict[str, Any]:
        """Unpack JSONB column to flattened columns via jsonb_extract_path_text."""

        @ibis.udf.scalar.builtin
        def jsonb_extract_path_text(_data, *paths) -> str: ...  # ty: ignore[invalid-return-type]

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
        """Pack flattened columns to JSONB via jsonb_object."""
        import ibis.expr.datatypes as dt

        @ibis.udf.scalar.builtin(output_type=dt.jsonb)
        def jsonb_object(_keys: list[str], _values: list[str]) -> str: ...  # ty: ignore[invalid-return-type]

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

    def transform_before_write(
        self, df: Frame, feature_key: FeatureKey, table_name: str
    ) -> Frame:
        """Transform DataFrame before writing to PostgreSQL.

        Handles:

        - `metaxy_materialization_id`: Cast to String (required by PostgreSQL)
        - Null-typed columns: Cast to String for Polars inputs (PostgreSQL rejects NULL-typed columns)

        Args:
            df: Input DataFrame
            feature_key: Feature key being written (unused for PostgreSQL)
            table_name: Target table name (unused for PostgreSQL)

        Returns:
            Transformed DataFrame ready for PostgreSQL insert
        """
        del feature_key, table_name  # Unused but required for signature compatibility
        from metaxy.models.constants import METAXY_MATERIALIZATION_ID

        # Cast materialization_id to string (required for PostgreSQL)
        if METAXY_MATERIALIZATION_ID in df.columns:
            df = df.with_columns(nw.col(METAXY_MATERIALIZATION_ID).cast(nw.String))

        # PostgreSQL rejects NULL-typed columns; cast Null columns to String for Polars inputs
        if df.implementation == nw.Implementation.POLARS:
            null_cols = [col for col, dtype in df.schema.items() if dtype == pl.Null]
            if null_cols:
                df = df.with_columns(
                    [nw.col(col).cast(nw.String).alias(col) for col in null_cols]
                )

        return df

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop metadata table for feature."""
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

        # Use IbisJsonCompatStore to read and unpack JSON columns.
        feature_key = self._resolve_feature_key(feature)
        lazy_frame = super().read_metadata_in_store(
            feature,
            feature_version=feature_version,
            filters=filters,
            columns=columns,
            **kwargs,
        )
        if lazy_frame is None:
            return None

        if self._is_system_table(feature_key):
            return lazy_frame

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

    def _list_features_local(self) -> list[FeatureKey]:
        """List non-system feature tables for display and diagnostics."""
        table_names = self.conn.list_tables()
        features: list[FeatureKey] = []
        for table_name in table_names:
            if table_name.startswith("ibis_"):
                continue
            feature_key = self._table_name_to_feature_key(table_name)  # ty: ignore[unresolved-attribute]
            if not self._is_system_table(feature_key):
                features.append(feature_key)
        return features

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
            details.append(f"features={len(self._list_features_local())}")
        detail_str = ", ".join(details)
        if detail_str:
            return f"PostgresMetadataStore({detail_str})"
        if self.connection_string:
            sanitized = sanitize_uri(self.connection_string)
            return f"PostgresMetadataStore(connection_string={sanitized})"
        return "PostgresMetadataStore()"
