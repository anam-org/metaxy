"""Ibis-based metadata store for SQL databases.

Supports any SQL database that Ibis supports:
- DuckDB, PostgreSQL, MySQL (local/embedded)
- ClickHouse, Snowflake, BigQuery (cloud analytical)
- And 20+ other backends
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import ibis
import ibis.common.exceptions as ibis_exceptions
import narwhals as nw
from narwhals.typing import Frame, IntoFrame
from pydantic import Field
from typing_extensions import Self

from metaxy._utils import collect_to_polars
from metaxy.metadata_store._sql_utils import (
    predicate_from_select_sql,
    validate_identifier,
)
from metaxy.metadata_store.base import (
    MetadataStore,
    MetadataStoreConfig,
    VersioningEngineOptions,
)
from metaxy.metadata_store.exceptions import (
    HashAlgorithmNotSupportedError,
    StoreNotOpenError,
    TableNotFoundError,
)
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import sanitize_uri
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.ibis import IbisVersioningEngine
from metaxy.versioning.types import HashAlgorithm


class IbisMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for IbisMetadataStore.

    Example:
        ```python
        config = IbisMetadataStoreConfig(
            connection_string="postgresql://user:pass@host:5432/db",
            table_prefix="prod_",
        )

        # Note: IbisMetadataStore is abstract, use a concrete implementation
        ```
    """

    connection_string: str | None = Field(
        default=None,
        description="Ibis connection string (e.g., 'clickhouse://host:9000/db').",
    )

    backend: str | None = Field(
        default=None,
        description="Ibis backend name (e.g., 'clickhouse', 'postgres', 'duckdb').",
        json_schema_extra={"mkdocs_metaxy_hide": True},
    )

    connection_params: dict[str, Any] | None = Field(
        default=None,
        description="Backend-specific connection parameters.",
    )

    table_prefix: str | None = Field(
        default=None,
        description="Optional prefix for all table names.",
    )

    auto_create_tables: bool | None = Field(
        default=None,
        description="If True, create tables on open. For development/testing only.",
    )


class IbisMetadataStore(MetadataStore, ABC):
    """
    Generic SQL metadata store using Ibis.

    Supports any Ibis backend that supports struct types, such as: DuckDB, PostgreSQL, ClickHouse, and others.

    Warning:
        Backends without native struct support (e.g., SQLite) are NOT supported.

    Storage layout:
    - Each feature gets its own table: {feature}__{key}
    - System tables: metaxy__system__feature_versions, metaxy__system__migrations
    - Uses Ibis for cross-database compatibility

    Note: Uses MD5 hash by default for cross-database compatibility.
    DuckDBMetadataStore overrides this with dynamic algorithm detection.
    For other backends, override the calculator instance variable with backend-specific implementations.

    Example:
        ```py
        # ClickHouse
        store = IbisMetadataStore("clickhouse://user:pass@host:9000/db")

        # PostgreSQL
        store = IbisMetadataStore("postgresql://user:pass@host:5432/db")

        # DuckDB (use DuckDBMetadataStore instead for better hash support)
        store = IbisMetadataStore("duckdb:///metadata.db")

        with store:
            store.write_metadata(MyFeature, df)
        ```
    """

    versioning_engine_cls = IbisVersioningEngine

    def __init__(
        self,
        versioning_engine: VersioningEngineOptions = "auto",
        connection_string: str | None = None,
        *,
        backend: str | None = None,
        connection_params: dict[str, Any] | None = None,
        table_prefix: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Ibis metadata store.

        Args:
            versioning_engine: Which versioning engine to use.
                - "auto": Prefer the store's native engine, fall back to Polars if needed
                - "native": Always use the store's native engine, raise `VersioningEngineMismatchError`
                    if provided dataframes are incompatible
                - "polars": Always use the Polars engine
            connection_string: Ibis connection string (e.g., "clickhouse://host:9000/db")
                If provided, backend and connection_params are ignored.
            backend: Ibis backend name (e.g., "clickhouse", "postgres", "duckdb")
                Used with connection_params for more control.
            connection_params: Backend-specific connection parameters
                e.g., {"host": "localhost", "port": 9000, "database": "default"}
            table_prefix: Optional prefix applied to all feature and system table names.
                Useful for logically separating environments (e.g., "prod_"). Must form a valid SQL
                identifier when combined with the generated table name.
            **kwargs: Passed to MetadataStore.__init__ (e.g., fallback_stores, hash_algorithm)

        Raises:
            ValueError: If neither connection_string nor backend is provided
            ImportError: If Ibis or required backend driver not installed

        Example:
            ```py
            # Using connection string
            store = IbisMetadataStore("clickhouse://user:pass@host:9000/db")

            # Using backend + params
            store = IbisMetadataStore(
                backend="clickhouse",
                connection_params={"host": "localhost", "port": 9000}
                )
            ```
        """
        self.connection_string = connection_string
        self.backend = backend
        self.connection_params = connection_params or {}
        self._conn: ibis.BaseBackend | None = None
        self._table_prefix = table_prefix or ""

        super().__init__(
            **kwargs,
            versioning_engine=versioning_engine,
        )

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        feature_key = self._resolve_feature_key(feature)
        table_name = self.get_table_name(feature_key)
        return table_name in self.conn.list_tables()

    def get_table_name(
        self,
        key: FeatureKey,
    ) -> str:
        """Generate the storage table name for a feature or system table.

        Applies the configured table_prefix (if any) to the feature key's table name.
        Subclasses can override this method to implement custom naming logic.

        Args:
            key: Feature key to convert to storage table name.

        Returns:
            Storage table name with optional prefix applied.
        """
        base_name = key.table_name

        return f"{self._table_prefix}{base_name}" if self._table_prefix else base_name

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for Ibis stores.

        Uses MD5 as it's universally supported across SQL databases.
        Subclasses like DuckDBMetadataStore can override for better algorithms.
        """
        return HashAlgorithm.MD5

    @contextmanager
    def _create_versioning_engine(
        self, plan: FeaturePlan
    ) -> Iterator[IbisVersioningEngine]:
        """Create provenance engine for Ibis backend as a context manager.

        Args:
            plan: Feature plan for the feature we're tracking provenance for

        Yields:
            IbisVersioningEngine with backend-specific hash functions.

        Note:
            Base implementation only supports MD5 (universally available).
            Subclasses can override _create_hash_functions() for backend-specific hashes.
        """
        if self._conn is None:
            raise RuntimeError(
                "Cannot create provenance engine: store is not open. "
                "Ensure store is used as context manager."
            )

        # Create hash functions for Ibis expressions
        hash_functions = self._create_hash_functions()

        # Create engine using the configured class (allows subclass override)
        engine = self.versioning_engine_cls(
            plan=plan,
            hash_functions=hash_functions,
        )

        try:
            yield engine
        finally:
            # No cleanup needed for Ibis engine
            pass

    @abstractmethod
    def _create_hash_functions(self):
        """Create hash functions for Ibis expressions.

        Base implementation returns empty dict. Subclasses must override
        to provide backend-specific hash function implementations.

        Returns:
            Dictionary mapping HashAlgorithm to Ibis expression functions
        """
        return {}

    def _validate_hash_algorithm_support(self) -> None:
        """Validate that the configured hash algorithm is supported by Ibis backend.

        Raises:
            ValueError: If hash algorithm is not supported
        """
        # Create hash functions to check what's supported
        hash_functions = self._create_hash_functions()

        if self.hash_algorithm not in hash_functions:
            supported = [algo.value for algo in hash_functions.keys()]
            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm '{self.hash_algorithm.value}' not supported. "
                f"Supported algorithms: {', '.join(supported)}"
            )

    @property
    def ibis_conn(self) -> "ibis.BaseBackend":
        """Get Ibis backend connection.

        Returns:
            Active Ibis backend connection

        Raises:
            StoreNotOpenError: If store is not open
        """
        if self._conn is None:
            raise StoreNotOpenError(
                "Ibis connection is not open. Store must be used as a context manager."
            )
        return self._conn

    @property
    def conn(self) -> "ibis.BaseBackend":
        """Get connection (alias for ibis_conn for consistency).

        Returns:
            Active Ibis backend connection

        Raises:
            StoreNotOpenError: If store is not open
        """
        return self.ibis_conn

    def _execute_raw_sql(self, sql: str) -> Any:
        """Run raw SQL on the underlying backend."""
        backend = self.conn
        return cast(Any, backend).raw_sql(sql)

    @staticmethod
    def _rowcount_or_default(result: Any, default: int) -> int:
        try:
            rowcount = int(result.rowcount)
            return rowcount if rowcount >= 0 else default
        except Exception:
            return default

    def _execute_count_sql(self, sql: str) -> int:
        """Execute a COUNT(*) query and return the scalar result."""
        result = self._execute_raw_sql(sql)

        # DB-API cursor style
        try:
            row = result.fetchone()
            if row is None:
                return 0
            if isinstance(row, (list, tuple)):
                return int(row[0])
            return int(row)
        except Exception:
            pass

        # Pandas DataFrame result (common ibis default)
        try:
            import pandas as pd

            if isinstance(result, pd.DataFrame):
                return int(result.iloc[0, 0])
        except Exception:
            pass

        # PyArrow Table result
        try:
            import pyarrow as pa

            if isinstance(result, pa.Table):
                return int(result.column(0)[0].as_py())
        except Exception:
            pass

        return 0

    def _supports_returning(self) -> bool:
        """Check whether backend likely supports SQL RETURNING for DML."""
        try:
            backend_name = self.conn.name
        except Exception:
            backend_name = ""
        return (backend_name or "").lower() in {"duckdb", "postgres", "postgresql"}

    def _needs_precount_delete(self) -> bool:
        """Backends that don't reliably report rowcount for DELETE."""
        try:
            backend_name = self.conn.name
        except Exception:
            backend_name = ""
        return (backend_name or "").lower() in {"duckdb", "clickhouse"}

    def _get_sql_dialect(self, backend_name: str) -> str:
        """Map Ibis backend name to SQL dialect string for value formatting.

        Args:
            backend_name: Ibis backend name (e.g., 'duckdb', 'postgres', 'clickhouse')

        Returns:
            SQL dialect string ('standard', 'clickhouse', 'delta', 'lancedb')
        """
        backend_lower = (backend_name or "").lower()
        if "clickhouse" in backend_lower:
            return "clickhouse"
        # Most SQL databases use standard SQL literals
        return "standard"

    def _build_in_predicate(self, columns: list[str], subquery_sql: str) -> str:
        """Build tuple/subquery IN predicate without touching aliases."""
        if len(columns) == 1:
            return f"{columns[0]} IN ({subquery_sql})"
        joined = ", ".join(columns)
        return f"({joined}) IN ({subquery_sql})"

    @contextmanager
    def open(self, mode: AccessMode = "read") -> Iterator[Self]:
        """Open connection to database via Ibis.

        Subclasses should override this to add backend-specific initialization
        (e.g., loading extensions) and must call this method via super().open(mode).

        Args:
            mode: Access mode. Subclasses may use this to set backend-specific connection
                parameters (e.g., `read_only` for DuckDB).

        Yields:
            Self: The store instance with connection open
        """
        # Increment context depth to support nested contexts
        self._context_depth += 1

        try:
            # Only perform actual open on first entry
            if self._context_depth == 1:
                self._access_mode = mode
                # Setup: Connect to database
                if self.connection_string:
                    # Use connection string
                    self._conn = ibis.connect(self.connection_string)
                else:
                    # Use backend + params via ibis.connect with URL scheme
                    assert self.backend is not None, (
                        "backend must be set if connection_string is None"
                    )
                    resource = f"{self.backend}://"
                    params = dict(self.connection_params)
                    if "database" in params:
                        resource = f"{resource}{params.pop('database')}"
                    self._conn = ibis.connect(resource, **params)

                # Mark store as open and validate
                self._is_open = True
                self._validate_after_open()

            yield self
        finally:
            # Decrement context depth
            self._context_depth -= 1

            # Only perform actual close on last exit
            if self._context_depth == 0:
                # Teardown: Close connection
                if self._conn is not None:
                    # Ibis connections may not have explicit close method
                    # but setting to None releases resources
                    self._conn = None
                self._is_open = False
                self._access_mode = None

    @property
    def sqlalchemy_url(self) -> str:
        """Get SQLAlchemy-compatible connection URL for tools like Alembic.

        Returns the connection string if available. If the store was initialized
        with backend + connection_params instead of a connection string, raises
        an error since constructing a proper URL is backend-specific.

        Returns:
            SQLAlchemy-compatible URL string

        Raises:
            ValueError: If connection_string is not available

        Example:
            ```python
            store = IbisMetadataStore("postgresql://user:pass@host:5432/db")
            print(store.sqlalchemy_url)  # postgresql://user:pass@host:5432/db
            ```
        """
        if self.connection_string:
            return self.connection_string

        raise ValueError(
            "SQLAlchemy URL not available. Store was initialized with backend + connection_params "
            "instead of a connection string. To use Alembic, initialize with a connection string: "
            f"IbisMetadataStore('postgresql://user:pass@host:5432/db') instead of "
            f"IbisMetadataStore(backend='{self.backend}', connection_params={{...}})"
        )

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """
        Internal write implementation using Ibis.

        Args:
            feature_key: Feature key to write to
            df: DataFrame with metadata (already validated)
            **kwargs: Backend-specific parameters (currently unused)

        Raises:
            TableNotFoundError: If table doesn't exist and auto_create_tables is False
        """
        table_name = self.get_table_name(feature_key)

        # Apply backend-specific transformations before writing
        df = self.transform_before_write(df, feature_key, table_name)

        if df.implementation == nw.Implementation.IBIS:
            df_to_insert = df.to_native()  # Ibis expression
        else:
            df_to_insert = collect_to_polars(df)  # Polars DataFrame

        try:
            self.conn.insert(table_name, obj=df_to_insert)  # ty: ignore[unresolved-attribute]
        except Exception as e:
            if not isinstance(e, ibis_exceptions.TableNotFound):
                raise
            if self.auto_create_tables:
                # Warn about auto-create (first time only)
                if self._should_warn_auto_create_tables:
                    warnings.warn(
                        f"AUTO_CREATE_TABLES is enabled - automatically creating table '{table_name}'. "
                        "Do not use in production! "
                        "Use proper database migration tools like Alembic for production deployments.",
                        UserWarning,
                        stacklevel=4,
                    )

                # Note: create_table(table_name, obj=df) both creates the table AND inserts the data
                # No separate insert needed - the data from df is already written
                self.conn.create_table(table_name, obj=df_to_insert)
            else:
                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist for feature {feature_key.to_string()}. "
                    f"Enable auto_create_tables=True to automatically create tables, "
                    f"or use proper database migration tools like Alembic to create the table first."
                ) from e

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop the table for a feature.

        Args:
            feature_key: Feature key to drop metadata for
        """
        table_name = self.get_table_name(feature_key)

        # Check if table exists
        if table_name in self.conn.list_tables():
            self.conn.drop_table(table_name)

    # Temp table infrastructure for semi-join optimization

    def _supports_temp_tables(self) -> bool:
        """Ibis SQL backends support temporary tables."""
        return True

    def _create_temp_table(
        self,
        temp_table_name: str,
        frame: IntoFrame,
        columns: list[str],
    ) -> None:
        """Create a temporary table with filter data for semi-join.

        Args:
            temp_table_name: Name for the temporary table
            frame: Frame containing filter data
            columns: Columns to include in temp table
        """
        # Convert to native format for Ibis
        frame_nw = nw.from_native(frame)
        if not isinstance(frame_nw, (nw.DataFrame, nw.LazyFrame)):
            raise TypeError("Frame must be a DataFrame or LazyFrame, not a Series")

        # Collect schema to avoid LazyFrame column access warnings
        _ = frame_nw.collect_schema()
        selected = frame_nw.select(columns).unique()

        if selected.implementation == nw.Implementation.IBIS:
            df_to_insert = selected.to_native()
        else:
            df_to_insert = collect_to_polars(selected)

        # Create temp table with the filtered columns
        # Note: Ibis create_table with temp=True creates session-scoped temp tables
        self.conn.create_table(temp_table_name, obj=df_to_insert, temp=True)

    def _drop_temp_table(self, temp_table_name: str) -> None:
        """Drop a temporary table created by _create_temp_table.

        Args:
            temp_table_name: Name of temp table to drop
        """
        try:
            if temp_table_name in self.conn.list_tables():
                self.conn.drop_table(temp_table_name)
        except Exception:
            # Temp tables are often automatically cleaned up on session end
            # Don't fail if drop fails
            pass

    def _delete_metadata_with_temp_table(
        self,
        feature_key: FeatureKey,
        temp_table_name: str,
        join_columns: list[str],
    ) -> int:
        """Backend-specific hard delete using temp table semi-join.

        Uses SQL IN subquery for efficient deletion without client-side materialization.

        Args:
            feature_key: Feature to delete from
            temp_table_name: Name of temporary table containing filter values
            join_columns: Columns to use for semi-join

        Returns:
            Number of rows deleted
        """
        table_name = self.get_table_name(feature_key)
        validate_identifier(table_name, context="table name")

        for column in join_columns:
            validate_identifier(column, context="column name")

        if table_name not in self.conn.list_tables():
            return 0

        temp_select = self.conn.table(temp_table_name).select(join_columns)
        predicate = self._build_in_predicate(join_columns, ibis.to_sql(temp_select))

        rows_before = None
        if self._needs_precount_delete():
            count_sql = f"SELECT COUNT(*) FROM {table_name} WHERE {predicate}"
            rows_before = self._execute_count_sql(count_sql)
            if rows_before == 0:
                return 0

        delete_sql = f"DELETE FROM {table_name} WHERE {predicate}"
        result = self._execute_raw_sql(delete_sql)
        rows_deleted = self._rowcount_or_default(result, default=-1)

        if rows_deleted >= 0:
            return int(rows_deleted)

        return int(rows_before or 0)

        return int(rows_deleted)

    def _delete_metadata_impl(
        self,
        feature_key: FeatureKey,
        filter_expr: nw.Expr,
    ) -> int:
        """Backend-specific hard delete implementation for Ibis/SQL stores.

        Uses SQL DELETE to physically remove records matching the filter.

        Args:
            feature_key: Feature to delete from
            filter_expr: Narwhals expression to filter records

        Returns:
            Number of rows deleted
        """
        table_name = self.get_table_name(feature_key)
        validate_identifier(table_name, context="table name")

        # Check if table exists
        if table_name not in self.conn.list_tables():
            return 0

        table = self.conn.table(table_name)
        nw_table = nw.from_native(table, eager_only=False)

        # Build predicate SQL directly from the filter expression without materializing rows
        filtered_native = nw_table.filter(filter_expr).to_native()  # type: ignore[arg-type]
        predicate = predicate_from_select_sql(ibis.to_sql(filtered_native))

        rows_before = None
        if self._needs_precount_delete():
            count_sql = f"SELECT COUNT(*) FROM {table_name} WHERE {predicate}"
            rows_before = self._execute_count_sql(count_sql)
            if rows_before == 0:
                return 0

        delete_sql = f"DELETE FROM {table_name} WHERE {predicate}"
        result = self._execute_raw_sql(delete_sql)
        rows_deleted = self._rowcount_or_default(result, default=-1)

        if rows_deleted >= 0:
            return int(rows_deleted)

        return int(rows_before or 0)

    def _mutate_metadata_impl(
        self,
        feature_key: FeatureKey,
        filter_expr: nw.Expr,
        updates: dict[str, Any],
    ) -> int:
        """Disable in-place UPDATE for SQL backends; use append-only fallback."""
        raise NotImplementedError(
            "Mutations are append-only for ibis SQL backends; in-place UPDATE disabled."
        )

    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """
        Read metadata from this store only (no fallback).

        Args:
            feature: Feature to read
            feature_version: Filter by specific feature_version (applied as SQL WHERE clause)
            filters: List of Narwhals filter expressions (converted to SQL WHERE clauses)
            columns: Optional list of columns to select
            **kwargs: Backend-specific parameters (currently unused)

        Returns:
            Narwhals LazyFrame with metadata, or None if not found
        """
        feature_key = self._resolve_feature_key(feature)
        table_name = self.get_table_name(feature_key)

        # Check if table exists
        existing_tables = self.conn.list_tables()
        if table_name not in existing_tables:
            return None

        # Get Ibis table reference
        table = self.conn.table(table_name)

        # Apply backend-specific transformations (e.g., cast JSON columns for ClickHouse)
        table = self.transform_after_read(table, feature_key)

        # Wrap Ibis table with Narwhals (stays lazy in SQL)
        nw_lazy = cast(
            nw.LazyFrame[Any], cast(object, nw.from_native(table, eager_only=False))
        )

        # Apply feature_version filter (stays in SQL via Narwhals)
        if feature_version is not None:
            nw_lazy = nw_lazy.filter(
                nw.col("metaxy_feature_version") == feature_version
            )

        # Apply generic Narwhals filters (stays in SQL)
        if filters is not None:
            for filter_expr in filters:
                nw_lazy = nw_lazy.filter(filter_expr)

        # Select columns (stays in SQL)
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        # Return Narwhals LazyFrame wrapping Ibis table (stays lazy in SQL)
        return nw_lazy

    def transform_after_read(
        self, table: "ibis.Table", feature_key: "FeatureKey"
    ) -> "ibis.Table":
        """Transform Ibis table before wrapping with Narwhals.

        Override in subclasses to apply backend-specific transformations.

        !!! example
            ClickHouse needs to cast JSON columns to Struct for
            PyArrow compatibility.

        Args:
            table: Ibis table reference
            feature_key: The feature key being read (use to get field names)

        Returns:
            Transformed Ibis table (default: unchanged)
        """
        return table

    def transform_before_write(
        self, df: Frame, feature_key: "FeatureKey", table_name: str
    ) -> Frame:
        """Transform DataFrame before writing to the store.

        Override in subclasses to apply backend-specific transformations.

        !!! example
            ClickHouse needs to convert Polars Struct columns to
            Map-compatible format when the table has Map columns.

        Args:
            df: Narwhals DataFrame to be written
            feature_key: The feature key being written to
            table_name: The target table name

        Returns:
            Transformed DataFrame (default: unchanged)
        """
        return df

    def _can_compute_native(self) -> bool:
        """
        Ibis backends support native field provenance calculations (Narwhals-based).

        Returns:
            True (use Narwhals components with Ibis-backed tables)

        Note: All Ibis stores now use Narwhals-based components (NarwhalsJoiner,
        PolarsProvenanceByFieldCalculator, NarwhalsDiffResolver) which work efficiently
        with Ibis-backed tables.
        """
        return True

    def display(self) -> str:
        """Display string for this store."""
        backend_info = self.connection_string or f"{self.backend}"
        # Sanitize connection strings that may contain credentials
        sanitized_info = sanitize_uri(backend_info)
        return f"{self.__class__.__name__}(backend={sanitized_info})"

    def _get_store_metadata_impl(
        self, feature_key: CoercibleToFeatureKey
    ) -> dict[str, Any]:
        """Return store metadata including table name.

        Args:
            feature_key: Feature key to get metadata for.

        Returns:
            Dictionary with `table_name` key.
        """
        resolved_key = self._resolve_feature_key(feature_key)
        return {"table_name": self.get_table_name(resolved_key)}

    @classmethod
    def config_model(cls) -> type[IbisMetadataStoreConfig]:
        return IbisMetadataStoreConfig
