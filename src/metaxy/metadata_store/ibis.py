"""Ibis-based metadata store for SQL databases.

Supports any SQL database that Ibis supports:
- DuckDB, PostgreSQL, MySQL (local/embedded)
- ClickHouse, Snowflake, BigQuery (cloud analytical)
- And 20+ other backends
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import narwhals as nw
from narwhals.typing import Frame
from pydantic import Field
from typing_extensions import Self

from metaxy._decorators import public
from metaxy.metadata_store.base import (
    MetadataStore,
    MetadataStoreConfig,
    VersioningEngineOptions,
)
from metaxy.metadata_store.exceptions import (
    FeatureNotFoundError,
    HashAlgorithmNotSupportedError,
    StoreNotOpenError,
    TableNotFoundError,
)
from metaxy.metadata_store.types import AccessMode
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.ibis import IbisVersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    import ibis
    import ibis.expr.types
    from ibis.backends.sql import SQLBackend


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


@public
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
        <!-- skip next -->
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
            <!-- skip next -->
            ```py
            # Using connection string
            store = IbisMetadataStore("clickhouse://user:pass@host:9000/db")

            # Using backend + params
            store = IbisMetadataStore(backend="clickhouse", connection_params={"host": "localhost", "port": 9000})
            ```
        """
        from ibis.backends.sql import SQLBackend

        self.connection_string = connection_string
        self.backend = backend
        self.connection_params = connection_params or {}
        self._conn: SQLBackend | None = None
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
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[IbisVersioningEngine]:
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
                "Cannot create provenance engine: store is not open. Ensure store is used as context manager."
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
    def conn(self) -> "SQLBackend":
        """Get Ibis backend connection.

        Returns:
            Active Ibis backend connection

        Raises:
            StoreNotOpenError: If store is not open
        """

        if self._conn is None:
            raise StoreNotOpenError("Ibis connection is not open. Store must be used as a context manager.")
        else:
            return self._conn  # ty: ignore[invalid-return-type]

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
        import ibis

        # Increment context depth to support nested contexts
        self._context_depth += 1

        try:
            # Only perform actual open on first entry
            if self._context_depth == 1:
                # Setup: Connect to database
                if self.connection_string:
                    # Use connection string
                    self._conn = ibis.connect(self.connection_string)
                else:
                    # Use backend + params
                    # Get backend-specific connect function
                    assert self.backend is not None, "backend must be set if connection_string is None"
                    backend_module = getattr(ibis, self.backend)
                    self._conn = backend_module.connect(**self.connection_params)

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
            <!-- skip next -->
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
            from metaxy._utils import collect_to_polars

            df_to_insert = collect_to_polars(df)  # Polars DataFrame

        try:
            self.conn.insert(table_name, obj=df_to_insert)  # ty: ignore[invalid-argument-type]
        except Exception as e:
            import ibis.common.exceptions

            if not isinstance(e, ibis.common.exceptions.TableNotFound):
                raise
            if self.auto_create_tables:
                # Warn about auto-create (first time only)
                if self._should_warn_auto_create_tables:
                    import warnings

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

    def _delete_metadata_impl(
        self,
        feature_key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        current_only: bool,  # noqa: ARG002 - version filtering handled by base class
    ) -> None:
        """Backend-specific hard delete implementation for SQL databases.

        Translates Narwhals filter expression to Ibis and executes DELETE in database.

        Args:
            feature_key: Feature to delete from
            filters: Narwhals expressions to filter records
            current_only: Not used here - version filtering handled by base class
        """
        from metaxy.metadata_store.utils import (
            _extract_where_expression,
            _strip_table_qualifiers,
        )

        table_name = self.get_table_name(feature_key)
        filter_list = list(filters or [])

        # Handle empty filters - truncate entire table
        if not filter_list:
            if table_name not in self.conn.list_tables():
                raise TableNotFoundError(f"Table '{table_name}' does not exist for feature {feature_key.to_string()}.")
            self.conn.truncate_table(table_name)  # ty: ignore[unresolved-attribute]
            return

        # Read and filter using store's lazy path to build WHERE clause
        filtered = self.read_metadata_in_store(feature_key, filters=filter_list)
        if filtered is None:
            raise FeatureNotFoundError(f"Feature {feature_key.to_string()} not found in store")

        # Extract WHERE clause from compiled SELECT statement
        ibis_filtered = cast("ibis.expr.types.Table", filtered.to_native())
        select_sql = str(ibis_filtered.compile())

        dialect = self._sql_dialect
        predicate = _extract_where_expression(select_sql, dialect=dialect)
        if predicate is None:
            raise ValueError(f"Cannot extract WHERE clause for DELETE on {self.__class__.__name__}")

        # Generate and execute DELETE statement
        predicate = predicate.transform(_strip_table_qualifiers())
        where_clause = predicate.sql(dialect=dialect) if dialect else predicate.sql()

        delete_stmt = f"DELETE FROM {table_name} WHERE {where_clause}"
        self.conn.raw_sql(delete_stmt)  # ty: ignore[unresolved-attribute]

    @property
    def _sql_dialect(self) -> str | None:
        """Extract SQL dialect from the active backend connection."""
        return self.conn.name

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
        native_frame = nw.from_native(table, eager_only=False)
        nw_lazy: nw.LazyFrame[Any] = cast(nw.LazyFrame[Any], cast(object, native_frame))

        # Apply feature_version filter (stays in SQL via Narwhals)
        if feature_version is not None:
            nw_lazy = nw_lazy.filter(nw.col("metaxy_feature_version") == feature_version)

        # Apply generic Narwhals filters (stays in SQL)
        if filters is not None:
            for filter_expr in filters:
                nw_lazy = nw_lazy.filter(filter_expr)

        # Select columns (stays in SQL)
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        # Return Narwhals LazyFrame wrapping Ibis table (stays lazy in SQL)
        return nw_lazy

    def transform_after_read(self, table: "ibis.Table", feature_key: "FeatureKey") -> "ibis.Table":
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

    def transform_before_write(self, df: Frame, feature_key: "FeatureKey", table_name: str) -> Frame:
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
        from metaxy.metadata_store.utils import sanitize_uri

        backend_info = self.connection_string or f"{self.backend}"
        # Sanitize connection strings that may contain credentials
        sanitized_info = sanitize_uri(backend_info)
        return f"{self.__class__.__name__}(backend={sanitized_info})"

    def _get_store_metadata_impl(self, feature_key: CoercibleToFeatureKey) -> dict[str, Any]:
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
