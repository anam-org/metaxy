"""Ibis-based metadata store for SQL databases.

Supports any SQL database that Ibis supports:
- DuckDB, PostgreSQL, MySQL (local/embedded)
- ClickHouse, Snowflake, BigQuery (cloud analytical)
- And 20+ other backends
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import Frame
from pydantic import Field
from typing_extensions import Self

from metaxy.metadata_store.base import (
    MetadataStore,
    MetadataStoreConfig,
    VersioningEngineOptions,
)
from metaxy.metadata_store.exceptions import (
    HashAlgorithmNotSupportedError,
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
        import ibis

        self.connection_string = connection_string
        self.backend = backend
        self.connection_params = connection_params or {}
        self._conn: ibis.BaseBackend | None = None
        self._table_prefix = table_prefix or ""

        super().__init__(
            **kwargs,
            versioning_engine=versioning_engine,
            versioning_engine_cls=IbisVersioningEngine,
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

        # Create engine (only accepts plan and hash_functions)
        engine = IbisVersioningEngine(
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
        from metaxy.metadata_store.exceptions import StoreNotOpenError

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
                    assert self.backend is not None, (
                        "backend must be set if connection_string is None"
                    )
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
                    # Call disconnect if available to properly close connection
                    if hasattr(self._conn, "disconnect"):
                        self._conn.disconnect()
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
        if df.implementation == nw.Implementation.IBIS:
            df_to_insert = df.to_native()  # Ibis expression
        else:
            from metaxy._utils import collect_to_polars

            df_to_insert = collect_to_polars(df)  # Polars DataFrame

        table_name = self.get_table_name(feature_key)

        try:
            self.conn.insert(table_name, obj=df_to_insert)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
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

        # Wrap Ibis table with Narwhals (stays lazy in SQL)
        nw_lazy: nw.LazyFrame[Any] = nw.from_native(table, eager_only=False)

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

    # ========== Error Tracking Implementation ==========

    def write_errors_to_store(
        self,
        feature_key: FeatureKey,
        errors_df: Frame,
    ) -> None:
        """Write error records to SQL error table using Ibis.

        Args:
            feature_key: Feature key to write errors for
            errors_df: Narwhals DataFrame with error records
        """
        # Convert to appropriate format for Ibis
        if errors_df.implementation == nw.Implementation.IBIS:
            df_to_insert = errors_df.to_native()
        else:
            from metaxy._utils import collect_to_polars

            df_to_insert = collect_to_polars(errors_df)

        # Get error table name using base class helper
        error_table_name = self._get_error_table_name(feature_key)
        # Apply table prefix if configured
        if self._table_prefix:
            error_table_name = f"{self._table_prefix}{error_table_name}"

        # Try to insert, create table if needed
        try:
            self.conn.insert(error_table_name, obj=df_to_insert)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        except Exception as e:
            import ibis.common.exceptions

            if not isinstance(e, ibis.common.exceptions.TableNotFound):
                raise

            # Auto-create error table
            if self.auto_create_tables:
                self.conn.create_table(error_table_name, obj=df_to_insert)
            else:
                raise TableNotFoundError(
                    f"Error table '{error_table_name}' does not exist. "
                    f"Set auto_create_tables=True to create automatically, "
                    f"or create the table manually with appropriate schema."
                ) from e

    def read_errors_from_store(
        self,
        feature_key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read error records from SQL error table using Ibis.

        Args:
            feature_key: Feature key to read errors for
            filters: Optional Narwhals filter expressions to apply

        Returns:
            Narwhals LazyFrame with error records, or None if error table doesn't exist
        """
        # Get error table name
        error_table_name = self._get_error_table_name(feature_key)
        if self._table_prefix:
            error_table_name = f"{self._table_prefix}{error_table_name}"

        # Check if error table exists
        existing_tables = self.conn.list_tables()
        if error_table_name not in existing_tables:
            return None

        # Get Ibis table reference
        table = self.conn.table(error_table_name)

        # Wrap with Narwhals (stays lazy in SQL)
        nw_lazy: nw.LazyFrame[Any] = nw.from_native(table, eager_only=False)

        # Apply filters if provided
        if filters is not None:
            for filter_expr in filters:
                nw_lazy = nw_lazy.filter(filter_expr)

        return nw_lazy

    def clear_errors_from_store(
        self,
        feature_key: FeatureKey,
        *,
        sample_uids: Sequence[dict[str, Any]] | None = None,
        feature_version: str | None = None,
    ) -> None:
        """Clear error records from SQL error table using Ibis.

        Args:
            feature_key: Feature key to clear errors for
            sample_uids: Optional list of sample ID dicts to clear
            feature_version: Optional feature version to clear
        """
        # Get error table name
        error_table_name = self._get_error_table_name(feature_key)
        if self._table_prefix:
            error_table_name = f"{self._table_prefix}{error_table_name}"

        # Check if error table exists
        if error_table_name not in self.conn.list_tables():
            return  # No-op if table doesn't exist

        # If no filters, drop entire error table
        if sample_uids is None and feature_version is None:
            self.conn.drop_table(error_table_name)
            return

        # Otherwise, use SQL DELETE with WHERE clause
        # Get table reference
        table = self.conn.table(error_table_name)

        # Build WHERE conditions
        conditions = []

        # Filter by feature_version if provided
        if feature_version is not None:
            conditions.append(
                table.metaxy_feature_version == feature_version  # type: ignore[attr-defined]
            )

        # Filter by sample_uids if provided
        if sample_uids is not None and len(sample_uids) > 0:
            # Get id_columns from feature spec
            feature_spec = self._resolve_feature_plan(feature_key).feature
            id_cols = list(feature_spec.id_columns)

            # Build OR condition for all samples
            sample_conditions = []
            for uid_dict in sample_uids:
                # Build AND condition for this sample
                and_conditions = []
                for col_name in id_cols:
                    col_value = uid_dict[col_name]
                    and_conditions.append(
                        table[col_name] == col_value  # type: ignore[index]
                    )

                # Combine with AND using reduce
                if len(and_conditions) == 1:
                    sample_cond = and_conditions[0]
                else:
                    import functools
                    import operator

                    sample_cond = functools.reduce(operator.and_, and_conditions)

                sample_conditions.append(sample_cond)

            # Combine all sample conditions with OR
            if len(sample_conditions) == 1:
                sample_filter = sample_conditions[0]
            else:
                import functools
                import operator

                sample_filter = functools.reduce(operator.or_, sample_conditions)

            conditions.append(sample_filter)

        # Execute selective delete
        # Note: Ibis doesn't have a direct DELETE API
        # Workaround: read non-matching rows, drop table, recreate with filtered data
        if conditions:
            # Combine all conditions with AND
            import functools
            import operator

            if len(conditions) == 1:
                where_clause = conditions[0]
            else:
                where_clause = functools.reduce(operator.and_, conditions)

            # Build keep clause (inverse of delete clause)
            keep_clause = ~where_clause

            # Read rows we want to KEEP
            rows_to_keep = table.filter(keep_clause).to_pandas()

            # Drop table
            self.conn.drop_table(error_table_name)

            # Recreate with remaining rows (if any)
            if len(rows_to_keep) > 0:
                self.conn.create_table(error_table_name, obj=rows_to_keep)

    def display(self) -> str:
        """Display string for this store."""
        from metaxy.metadata_store.utils import sanitize_uri

        backend_info = self.connection_string or f"{self.backend}"
        # Sanitize connection strings that may contain credentials
        sanitized_info = sanitize_uri(backend_info)
        return f"{self.__class__.__name__}(backend={sanitized_info})"

    @classmethod
    def config_model(cls) -> type[IbisMetadataStoreConfig]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return IbisMetadataStoreConfig
