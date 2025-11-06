"""Ibis-based metadata store for SQL databases.

Supports any SQL database that Ibis supports:
- DuckDB, PostgreSQL, MySQL (local/embedded)
- ClickHouse, Snowflake, BigQuery (cloud analytical)
- And 20+ other backends
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import BaseFeature
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    import ibis
    import ibis.expr.types


class IbisMetadataStore(MetadataStore):
    """
    Generic SQL metadata store using Ibis.

    Supports any Ibis backend that supports struct types:
    - DuckDB: Fast local analytical database
    - PostgreSQL: Production-grade RDBMS
    - MySQL: Popular RDBMS
    - ClickHouse: High-performance analytical database
    - And other backends with struct support

    Note: Backends without native struct support (e.g., SQLite) are NOT supported.
    The metaxy_provenance_by_field field requires struct type support for proper storage.

    Storage layout:
    - Each feature gets its own table: {namespace}__{feature_name}
    - System tables: __metaxy__feature_versions, __metaxy__migrations
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
        connection_string: str | None = None,
        *,
        backend: str | None = None,
        connection_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Ibis metadata store.

        Args:
            connection_string: Ibis connection string (e.g., "clickhouse://host:9000/db")
                If provided, backend and connection_params are ignored.
            backend: Ibis backend name (e.g., "clickhouse", "postgres", "duckdb")
                Used with connection_params for more control.
            connection_params: Backend-specific connection parameters
                e.g., {"host": "localhost", "port": 9000, "database": "default"}
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
        try:
            import ibis

            self._ibis = ibis
        except ImportError as e:
            raise ImportError(
                "Ibis is required for IbisMetadataStore. "
                "Install with: pip install ibis-framework[BACKEND] "
                "where BACKEND is one of: duckdb, postgres, clickhouse, mysql, etc."
            ) from e

        if connection_string is None and backend is None:
            raise ValueError(
                "Must provide either connection_string or backend. "
                "Example: connection_string='clickhouse://host:9000/db' "
                "or backend='clickhouse' with connection_params"
            )

        self.connection_string = connection_string
        self.backend = backend
        self.connection_params = connection_params or {}
        self._conn: ibis.BaseBackend | None = None

        super().__init__(**kwargs)

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for Ibis stores.

        Uses MD5 as it's universally supported across SQL databases.
        Subclasses like DuckDBMetadataStore can override for better algorithms.
        """
        return HashAlgorithm.MD5

    def _supports_native_components(self) -> bool:
        """Ibis stores support native (Ibis-based) provenance tracking when connection is open."""
        return self._conn is not None

    def _create_provenance_tracker(self):
        """Create provenance tracker for Ibis backend.

        Returns IbisProvenanceTracker with backend-specific hash functions.
        Base implementation only supports MD5 (universally available).
        Subclasses can override _create_hash_functions() for backend-specific hashes.
        """
        from metaxy.provenance.ibis import IbisProvenanceTracker

        if self._conn is None:
            raise RuntimeError(
                "Cannot create provenance tracker: store is not open. "
                "Ensure store is used as context manager."
            )

        # Get graph from active context (available during resolve_update)
        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()

        # Create hash functions for Ibis expressions
        hash_functions = self._create_hash_functions()

        return IbisProvenanceTracker(
            graph=graph,
            backend=self._conn,
            hash_functions=hash_functions,
        )

    def _create_hash_functions(self):
        """Create hash functions for Ibis expressions.

        Base implementation only supports MD5 (universally available in SQL).
        Subclasses override to add backend-specific hash functions.

        Returns:
            Dictionary mapping HashAlgorithm to Ibis expression functions
        """
        import ibis

        # Use Ibis's builtin UDF decorator to wrap MD5 SQL function
        @ibis.udf.scalar.builtin
        def MD5(x: str) -> str:
            """SQL MD5() function."""
            ...

        @ibis.udf.scalar.builtin
        def HEX(x: str) -> str:
            """SQL HEX() function - converts binary to hex string."""
            ...

        @ibis.udf.scalar.builtin
        def LOWER(x: str) -> str:
            """SQL LOWER() function - converts to lowercase."""
            ...

        # MD5 is universally available in SQL databases
        def md5_hash(col_expr):
            # MD5 returns binary data in most databases, convert to lowercase hex
            return LOWER(HEX(MD5(col_expr.cast(str))))

        return {HashAlgorithm.MD5: md5_hash}

    def _validate_hash_algorithm_support(self) -> None:
        """Validate that the configured hash algorithm is supported by Ibis backend.

        Raises:
            ValueError: If hash algorithm is not supported
        """
        # Create hash functions to check what's supported
        hash_functions = self._create_hash_functions()

        if self.hash_algorithm not in hash_functions:
            supported = [algo.value for algo in hash_functions.keys()]
            raise ValueError(
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

    def open(self) -> None:
        """Open connection to database via Ibis.

        Subclasses should override this to add backend-specific initialization
        (e.g., loading extensions) and should call super().open() first.

        If auto_create_tables is enabled, creates system tables.
        """
        if self.connection_string:
            # Use connection string
            self._conn = self._ibis.connect(self.connection_string)
        else:
            # Use backend + params
            # Get backend-specific connect function
            assert self.backend is not None, (
                "backend must be set if connection_string is None"
            )
            backend_module = getattr(self._ibis, self.backend)
            self._conn = backend_module.connect(**self.connection_params)

        # Auto-create system tables if enabled (warning is handled in base class)
        if self.auto_create_tables:
            self._create_system_tables()

    def _create_system_tables(self) -> None:
        """Create system tables if they don't exist.

        Creates empty system tables with proper schemas:
        - metaxy-system__feature_versions: Tracks feature versions and graph snapshots
        - metaxy-system__migration_events: Tracks migration execution events

        This method is idempotent - safe to call multiple times.
        """
        from metaxy.metadata_store.system_tables import (
            FEATURE_VERSIONS_KEY,
            FEATURE_VERSIONS_SCHEMA,
            MIGRATION_EVENTS_KEY,
            MIGRATION_EVENTS_SCHEMA,
        )

        existing_tables = self.conn.list_tables()

        # Create feature_versions table if it doesn't exist
        feature_versions_table = FEATURE_VERSIONS_KEY.table_name
        if feature_versions_table not in existing_tables:
            empty_df = pl.DataFrame(schema=FEATURE_VERSIONS_SCHEMA)
            self.conn.create_table(feature_versions_table, obj=empty_df)

        # Create migration_events table if it doesn't exist
        migration_events_table = MIGRATION_EVENTS_KEY.table_name
        if migration_events_table not in existing_tables:
            empty_df = pl.DataFrame(schema=MIGRATION_EVENTS_SCHEMA)
            self.conn.create_table(migration_events_table, obj=empty_df)

    def close(self) -> None:
        """Close the Ibis connection."""
        if self._conn is not None:
            # Ibis connections may not have explicit close method
            # but setting to None releases resources
            self._conn = None

    def _table_name_to_feature_key(self, table_name: str) -> FeatureKey:
        """Convert table name back to feature key."""
        return FeatureKey(table_name.split("__"))

    def _write_metadata_impl(
        self,
        feature_key: FeatureKey,
        df: pl.DataFrame,
    ) -> None:
        """
        Internal write implementation using Ibis.

        Args:
            feature_key: Feature key to write to
            df: DataFrame with metadata (already validated)

        Raises:
            TableNotFoundError: If table doesn't exist and auto_create_tables is False
        """
        table_name = feature_key.table_name

        # Check if table exists
        existing_tables = self.conn.list_tables()

        if table_name not in existing_tables:
            # Table doesn't exist - create it if auto_create_tables is enabled
            if not self.auto_create_tables:
                from metaxy.metadata_store.exceptions import TableNotFoundError

                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist for feature {feature_key.to_string()}. "
                    f"Enable auto_create_tables=True to automatically create tables, "
                    f"or use proper database migration tools like Alembic to create the table first."
                )

            # Create table from DataFrame
            # Ensure NULL columns have proper types by filling with a typed value
            # This handles cases like snapshot_version which can be NULL
            df_typed = df
            for col in df.columns:
                if df[col].dtype == pl.Null:
                    # Cast NULL columns to String
                    df_typed = df_typed.with_columns(pl.col(col).cast(pl.Utf8))

            self.conn.create_table(table_name, obj=df_typed)
        else:
            # Append to existing table
            self.conn.insert(table_name, obj=df)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop the table for a feature.

        Args:
            feature_key: Feature key to drop metadata for
        """
        table_name = feature_key.table_name

        # Check if table exists
        if table_name in self.conn.list_tables():
            self.conn.drop_table(table_name)

    def read_metadata_in_store(
        self,
        feature: FeatureKey | type[BaseFeature],
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """
        Read metadata from this store only (no fallback).

        Args:
            feature: Feature to read
            feature_version: Filter by specific feature_version (applied as SQL WHERE clause)
            filters: List of Narwhals filter expressions (converted to SQL WHERE clauses)
            columns: Optional list of columns to select

        Returns:
            Narwhals LazyFrame with metadata, or None if not found
        """
        feature_key = self._resolve_feature_key(feature)
        table_name = feature_key.table_name

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

    def _list_features_local(self) -> list[FeatureKey]:
        """
        List all features in this store.

        Returns:
            List of FeatureKey objects (excluding system tables)
        """
        # Query all table names
        table_names = self.conn.list_tables()

        features = []
        for table_name in table_names:
            # Skip Ibis internal tables (start with "ibis_")
            if table_name.startswith("ibis_"):
                continue

            feature_key = self._table_name_to_feature_key(table_name)

            # Skip system tables
            if not self._is_system_table(feature_key):
                features.append(feature_key)

        return features

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
        if self._is_open:
            num_features = len(self._list_features_local())
            return f"IbisMetadataStore(backend={backend_info}, features={num_features})"
        else:
            return f"IbisMetadataStore(backend={backend_info})"
