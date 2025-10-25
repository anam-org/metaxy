"""Ibis-based metadata store for SQL databases.

Supports any SQL database that Ibis supports:
- DuckDB, PostgreSQL, MySQL, SQLite (local/embedded)
- ClickHouse, Snowflake, BigQuery (cloud analytical)
- And 20+ other backends
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import Feature
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    import ibis
    import ibis.expr.types

    from metaxy.data_versioning.calculators.ibis import HashSQLGenerator


class IbisMetadataStore(MetadataStore):
    """
    Generic SQL metadata store using Ibis.

    Supports any Ibis backend including:
    - DuckDB: Fast local analytical database
    - PostgreSQL: Production-grade RDBMS
    - MySQL: Popular RDBMS
    - ClickHouse: High-performance analytical database
    - SQLite: Embedded database
    - And 20+ other backends

    Storage layout:
    - Each feature gets its own table: {namespace}__{feature_name}
    - System tables: __metaxy__feature_versions, __metaxy__migrations
    - Uses Ibis for cross-database compatibility

    Note: Uses MD5 hash by default for cross-database compatibility.
    DuckDBMetadataStore overrides this with dynamic algorithm detection.
    For other backends, override the calculator instance variable with backend-specific implementations.

    Example:
        >>> # ClickHouse
        >>> store = IbisMetadataStore("clickhouse://user:pass@host:9000/db")
        >>>
        >>> # PostgreSQL
        >>> store = IbisMetadataStore("postgresql://user:pass@host:5432/db")
        >>>
        >>> # DuckDB (use DuckDBMetadataStore instead for better hash support)
        >>> store = IbisMetadataStore("duckdb:///metadata.db")
        >>>
        >>> with store:
        ...     store.write_metadata(MyFeature, df)
    """

    @classmethod
    def supports_structs(cls) -> bool:
        """Check if backend supports struct types natively.

        Subclasses should override this for backends that don't support structs.
        Default implementation returns True (most SQL databases support structs).

        Returns:
            True if backend supports structs, False if needs JSON serialization
        """
        return True

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        backend: str | None = None,
        connection_params: dict[str, Any] | None = None,
        **kwargs,
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
            >>> # Using connection string
            >>> store = IbisMetadataStore("clickhouse://user:pass@host:9000/db")
            >>>
            >>> # Using backend + params
            >>> store = IbisMetadataStore(
            ...     backend="clickhouse",
            ...     connection_params={"host": "localhost", "port": 9000}
            ... )
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
        """Ibis stores support native (Ibis-based) components when connection is open."""
        return self._conn is not None

    def _create_native_components(self):
        """Create components for native SQL execution via Ibis."""
        from metaxy.data_versioning.calculators.ibis import IbisDataVersionCalculator
        from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver
        from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner

        if self._conn is None:
            raise RuntimeError(
                "Cannot create native data version calculations: store is not open. "
                "Ensure store is used as context manager."
            )

        # All components accept/return Narwhals LazyFrames
        # IbisDataVersionCalculator converts to Ibis internally for SQL hash generation
        joiner = NarwhalsJoiner()
        calculator = IbisDataVersionCalculator(
            backend=self._conn,
            hash_sql_generators=self._get_hash_sql_generators(),
        )
        diff_resolver = NarwhalsDiffResolver()

        return joiner, calculator, diff_resolver

    def _get_hash_sql_generators(self) -> dict[HashAlgorithm, "HashSQLGenerator"]:
        """Get hash SQL generators for this backend.

        Base implementation only supports MD5 (universally available in SQL).
        Subclasses override to add backend-specific hash functions.

        Returns:
            Dictionary mapping HashAlgorithm to SQL generator functions
        """

        def md5_generator(table, concat_columns: dict[str, str]) -> str:
            """Generate SQL to compute MD5 hashes (universal SQL support).

            Note: This generic implementation assumes MD5() returns a hex string.
            Subclasses should override if their backend returns binary or different format.
            For example, ClickHouse returns binary and needs lower(hex(MD5(...))).
            """
            # Build SELECT clause with hash columns
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{field_key}"
                # Use MD5 function (universally available in SQL databases)
                # WARNING: Different databases return different formats (hex string vs binary)
                # This generic version assumes hex string output
                hash_expr = f"MD5({concat_col})"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        return {HashAlgorithm.MD5: md5_generator}

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

    def close(self) -> None:
        """Close the Ibis connection."""
        if self._conn is not None:
            # Ibis connections may not have explicit close method
            # but setting to None releases resources
            self._conn = None

    def _feature_key_to_table_name(self, feature_key: FeatureKey) -> str:
        """
        Convert feature key to SQL table name.

        Examples:
            FeatureKey(["my_namespace", "my_feature"]) -> "my_namespace__my_feature"
            FeatureKey(["metaxy-system", "feature_versions"]) -> "metaxy-system__feature_versions"
        """
        return "__".join(feature_key)

    def _table_name_to_feature_key(self, table_name: str) -> FeatureKey:
        """Convert table name back to feature key."""
        return FeatureKey(table_name.split("__"))

    def _serialize_for_storage(self, df: pl.DataFrame) -> pl.DataFrame:
        """Serialize DataFrame for storage (e.g., convert structs to JSON).

        Base implementation does nothing - backends that don't support structs
        should override this method.

        Args:
            df: DataFrame to serialize

        Returns:
            Serialized DataFrame
        """
        return df

    def _deserialize_from_storage(self, df: pl.DataFrame) -> pl.DataFrame:
        """Deserialize DataFrame from storage (e.g., convert JSON back to structs).

        Base implementation does nothing - backends that don't support structs
        should override this method.

        Args:
            df: DataFrame to deserialize

        Returns:
            Deserialized DataFrame
        """
        return df

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
        """
        table_name = self._feature_key_to_table_name(feature_key)

        # Serialize for storage (e.g., convert structs to JSON for SQLite)
        df = self._serialize_for_storage(df)

        # Check if table exists
        existing_tables = self.conn.list_tables()

        if table_name not in existing_tables:
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
        table_name = self._feature_key_to_table_name(feature_key)

        # Check if table exists
        if table_name in self.conn.list_tables():
            self.conn.drop_table(table_name)

    def _read_metadata_native(
        self,
        feature: FeatureKey | type[Feature],
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
        table_name = self._feature_key_to_table_name(feature_key)

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
            nw_lazy = nw_lazy.filter(nw.col("feature_version") == feature_version)

        # Apply generic Narwhals filters (stays in SQL)
        if filters is not None:
            for filter_expr in filters:
                nw_lazy = nw_lazy.filter(filter_expr)

        # Select columns (stays in SQL)
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        # For backends that don't support structs (e.g., SQLite),
        # we need to deserialize JSON strings to structs
        if not self.supports_structs():
            # Convert to Polars, deserialize, then wrap back as Narwhals lazy
            table_native = nw_lazy.to_native()
            if hasattr(table_native, "to_polars"):
                # Ibis table
                df_polars = table_native.to_polars()
            else:
                # Already Polars
                df_polars = (
                    table_native
                    if isinstance(table_native, pl.DataFrame)
                    else table_native.collect()
                )

            # Deserialize JSON → structs
            df_polars = self._deserialize_from_storage(df_polars)

            # Make lazy and wrap in Narwhals
            return nw.from_native(df_polars.lazy())

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
        Ibis backends support native data version calculations (Narwhals-based).

        Returns:
            True (use Narwhals components with Ibis-backed tables)

        Note: All Ibis stores now use Narwhals-based components (NarwhalsJoiner,
        PolarsDataVersionCalculator, NarwhalsDiffResolver) which work efficiently
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
