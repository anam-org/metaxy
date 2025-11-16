"""Ibis implementation of VersioningEngine.

CRITICAL: This implementation NEVER materializes lazy expressions.
All operations stay in the lazy Ibis world for SQL execution.
"""

from typing import Protocol, cast

import narwhals as nw
from ibis import Expr as IbisExpr
from narwhals.typing import FrameT

from metaxy.models.plan import FeaturePlan
from metaxy.versioning.engine import VersioningEngine
from metaxy.versioning.types import HashAlgorithm


class IbisHashFn(Protocol):
    def __call__(self, expr: IbisExpr) -> IbisExpr: ...


class IbisVersioningEngine(VersioningEngine):
    """Provenance engine using Ibis for SQL databases.

    Only implements hash_string_column and record_field_versions.
    All logic lives in the base class.

    CRITICAL: This implementation NEVER leaves the lazy world.
    All operations stay as Ibis expressions that compile to SQL.
    """

    def __init__(
        self,
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
            df: Narwhals DataFrame backed by Ibis
            source_column: Name of string column to hash
            target_column: Name for the new column containing the hash
            hash_algo: Hash algorithm to use

        Returns:
            Narwhals DataFrame with new hashed column added, backed by Ibis.
            The source column remains unchanged.
        """
        if hash_algo not in self.hash_functions:
            raise ValueError(
                f"Hash algorithm {hash_algo} not supported by this Ibis backend. "
                f"Supported: {list(self.hash_functions.keys())}"
            )

        # Import ibis lazily (module-level import restriction)
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Get hash function
        hash_fn = self.hash_functions[hash_algo]

        # Apply hash to source column
        # Hash functions are responsible for returning strings
        hashed = hash_fn(ibis_table[source_column])

        # Add new column with the hash
        result_table = ibis_table.mutate(**{target_column: hashed})  # pyright: ignore[reportArgumentType]

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def record_field_versions(
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Persist field-level versions using a struct column.

        Args:
            df: Narwhals DataFrame backed by Ibis
            struct_name: Name for the new struct column
            field_columns: Mapping from struct field names to column names

        Returns:
            Narwhals DataFrame with the struct column added.
        """
        # Import ibis lazily
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Build struct expression - reference columns by name
        struct_expr = ibis.struct(
            {
                field_name: ibis_table[col_name]
                for field_name, col_name in field_columns.items()
            }
        )

        # Add struct column
        result_table = ibis_table.mutate(**{struct_name: struct_expr})

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def aggregate_with_string_concat(
        df: FrameT,
        group_by_columns: list[str],
        concat_column: str,
        concat_separator: str,
        exclude_columns: list[str],
    ) -> FrameT:
        """Aggregate DataFrame by grouping and concatenating strings.

        Args:
            df: Narwhals DataFrame backed by Ibis
            group_by_columns: Columns to group by
            concat_column: Column containing strings to concatenate within groups
            concat_separator: Separator to use when concatenating strings
            exclude_columns: Columns to exclude from aggregation

        Returns:
            Narwhals DataFrame with one row per group.
        """
        # Import ibis lazily
        import ibis
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Build aggregation expressions
        agg_exprs = {}

        # Concatenate the concat_column with separator
        agg_exprs[concat_column] = ibis_table[concat_column].group_concat(
            concat_separator
        )

        # Take first value for all other columns (except group_by and exclude)
        all_columns = set(ibis_table.columns)
        columns_to_aggregate = (
            all_columns - set(group_by_columns) - {concat_column} - set(exclude_columns)
        )

        for col in columns_to_aggregate:
            agg_exprs[col] = ibis_table[
                col
            ].arbitrary()  # Take any value (like first())

        # Perform groupby and aggregate
        result_table = ibis_table.group_by(group_by_columns).aggregate(**agg_exprs)

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_column: str,
    ) -> FrameT:
        """Keep only the latest row per group based on a timestamp column.

        Uses argmax aggregation to get the value from each column where the
        timestamp is maximum. This is simpler and more semantically clear than
        window functions.

        Args:
            df: Narwhals DataFrame/LazyFrame backed by Ibis
            group_columns: Columns to group by (typically ID columns)
            timestamp_column: Column to use for determining "latest" (typically metaxy_created_at)

        Returns:
            Narwhals DataFrame/LazyFrame with only the latest row per group

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
        self._table_prefix = table_prefix or ""

        super().__init__(**kwargs, versioning_engine_cls=IbisVersioningEngine)

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
    def open(self, mode: AccessMode = AccessMode.READ) -> Iterator[Self]:
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
                # Setup: Connect to database
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

    def _table_name_to_feature_key(self, table_name: str) -> FeatureKey:
        """Convert table name back to feature key."""
        return FeatureKey(table_name.split("__"))

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
    ) -> None:
        """Initialize the Ibis engine.

        Args:
            plan: Feature plan to track provenance for
            backend: Ibis backend instance (e.g., ibis.duckdb.connect())
            hash_functions: Mapping from HashAlgorithm to Ibis hash functions.
                Each function takes an Ibis expression and returns an Ibis expression.
        """
        super().__init__(plan)
        self.hash_functions: dict[HashAlgorithm, IbisHashFn] = hash_functions

    @classmethod
    def implementation(cls) -> nw.Implementation:
        return nw.Implementation.IBIS

    def hash_string_column(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        hash_algo: HashAlgorithm,
    ) -> FrameT:
        """Hash a string column using Ibis hash functions.

        Args:
            df: Narwhals DataFrame backed by Ibis
            source_column: Name of string column to hash
            target_column: Name for the new column containing the hash
            hash_algo: Hash algorithm to use

        Returns:
            Narwhals DataFrame with new hashed column added, backed by Ibis.
            The source column remains unchanged.
        """
        if hash_algo not in self.hash_functions:
            raise ValueError(
                f"Hash algorithm {hash_algo} not supported by this Ibis backend. "
                f"Supported: {list(self.hash_functions.keys())}"
            )

        # Import ibis lazily (module-level import restriction)
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Get hash function
        hash_fn = self.hash_functions[hash_algo]

        # Apply hash to source column
        # Hash functions are responsible for returning strings
        hashed = hash_fn(ibis_table[source_column])

        # Add new column with the hash
        result_table = ibis_table.mutate(**{target_column: hashed})  # pyright: ignore[reportArgumentType]

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def build_struct_column(
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Build a struct column from existing columns.

        Args:
            df: Narwhals DataFrame backed by Ibis
            struct_name: Name for the new struct column
            field_columns: Mapping from struct field names to column names

        Returns:
            Narwhals DataFrame with new struct column added, backed by Ibis.
            The source columns remain unchanged.
        """
        # Import ibis lazily
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Build struct expression - reference columns by name
        struct_expr = ibis.struct(
            {
                field_name: ibis_table[col_name]
                for field_name, col_name in field_columns.items()
            }
        )

        # Add struct column
        result_table = ibis_table.mutate(**{struct_name: struct_expr})

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def aggregate_with_string_concat(
        df: FrameT,
        group_by_columns: list[str],
        concat_column: str,
        concat_separator: str,
        exclude_columns: list[str],
    ) -> FrameT:
        """Aggregate DataFrame by grouping and concatenating strings.

        Args:
            df: Narwhals DataFrame backed by Ibis
            group_by_columns: Columns to group by
            concat_column: Column containing strings to concatenate within groups
            concat_separator: Separator to use when concatenating strings
            exclude_columns: Columns to exclude from aggregation

        Returns:
            Narwhals DataFrame with one row per group.
        """
        # Import ibis lazily
        import ibis
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Build aggregation expressions
        agg_exprs = {}

        # Concatenate the concat_column with separator
        agg_exprs[concat_column] = ibis_table[concat_column].group_concat(
            concat_separator
        )

        # Take first value for all other columns (except group_by and exclude)
        all_columns = set(ibis_table.columns)
        columns_to_aggregate = (
            all_columns - set(group_by_columns) - {concat_column} - set(exclude_columns)
        )

        for col in columns_to_aggregate:
            agg_exprs[col] = ibis_table[
                col
            ].arbitrary()  # Take any value (like first())

        # Perform groupby and aggregate
        result_table = ibis_table.group_by(group_by_columns).aggregate(**agg_exprs)

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_column: str,
    ) -> FrameT:
        """Keep only the latest row per group based on a timestamp column.

        Uses argmax aggregation to get the value from each column where the
        timestamp is maximum. This is simpler and more semantically clear than
        window functions.

        Args:
            df: Narwhals DataFrame/LazyFrame backed by Ibis
            group_columns: Columns to group by (typically ID columns)
            timestamp_column: Column to use for determining "latest" (typically metaxy_created_at)

        Returns:
            Narwhals DataFrame/LazyFrame with only the latest row per group

        Raises:
            ValueError: If timestamp_column doesn't exist in df
        """
        # Import ibis lazily
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )

        # Check if timestamp_column exists
        if timestamp_column not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_column}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )

        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Use argmax aggregation: for each column, get the value where timestamp is maximum
        # This directly expresses "get the row with the latest timestamp per group"
        all_columns = set(ibis_table.columns)
        non_group_columns = all_columns - set(group_columns)

        # Build aggregation dict: for each non-group column, use argmax(timestamp)
        agg_exprs = {
            col: ibis_table[col].argmax(ibis_table[timestamp_column])
            for col in non_group_columns
        }

        # Perform groupby and aggregate
        result_table = ibis_table.group_by(group_columns).aggregate(**agg_exprs)

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))
