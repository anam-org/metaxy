"""SQLite metadata store using ADBC for high-performance bulk ingestion.

Uses Arrow Database Connectivity (ADBC) for zero-copy data transfer and bulk writes.
Provides 2-10x faster write performance for SQLite compared to traditional approaches.

Note: SQLite is single-threaded, so max_connections>1 won't provide concurrency benefits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field

from metaxy.metadata_store.adbc import ADBCMetadataStore, ADBCMetadataStoreConfig
from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ADBCSQLiteMetadataStoreConfig(ADBCMetadataStoreConfig):
    """Configuration for ADBC SQLite metadata store.

    Example:
        ```python
        config = ADBCSQLiteMetadataStoreConfig(
            database="metadata.db",
        )
        ```
    """

    database: str | Path = Field(
        description="Database file path or :memory: for in-memory database.",
    )


class ADBCSQLiteMetadataStore(ADBCMetadataStore):
    """
    SQLite metadata store using ADBC for high-performance bulk writes.

    Provides lightweight metadata storage using SQLite with:
    - Zero-copy Arrow data transfer via ADBC
    - Bulk ingestion (2-10x faster than traditional SQLite)
    - File-based and in-memory storage
    - No external dependencies (perfect for testing and CI)

    Note:
        This is an initial implementation. Some advanced features (JSON struct storage,
        filters, deletes) are marked as TODO and will be implemented in subsequent PRs.
        SQLite is single-threaded, so max_connections>1 has no effect.

    Example:
        ```python
        from metaxy.metadata_store.adbc_sqlite import ADBCSQLiteMetadataStore

        # File-based storage
        store = ADBCSQLiteMetadataStore(
            database="metadata.db",
            hash_algorithm=HashAlgorithm.MD5,
        )

        # In-memory database (for testing)
        store = ADBCSQLiteMetadataStore(
            database=":memory:",
        )
        ```
    """

    def __init__(
        self,
        database: str | Path,
        **kwargs: Any,
    ) -> None:
        """Initialize ADBC SQLite metadata store.

        Args:
            database: Database file path or :memory: for in-memory database.
                Examples: "metadata.db", Path("metadata.db"), ":memory:"
            **kwargs: Passed to ADBCMetadataStore (e.g., hash_algorithm).
                Note: max_connections>1 has no effect (SQLite is single-threaded).
        """
        # Convert Path to string
        database_str = str(database)

        # ADBC SQLite driver expects "uri" parameter
        super().__init__(connection_params={"uri": database_str}, **kwargs)

        self.database = database_str

        # SQLite supports MD5 via crypto extension (optional)
        # For now, we'll support MD5 as it's the most portable
        supported_algorithms = {HashAlgorithm.MD5}
        if self.hash_algorithm not in supported_algorithms:
            raise HashAlgorithmNotSupportedError(
                f"ADBCSQLiteMetadataStore currently supports only MD5. Requested: {self.hash_algorithm}"
            )

    @classmethod
    def config_model(cls) -> type[ADBCMetadataStoreConfig]:
        """Return the configuration model for this store type."""
        return ADBCSQLiteMetadataStoreConfig

    def _get_driver_name(self) -> str:
        """Get the ADBC driver path for SQLite."""
        import adbc_driver_sqlite  # type: ignore[import-untyped]

        return adbc_driver_sqlite._driver_path()  # type: ignore[no-any-return]

    def _get_connection_options(self) -> dict[str, Any]:
        """Get SQLite-specific connection options for ADBC."""
        options: dict[str, Any] = {}

        # ADBC SQLite driver uses 'uri' parameter
        if self.connection_params:
            options.update(self.connection_params)
        elif self.connection_string:
            # If connection string was provided, use it as uri
            options["uri"] = self.connection_string

        return options

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Write metadata using ADBC bulk ingestion.

        Args:
            feature_key: Feature identifier
            df: Narwhals DataFrame to write
            **kwargs: Backend-specific options
        """
        from metaxy.metadata_store.exceptions import StoreNotOpenError

        if not self._is_open or self._conn is None:
            raise StoreNotOpenError("Store must be open to write metadata")

        # Convert to Polars for processing
        polars_df: pl.DataFrame
        native = df.to_native()
        if isinstance(native, pl.LazyFrame):
            polars_df = native.collect()
        else:
            polars_df = native  # type: ignore[assignment]

        # Flatten struct columns to separate columns (e.g., metaxy_provenance_by_field__foo)
        # This matches the PostgreSQL approach and preserves type information
        from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD

        struct_columns = {METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD}
        for col in struct_columns:
            if col in polars_df.columns:
                col_dtype = polars_df[col].dtype
                if col_dtype == pl.Struct:
                    # Get field names from struct
                    struct_fields = polars_df[col].dtype.fields

                    # Create flattened columns for each field
                    flatten_exprs = [
                        pl.col(col).struct.field(field.name).alias(f"{col}__{field.name}") for field in struct_fields
                    ]
                    polars_df = polars_df.with_columns(flatten_exprs)

                    # Drop the original struct column
                    polars_df = polars_df.drop(col)

        # Convert to Arrow for ADBC
        arrow_table = polars_df.to_arrow()

        # Use ADBC bulk ingestion via statement
        table_name = self._table_name(feature_key)

        # Check if table exists and determine ingestion mode
        import adbc_driver_manager  # ty: ignore[unresolved-import]

        from metaxy.metadata_store.exceptions import TableNotFoundError

        table_exists = self._has_feature_impl(feature_key)

        if not table_exists:
            if not self.auto_create_tables:
                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist for feature {feature_key.to_string()}. "
                    f"Enable auto_create_tables=True to automatically create tables, "
                    f"or use proper database migration tools to create the table first."
                )
            # Use CREATE mode to create the table
            ingest_mode = adbc_driver_manager.INGEST_OPTION_MODE_CREATE
            if self._should_warn_auto_create_tables:
                import warnings

                warnings.warn(
                    f"AUTO_CREATE_TABLES is enabled - automatically creating table '{table_name}'. "
                    "Do not use in production! "
                    "Use proper database migration tools for production deployments.",
                    UserWarning,
                    stacklevel=4,
                )
        else:
            # Table exists, check if we need to add columns for schema evolution
            self._ensure_columns_exist(table_name, arrow_table.schema)
            # Use APPEND mode
            ingest_mode = adbc_driver_manager.INGEST_OPTION_MODE_APPEND

        # Create statement for ingestion
        stmt = adbc_driver_manager.AdbcStatement(self._conn)
        try:
            # Set ingestion options
            stmt.set_options(
                **{
                    adbc_driver_manager.INGEST_OPTION_TARGET_TABLE: table_name,
                    adbc_driver_manager.INGEST_OPTION_MODE: ingest_mode,
                }
            )

            # Bind Arrow data and execute
            stmt.bind_stream(arrow_table)
            stmt.execute_update()
        finally:
            stmt.close()

    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Any = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata from SQLite via ADBC.

        Args:
            feature: Feature to read
            filters: Narwhals filter expressions to apply after reading
            **kwargs: Backend-specific options

        Returns:
            Narwhals LazyFrame or None if not found

        Note:
            Filters are applied in Narwhals after reading data from the database.
            Future optimization: Convert filters to SQL WHERE clauses.
        """
        from metaxy.metadata_store.exceptions import StoreNotOpenError

        if not self._is_open or self._conn is None:
            raise StoreNotOpenError("Store must be open to read metadata")

        feature_key = self._resolve_feature_key(feature)
        table_name = self._table_name(feature_key)

        # Check if table exists
        if not self._has_feature_impl(feature_key):
            return None

        # Query via ADBC statement
        query = f'SELECT * FROM "{table_name}"'

        import adbc_driver_manager  # ty: ignore[unresolved-import]

        stmt = adbc_driver_manager.AdbcStatement(self._conn)
        try:
            stmt.set_sql_query(query)
            stream, _ = stmt.execute_query()

            # Convert ADBC stream to Arrow table
            import pyarrow as pa

            reader = pa.RecordBatchReader.from_stream(stream)
            arrow_table = reader.read_all()

            # Convert to Polars
            result = pl.from_arrow(arrow_table)

            # Ensure we have a DataFrame
            if not isinstance(result, pl.DataFrame):
                raise TypeError(f"Expected DataFrame from ADBC query, got {type(result)}")

            # Reconstruct struct columns from flattened columns
            # This preserves ALL fields across schema changes
            result = self._reconstruct_struct_columns(result, feature_key)

            # Convert datetime text columns back to datetime
            # SQLite stores timestamps as TEXT, need to parse them back
            result = self._convert_datetime_columns(result)

            # Convert Int64 null columns to their proper types
            # SQLite stores NULL-only columns as Int64, need to cast them
            result = self._convert_null_columns(result)

            # Convert to Narwhals LazyFrame
            lazy_result = nw.from_native(result.lazy())

            # Apply filters in Narwhals (TODO: optimize by converting to SQL WHERE clauses)
            if filters:
                for filter_expr in filters:
                    lazy_result = lazy_result.filter(filter_expr)

            return lazy_result
        finally:
            stmt.close()

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop all metadata for a feature.

        Args:
            feature_key: Feature to drop
        """
        from metaxy.metadata_store.exceptions import StoreNotOpenError

        if not self._is_open or self._conn is None:
            raise StoreNotOpenError("Store must be open to drop metadata")

        table_name = self._table_name(feature_key)
        query = f'DROP TABLE IF EXISTS "{table_name}"'

        import adbc_driver_manager  # ty: ignore[unresolved-import]

        stmt = adbc_driver_manager.AdbcStatement(self._conn)
        try:
            stmt.set_sql_query(query)
            stmt.execute_update()
        finally:
            stmt.close()

    def _delete_metadata_impl(
        self,
        feature_key: FeatureKey,
        filters: Any,
        *,
        current_only: bool,
    ) -> None:
        """Delete metadata rows.

        Args:
            feature_key: Feature to delete from
            filters: Filter expressions (TODO: not yet implemented)
            current_only: If True, only delete non-deleted records
        """
        # TODO: Implement DELETE with filters via ADBC in future PR
        raise NotImplementedError("DELETE operations not yet implemented for ADBC stores")

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        """Check if feature exists in store.

        Args:
            feature: Feature to check

        Returns:
            True if feature table exists
        """
        from metaxy.metadata_store.exceptions import StoreNotOpenError

        if not self._is_open or self._conn is None:
            raise StoreNotOpenError("Store must be open to check features")

        feature_key = self._resolve_feature_key(feature)
        table_name = self._table_name(feature_key)

        # Query sqlite_master to check table existence
        query = f"""
            SELECT EXISTS (
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = '{table_name}'
            )
        """

        import adbc_driver_manager  # ty: ignore[unresolved-import]

        stmt = adbc_driver_manager.AdbcStatement(self._conn)
        try:
            stmt.set_sql_query(query)
            stream, _ = stmt.execute_query()

            # Convert ADBC stream to Arrow table
            import pyarrow as pa

            reader = pa.RecordBatchReader.from_stream(stream)
            arrow_table = reader.read_all()

            # Get the boolean result
            result = arrow_table.to_pylist()
            # SQLite returns the result with a generated column name
            if result and len(result) > 0:
                # Get the first column value (EXISTS result)
                first_col_name = list(result[0].keys())[0]
                return bool(result[0][first_col_name])
            return False
        finally:
            stmt.close()

    def _convert_null_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert null-only Int64 columns to their proper types.

        SQLite stores NULL-only columns as Int64. When certain string columns
        have all NULL values, they come back as Int64 and need to be cast.

        Args:
            df: DataFrame with Int64 null columns

        Returns:
            DataFrame with proper types
        """
        from metaxy.models.constants import METAXY_MATERIALIZATION_ID

        # String columns that might be stored as Int64 when all NULL
        string_columns = [METAXY_MATERIALIZATION_ID]

        for col in string_columns:
            if col not in df.columns:
                continue

            col_dtype = df[col].dtype

            # If it's Int64 (likely all NULL), cast to String
            if col_dtype == pl.Int64:
                df = df.with_columns(pl.col(col).cast(pl.Utf8).alias(col))

        return df

    def _convert_datetime_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert text datetime columns back to datetime type.

        SQLite stores timestamps as TEXT. When read back via Arrow, they're strings.
        This converts known datetime columns back to proper datetime type.

        Args:
            df: DataFrame with text datetime columns

        Returns:
            DataFrame with datetime columns converted
        """
        from metaxy.models.constants import METAXY_CREATED_AT, METAXY_DELETED_AT

        datetime_columns = [METAXY_CREATED_AT, METAXY_DELETED_AT]

        for col in datetime_columns:
            if col not in df.columns:
                continue

            col_dtype = df[col].dtype

            # If it's already datetime, skip
            if col_dtype == pl.Datetime:
                continue

            # If it's string (text), convert to datetime
            if col_dtype == pl.Utf8:
                df = df.with_columns(pl.col(col).str.to_datetime(time_unit="us", time_zone="UTC").alias(col))
            # If it's Int64 (null-only column), cast to datetime
            elif col_dtype == pl.Int64:
                df = df.with_columns(pl.col(col).cast(pl.Datetime(time_unit="us", time_zone="UTC")).alias(col))

        return df

    def _reconstruct_struct_columns(self, df: pl.DataFrame, feature_key: FeatureKey) -> pl.DataFrame:
        """Reconstruct struct columns from flattened columns.

        Includes ALL fields present in the database to preserve historical data
        across schema changes.

        Args:
            df: Polars DataFrame with flattened columns
            feature_key: Feature key

        Returns:
            DataFrame with struct columns reconstructed
        """
        from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD

        struct_columns = {METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD}

        for struct_col in struct_columns:
            # Find ALL flattened columns for this struct
            prefix = f"{struct_col}__"
            all_flattened_cols = [col for col in df.columns if col.startswith(prefix)]

            if all_flattened_cols:
                # Extract all field names from column names (preserves all historical data)
                all_field_names = {col[len(prefix) :] for col in all_flattened_cols}

                # Build struct from ALL fields present in the data
                field_exprs = []
                for field_name in sorted(all_field_names):
                    col_name = f"{prefix}{field_name}"
                    field_exprs.append(pl.col(col_name).alias(field_name))

                # Build struct from fields
                struct_expr = pl.struct(field_exprs).alias(struct_col)
                df = df.with_columns(struct_expr)

                # Drop the flattened columns
                df = df.drop(all_flattened_cols)

        return df

    def _ensure_columns_exist(self, table_name: str, new_schema: Any) -> None:
        """Ensure all columns from new_schema exist in the table.

        Adds missing columns via ALTER TABLE to support schema evolution.

        Args:
            table_name: Name of the table to check
            new_schema: PyArrow schema with new columns
        """
        import adbc_driver_manager
        import pyarrow as pa

        stmt = adbc_driver_manager.AdbcStatement(self._conn)
        try:
            # Query the table to get its schema (just schema, no data)
            stmt.set_sql_query(f'SELECT * FROM "{table_name}" LIMIT 0')
            stream, _ = stmt.execute_query()
            reader = pa.RecordBatchReader.from_stream(stream)
            current_schema = reader.schema

            # Find columns that exist in new_schema but not in current_schema
            current_field_names = {field.name for field in current_schema}
            missing_fields = [field for field in new_schema if field.name not in current_field_names]

            if missing_fields:
                # Add missing columns using ALTER TABLE
                for field in missing_fields:
                    sqlite_type = self._arrow_to_sqlite_type(field.type)
                    alter_query = f'ALTER TABLE "{table_name}" ADD COLUMN "{field.name}" {sqlite_type}'

                    alter_stmt = adbc_driver_manager.AdbcStatement(self._conn)
                    try:
                        alter_stmt.set_sql_query(alter_query)
                        alter_stmt.execute_update()
                    finally:
                        alter_stmt.close()
        finally:
            stmt.close()

    def _arrow_to_sqlite_type(self, arrow_type: Any) -> str:
        """Convert Arrow type to SQLite type string.

        Args:
            arrow_type: PyArrow DataType

        Returns:
            SQLite type string
        """
        import pyarrow as pa

        # SQLite has limited types: NULL, INTEGER, REAL, TEXT, BLOB
        if pa.types.is_integer(arrow_type):
            return "INTEGER"
        elif pa.types.is_floating(arrow_type):
            return "REAL"
        elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return "TEXT"
        elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            return "BLOB"
        elif pa.types.is_boolean(arrow_type):
            return "INTEGER"  # SQLite stores booleans as 0/1
        elif pa.types.is_timestamp(arrow_type) or pa.types.is_date(arrow_type):
            return "TEXT"  # SQLite stores timestamps as text or integer
        else:
            # Default to TEXT for unknown types
            return "TEXT"

    def display(self) -> str:
        """Return a human-readable display string for this store."""
        if self.database == ":memory:":
            return "ADBCSQLiteMetadataStore(:memory:)"
        return f"ADBCSQLiteMetadataStore(database={self.database})"
