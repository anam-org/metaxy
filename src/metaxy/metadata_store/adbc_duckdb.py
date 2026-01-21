"""DuckDB metadata store using ADBC for high-performance bulk ingestion.

Uses Arrow Database Connectivity (ADBC) for zero-copy data transfer and bulk writes.
Provides excellent write performance for DuckDB with built-in ADBC support.

Note: DuckDB's ADBC driver is built into the main duckdb package.
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


class ADBCDuckDBMetadataStoreConfig(ADBCMetadataStoreConfig):
    """Configuration for ADBC DuckDB metadata store.

    Example:
        ```python
        config = ADBCDuckDBMetadataStoreConfig(
            database="metadata.duckdb",
        )
        ```
    """

    database: str | Path = Field(
        description="Database file path or :memory: for in-memory database.",
    )


class ADBCDuckDBMetadataStore(ADBCMetadataStore):
    """
    DuckDB metadata store using ADBC for high-performance bulk writes.

    Provides high-performance metadata storage using DuckDB with:
    - Zero-copy Arrow data transfer via ADBC
    - Bulk ingestion optimized for DuckDB
    - File-based and in-memory storage
    - Built-in ADBC driver (no external dependencies)

    Example:
        ```python
        from metaxy.metadata_store.adbc_duckdb import ADBCDuckDBMetadataStore

        # File-based storage
        store = ADBCDuckDBMetadataStore(
            database="metadata.duckdb",
            hash_algorithm=HashAlgorithm.XXHASH64,
        )

        # In-memory database (for testing)
        store = ADBCDuckDBMetadataStore(
            database=":memory:",
        )
        ```
    """

    def __init__(
        self,
        database: str | Path,
        **kwargs: Any,
    ) -> None:
        """Initialize ADBC DuckDB metadata store.

        Args:
            database: Database file path or :memory: for in-memory database.
                Examples: "metadata.duckdb", Path("metadata.duckdb"), ":memory:"
            **kwargs: Passed to ADBCMetadataStore (e.g., hash_algorithm).
        """
        # Convert Path to string
        database_str = str(database)

        # DuckDB ADBC driver expects database path directly
        super().__init__(connection_params={"database": database_str}, **kwargs)

        self.database = database_str

        # DuckDB supports xxhash64 natively via hashfuncs extension
        supported_algorithms = {HashAlgorithm.XXHASH64, HashAlgorithm.MD5}
        if self.hash_algorithm not in supported_algorithms:
            raise HashAlgorithmNotSupportedError(
                f"ADBCDuckDBMetadataStore currently supports XXHASH64 and MD5. Requested: {self.hash_algorithm}"
            )

    @classmethod
    def config_model(cls) -> type[ADBCMetadataStoreConfig]:
        """Return the configuration model for this store type."""
        return ADBCDuckDBMetadataStoreConfig

    def _get_driver_name(self) -> str:
        """Get the ADBC driver path for DuckDB."""
        import adbc_driver_duckdb

        return adbc_driver_duckdb.driver_path()

    def _get_connection_options(self) -> dict[str, Any]:
        """Get DuckDB-specific connection options for ADBC."""
        options: dict[str, Any] = {}

        # DuckDB ADBC driver requires specific entrypoint
        options["entrypoint"] = "duckdb_adbc_init"

        # ADBC DuckDB driver uses 'path' parameter (not 'uri')
        if self.connection_params:
            # Map 'database' to 'path' for ADBC driver
            if "database" in self.connection_params:
                db_path = self.connection_params["database"]
                # Convert Path to string if necessary
                options["path"] = str(db_path) if db_path != ":memory:" else ":memory:"
            else:
                # Copy other params but rename 'database' to 'path'
                for key, value in self.connection_params.items():
                    if key == "database":
                        options["path"] = str(value) if value != ":memory:" else ":memory:"
                    else:
                        options[key] = value
        elif self.connection_string:
            # If connection string was provided, use it as path
            options["path"] = self.connection_string

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
        # This matches the PostgreSQL/SQLite approach and preserves type information
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

            # CRITICAL: DuckDB ADBC APPEND mode requires columns in exact same order as table
            # Reorder arrow_table columns to match existing table schema
            arrow_table = self._reorder_columns_to_match_table(table_name, arrow_table)

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

            # Bind Arrow table directly (DuckDB ADBC supports Arrow Tables)
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
        """Read metadata from DuckDB via ADBC.

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

        # Query via ADBC statement (filters will be applied by Narwhals after reading)
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

            # Normalize datetime columns to UTC
            # DuckDB may return timestamps with local timezone
            result = self._normalize_datetime_columns(result)

            # Reconstruct struct columns from flattened columns
            # This preserves ALL fields across schema changes
            result = self._reconstruct_struct_columns(result, feature_key)

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

        # Query information_schema to check table existence
        query = f"""
            SELECT COUNT(*) as cnt FROM information_schema.tables
            WHERE table_name = '{table_name}'
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

            # Get the count result
            result = arrow_table.to_pylist()
            if result and len(result) > 0:
                return bool(result[0]["cnt"] > 0)
            return False
        finally:
            stmt.close()

    def _normalize_datetime_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize datetime columns to UTC.

        DuckDB may return timestamps with local timezone. This ensures all
        datetime columns are converted to UTC for consistency.

        Args:
            df: Polars DataFrame

        Returns:
            DataFrame with datetime columns normalized to UTC
        """
        from metaxy.models.constants import METAXY_CREATED_AT, METAXY_DELETED_AT

        datetime_columns = [METAXY_CREATED_AT, METAXY_DELETED_AT]

        for col in datetime_columns:
            if col not in df.columns:
                continue

            col_dtype = df[col].dtype

            # If it's already UTC datetime, skip
            if col_dtype == pl.Datetime(time_unit="us", time_zone="UTC"):
                continue

            # If it's a datetime with a different timezone, convert to UTC
            if isinstance(col_dtype, pl.Datetime):
                df = df.with_columns(pl.col(col).dt.convert_time_zone("UTC").alias(col))

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

            # Get schema from stream
            reader = pa.RecordBatchReader.from_stream(stream)
            try:
                current_schema = reader.schema

                # Find columns that exist in new_schema but not in current_schema
                current_field_names = {field.name for field in current_schema}
                missing_fields = [field for field in new_schema if field.name not in current_field_names]

                if missing_fields:
                    # Add missing columns using ALTER TABLE
                    for field in missing_fields:
                        duckdb_type = self._arrow_to_duckdb_type(field.type)
                        alter_query = f'ALTER TABLE "{table_name}" ADD COLUMN "{field.name}" {duckdb_type}'

                        alter_stmt = adbc_driver_manager.AdbcStatement(self._conn)
                        try:
                            alter_stmt.set_sql_query(alter_query)
                            alter_stmt.execute_update()
                        finally:
                            alter_stmt.close()
            finally:
                # Close the reader to release resources
                reader.close()
        finally:
            stmt.close()

    def _reorder_columns_to_match_table(self, table_name: str, arrow_table: Any) -> Any:
        """Reorder Arrow table columns to match existing table schema.

        DuckDB ADBC APPEND mode requires columns in the exact same order as the table.
        This method queries the table schema and reorders the Arrow table columns to match.

        Args:
            table_name: Name of the table
            arrow_table: PyArrow Table to reorder

        Returns:
            PyArrow Table with columns reordered to match table schema
        """
        import adbc_driver_manager
        import pyarrow as pa

        # Query table schema
        stmt = adbc_driver_manager.AdbcStatement(self._conn)
        try:
            stmt.set_sql_query(f'SELECT * FROM "{table_name}" LIMIT 0')
            stream, _ = stmt.execute_query()

            reader = pa.RecordBatchReader.from_stream(stream)
            try:
                table_schema = reader.schema
                table_column_names = [field.name for field in table_schema]

                # Reorder arrow_table columns to match table schema
                # Include only columns that exist in both schemas
                arrow_column_names = set(arrow_table.column_names)
                reordered_columns = [col for col in table_column_names if col in arrow_column_names]

                # Select columns in the correct order
                return arrow_table.select(reordered_columns)
            finally:
                reader.close()
        finally:
            stmt.close()

    def _arrow_to_duckdb_type(self, arrow_type: Any) -> str:
        """Convert Arrow type to DuckDB type string.

        Args:
            arrow_type: PyArrow DataType

        Returns:
            DuckDB type string
        """
        import pyarrow as pa

        if pa.types.is_integer(arrow_type):
            if arrow_type == pa.int8():
                return "TINYINT"
            elif arrow_type == pa.int16():
                return "SMALLINT"
            elif arrow_type == pa.int32():
                return "INTEGER"
            elif arrow_type == pa.int64():
                return "BIGINT"
            elif arrow_type == pa.uint8():
                return "UTINYINT"
            elif arrow_type == pa.uint16():
                return "USMALLINT"
            elif arrow_type == pa.uint32():
                return "UINTEGER"
            elif arrow_type == pa.uint64():
                return "UBIGINT"
            return "BIGINT"
        elif pa.types.is_floating(arrow_type):
            if arrow_type == pa.float32():
                return "FLOAT"
            return "DOUBLE"
        elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return "VARCHAR"
        elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            return "BLOB"
        elif pa.types.is_boolean(arrow_type):
            return "BOOLEAN"
        elif pa.types.is_timestamp(arrow_type):
            return "TIMESTAMP"
        elif pa.types.is_date(arrow_type):
            return "DATE"
        elif pa.types.is_time(arrow_type):
            return "TIME"
        else:
            # Default to VARCHAR for unknown types
            return "VARCHAR"

    def display(self) -> str:
        """Return a human-readable display string for this store."""
        if self.database == ":memory:":
            return "ADBCDuckDBMetadataStore(:memory:)"
        return f"ADBCDuckDBMetadataStore(database={self.database})"
