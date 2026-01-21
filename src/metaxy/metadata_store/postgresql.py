"""PostgreSQL metadata store with storage/compute separation.

Uses PostgreSQL for storage (JSONB for struct columns) and Polars for versioning compute.
Always uses PolarsVersioningEngine since PostgreSQL lacks native struct support.

Requirements:
    - PostgreSQL 12+ (JSONB support)
    - ibis-framework[postgres] package
"""

from typing import TYPE_CHECKING, Any

import ibis
import ibis.expr.datatypes as dt
import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field

from metaxy._public import public
from metaxy._utils import collect_to_polars
from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.ibis import IbisMetadataStore, IbisMetadataStoreConfig
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    pass

# Metaxy struct columns that need JSONB ↔ Struct conversion
_METAXY_STRUCT_COLUMNS = frozenset({METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD})


@public
class PostgreSQLMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for PostgreSQLMetadataStore.

    Inherits connection_string, connection_params, table_prefix, auto_create_tables from IbisMetadataStoreConfig.
    """

    auto_cast_struct_for_jsonb: bool = Field(
        default=True,
        description="Auto-convert DataFrame Struct columns to JSONB on write. Metaxy system columns are always converted.",
    )

    @classmethod
    def config_model(cls) -> type["PostgreSQLMetadataStoreConfig"]:
        """Return the config model for this metadata store."""
        return PostgreSQLMetadataStoreConfig


@public
class PostgreSQLMetadataStore(IbisMetadataStore):
    """PostgreSQL metadata store with storage/compute separation.

    Uses PostgreSQL for storage (JSONB for struct columns) and Polars for versioning.
    Filters push down to SQL WHERE clauses, then data materializes to Polars.

    Example:
        ```python
        from metaxy.metadata_store.postgresql import PostgreSQLMetadataStore

        store = PostgreSQLMetadataStore(connection_string="postgresql://user:pass@localhost:5432/metaxy")

        with store:
            increment = store.resolve_update(MyFeature)
            store.write_metadata(MyFeature, increment.added)
        ```
    """

    # Override to use Polars versioning engine (PostgreSQL has no native struct support)
    versioning_engine_cls = PolarsVersioningEngine

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list["MetadataStore"] | None = None,
        auto_cast_struct_for_jsonb: bool = True,
        **kwargs: Any,
    ):
        """Initialize PostgreSQL metadata store.

        Args:
            connection_string: PostgreSQL connection URI
                (e.g., "postgresql://user:pass@host:5432/db")
            connection_params: Dict with keys: host, port, database, user, password
                (alternative to connection_string)
            fallback_stores: List of fallback stores for chaining
            auto_cast_struct_for_jsonb: If True, convert all Struct columns to JSONB.
                If False, only convert metaxy system columns.
            **kwargs: Additional arguments passed to IbisMetadataStore
        """
        if connection_string is None and connection_params is None:
            raise ValueError("Must provide either connection_string or connection_params for PostgreSQL")

        self.auto_cast_struct_for_jsonb = auto_cast_struct_for_jsonb

        super().__init__(
            connection_string=connection_string,
            backend="postgres",
            connection_params=connection_params,
            fallback_stores=fallback_stores,
            **kwargs,
        )

    @classmethod
    def config_model(cls) -> type[PostgreSQLMetadataStoreConfig]:
        """Return the config model for this metadata store."""
        return PostgreSQLMetadataStoreConfig

    def native_implementation(self) -> nw.Implementation:
        """Force Polars implementation for versioning operations."""
        return nw.Implementation.POLARS

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for PostgreSQL (XXHASH64 computed in Polars)."""
        return HashAlgorithm.XXHASH64

    def _validate_hash_algorithm_support(self) -> None:
        """Skip hash algorithm validation - PolarsVersioningEngine validates at runtime.

        PostgreSQL uses PolarsVersioningEngine which validates hash algorithms
        when the engine is created. No need for store-level validation.
        """
        # No validation needed - PolarsVersioningEngine handles it
        pass

    def _get_struct_columns_for_jsonb(self, schema: dict[str, Any]) -> list[str]:
        """Identify which struct columns should be converted to JSONB.

        Args:
            schema: Polars schema dict (column_name -> dtype)

        Returns:
            List of column names that should be converted to JSONB
        """
        if self.auto_cast_struct_for_jsonb:
            # Convert ALL Struct columns
            return [col_name for col_name, col_dtype in schema.items() if isinstance(col_dtype, pl.Struct)]
        else:
            # Only convert metaxy system columns
            return [
                col_name
                for col_name in _METAXY_STRUCT_COLUMNS
                if col_name in schema and isinstance(schema[col_name], pl.Struct)
            ]

    def transform_before_write(self, df: Frame, feature_key: FeatureKey, table_name: str) -> Frame:
        """Convert Struct columns to JSON strings using Polars.

        Uses Polars' struct.json_encode() to serialize structs to JSON strings.
        The actual JSONB casting for PostgreSQL is handled in write_metadata_to_store.
        """
        # Materialize to Polars and identify struct columns
        pl_df = collect_to_polars(df)
        struct_columns = self._get_struct_columns_for_jsonb(pl_df.schema)

        # Convert struct to JSON string using Polars' json_encode()
        if struct_columns:
            transforms = [pl.col(col_name).struct.json_encode().alias(col_name) for col_name in struct_columns]
            pl_df = pl_df.with_columns(transforms)

        return nw.from_native(pl_df)

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Write metadata ensuring JSONB types for struct columns.

        Transforms are done via transform_before_write (Polars' struct.json_encode()).
        This method handles PostgreSQL-specific JSONB type casting using raw SQL.
        """
        import warnings

        from metaxy.metadata_store.exceptions import TableNotFoundError

        table_name = self.get_table_name(feature_key)

        # Identify struct columns BEFORE transformation
        pl_df_original = collect_to_polars(df)
        struct_columns = self._get_struct_columns_for_jsonb(pl_df_original.schema)

        # Apply Polars transformations (struct -> JSON string)
        df = self.transform_before_write(df, feature_key, table_name)
        pl_df = collect_to_polars(df)

        # Check if table exists
        table_exists = table_name in self.conn.list_tables()

        # Create table with JSONB types if needed
        if not table_exists:
            if not self.auto_create_tables:
                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist for feature {feature_key.to_string()}. "
                    f"Enable auto_create_tables=True to automatically create tables, "
                    f"or use proper database migration tools like Alembic to create the table first."
                )

            if self._should_warn_auto_create_tables:
                warnings.warn(
                    f"AUTO_CREATE_TABLES is enabled - automatically creating table '{table_name}'. "
                    "Do not use in production! "
                    "Use proper database migration tools like Alembic for production deployments.",
                    UserWarning,
                    stacklevel=4,
                )

            if struct_columns:
                # Create table with explicit JSONB types using raw SQL
                memtable = ibis.memtable(pl_df)
                col_defs = []
                for col_name in pl_df.columns:
                    if col_name in struct_columns:
                        col_defs.append(f'"{col_name}" JSONB')
                    else:
                        col_type = memtable.schema()[col_name]
                        pg_type = self._ibis_to_pg_type(col_type)
                        col_defs.append(f'"{col_name}" {pg_type}')

                create_sql = f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})'
                self.conn.raw_sql(create_sql)  # type: ignore
            else:
                # No struct columns - create empty table, data inserted below
                # Don't pass obj=pl_df to avoid double-insert
                memtable = ibis.memtable(pl_df)
                self.conn.create_table(table_name, schema=memtable.schema(), overwrite=False)

        # Insert data using PostgreSQL COPY for high performance
        if len(pl_df) > 0:
            if struct_columns:
                # Use COPY command for 10x-50x faster bulk inserts
                # Polars writes CSV, PostgreSQL auto-casts strings to JSONB
                from io import BytesIO

                # Write to in-memory CSV buffer without header
                csv_buffer = BytesIO()
                pl_df.write_csv(csv_buffer, include_header=False)
                csv_buffer.seek(0)

                # Use COPY for bulk insert (much faster than INSERT for large datasets)
                cols = ", ".join(f'"{col}"' for col in pl_df.columns)
                copy_sql = f'COPY "{table_name}" ({cols}) FROM STDIN (FORMAT CSV)'

                with self.conn.con.cursor() as cursor:  # type: ignore
                    with cursor.copy(copy_sql) as copy:
                        copy.write(csv_buffer.read())
            else:
                # No struct columns - use regular Ibis insert
                self.conn.insert(table_name, obj=pl_df)  # type: ignore

    def _ibis_to_pg_type(self, ibis_type: dt.DataType) -> str:
        """Convert Ibis data type to PostgreSQL type string."""
        if isinstance(ibis_type, dt.Int64):
            return "BIGINT"
        elif isinstance(ibis_type, dt.Int32):
            return "INTEGER"
        elif isinstance(ibis_type, dt.Float64):
            return "DOUBLE PRECISION"
        elif isinstance(ibis_type, dt.String):
            return "TEXT"
        elif isinstance(ibis_type, dt.Boolean):
            return "BOOLEAN"
        elif isinstance(ibis_type, dt.Timestamp):
            # Use TIMESTAMPTZ to preserve timezone information
            # PostgreSQL TIMESTAMP (without TZ) loses timezone, but TIMESTAMPTZ preserves it
            if ibis_type.timezone:
                return "TIMESTAMPTZ"
            else:
                return "TIMESTAMP"
        elif isinstance(ibis_type, dt.Date):
            return "DATE"
        elif isinstance(ibis_type, dt.JSON):
            return "JSONB"
        else:
            # Default to TEXT for unknown types
            return "TEXT"

    def _get_json_columns_for_struct(self, ibis_schema: Any) -> list[str]:
        """Identify which JSONB/TEXT columns should be converted back to Structs.

        Only converts columns that are actually JSONB (JSON) or TEXT in PostgreSQL.
        Metaxy system columns are always converted if they exist.

        Args:
            ibis_schema: Ibis schema object

        Returns:
            List of column names that should be converted from JSONB/TEXT to Struct
        """
        json_or_text_columns = []

        for col_name, dtype in ibis_schema.items():
            is_json_or_text = isinstance(dtype, (dt.JSON, dt.String))
            is_metaxy_column = col_name in _METAXY_STRUCT_COLUMNS

            # Always convert metaxy system columns if they're JSON or TEXT
            if is_metaxy_column and is_json_or_text:
                json_or_text_columns.append(col_name)
            # Convert user columns only if auto_cast is enabled and they're JSON or TEXT
            elif self.auto_cast_struct_for_jsonb and is_json_or_text:
                # Only convert if it looks like a struct column (ends with _by_field)
                # This avoids converting arbitrary text columns
                if col_name.endswith("_by_field"):
                    json_or_text_columns.append(col_name)

        return json_or_text_columns

    def transform_after_read(self, table: "ibis.Table", feature_key: FeatureKey) -> "ibis.Table":
        """Cast JSONB columns to String so Polars can parse them back to Structs.

        PostgreSQL JSONB columns appear as JSON type in Ibis. We cast to String
        so that when materialized to Polars, we can parse them back to Structs.
        """
        schema = table.schema()
        json_columns = self._get_json_columns_for_struct(schema)

        if json_columns:
            mutations = {col_name: table[col_name].cast("string") for col_name in json_columns}
            return table.mutate(**mutations)
        return table

    def _parse_json_to_struct_columns(
        self, pl_df: pl.DataFrame, feature_key: FeatureKey, json_columns: list[str]
    ) -> pl.DataFrame:
        """Parse specified JSON string columns back to Polars Structs.

        Args:
            pl_df: Polars DataFrame with JSON string columns
            feature_key: Feature key for getting field names
            json_columns: List of column names to parse from JSON to Struct

        Returns:
            DataFrame with JSON strings parsed to Structs
        """
        if not json_columns:
            return pl_df

        # Get feature spec for struct schema (for metaxy system columns)
        graph = FeatureGraph.get_active()
        field_names: list[str] = []
        if feature_key in graph.features_by_key:
            feature_cls = graph.features_by_key[feature_key]
            field_names = [f.key.to_struct_key() for f in feature_cls.spec().fields]

        # Parse each JSON string column
        for col_name in json_columns:
            if col_name not in pl_df.columns:
                continue
            if isinstance(pl_df.schema[col_name], pl.Struct):
                continue  # Already a struct

            # For metaxy system columns with known schema
            if col_name in _METAXY_STRUCT_COLUMNS and field_names:
                if len(pl_df) == 0 or pl_df[col_name].null_count() == len(pl_df):
                    # Empty/null data: construct struct with known schema
                    struct_schema = pl.Struct([pl.Field(name, pl.Utf8) for name in field_names])
                    pl_df = pl_df.with_columns(pl.lit(None, dtype=struct_schema).alias(col_name))
                else:
                    # Parse JSON string to struct
                    pl_df = pl_df.with_columns(pl_df[col_name].str.json_decode().alias(col_name))
            else:
                # User columns: simple JSON decode
                pl_df = pl_df.with_columns(pl_df[col_name].str.json_decode().alias(col_name))

        return pl_df

    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read from PostgreSQL and materialize to Polars with struct parsing.

        Override to ensure data is Polars-backed and JSONB columns are parsed to Structs.
        """
        feature_key = self._resolve_feature_key(feature)
        table_name = self.get_table_name(feature_key)

        # Check if table exists
        if table_name not in self.conn.list_tables():
            return None

        # Get original table schema to identify JSONB/TEXT columns
        table = self.conn.table(table_name)
        original_schema = table.schema()
        json_columns_to_parse = self._get_json_columns_for_struct(original_schema)

        # Read from Ibis (JSONB columns are cast to String in transform_after_read)
        ibis_lazy_frame = super().read_metadata_in_store(feature, **kwargs)

        # If reading failed, return None
        if ibis_lazy_frame is None:
            return None

        # Materialize to Polars and parse only the identified JSON columns back to Structs
        pl_df = collect_to_polars(ibis_lazy_frame)
        pl_df = self._parse_json_to_struct_columns(pl_df, feature_key, json_columns_to_parse)

        return nw.from_native(pl_df.lazy())
