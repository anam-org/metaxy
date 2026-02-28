"""PostgreSQL metadata store with storage/compute separation.

Uses PostgreSQL for storage and Polars for versioning compute.
Metaxy system struct columns are round-tripped via JSON serialization.
User struct columns are JSON-encoded on write and decode behavior on read depends on
the effective SQL column type (JSON/JSONB).
Always uses PolarsVersioningEngine since PostgreSQL lacks native struct support.

Requirements:
    - PostgreSQL 12+ (JSONB support)
    - ibis-framework[postgres] package
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import ibis.expr.datatypes as dt
import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field

from metaxy._decorators import public
from metaxy._utils import collect_to_polars
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
    import ibis
    from ibis.expr.schema import Schema as IbisSchema

    from metaxy.metadata_store.base import MetadataStore

# Metaxy struct columns that need JSON-string ↔ Struct conversion
_METAXY_STRUCT_COLUMNS = frozenset({METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD})


def _column_contains_json_objects(series: pl.Series) -> bool:
    """Check whether a string series contains JSON object/array values.

    Inspects the first non-null value to decide if the column should be decoded
    from JSON to a Polars Struct. Returns False for empty, all-null, or plain
    string columns.
    """
    if len(series) == 0 or series.null_count() == len(series):
        return False
    first_value = series.drop_nulls()[0]
    if not isinstance(first_value, str):
        return False
    trimmed = first_value.lstrip()
    return len(trimmed) > 0 and trimmed[0] in ("{", "[")


@public
class PostgreSQLMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for PostgreSQLMetadataStore.

    Inherits connection_string, connection_params, table_prefix, auto_create_tables from IbisMetadataStoreConfig.
    """

    auto_cast_struct_for_jsonb: bool = Field(
        default=True,
        description="Auto-convert DataFrame Struct columns to JSON strings on write. Metaxy system columns are always converted.",
    )


@public
class PostgreSQLMetadataStore(IbisMetadataStore):
    """PostgreSQL metadata store with storage/compute separation.

    Uses PostgreSQL for storage and Polars for versioning.
    Filters push down to SQL WHERE clauses, then data materializes to Polars.

    Example:
        <!-- skip next -->
        ```python
        from metaxy.metadata_store.postgresql import PostgreSQLMetadataStore

        store = PostgreSQLMetadataStore(connection_string="postgresql://user:pass@localhost:5432/metaxy")

        with store:
            increment = store.resolve_update(MyFeature)
            store.write(MyFeature, increment.added)
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

    def _create_hash_functions(self):
        """Not needed — hashing is handled by PolarsVersioningEngine."""
        return {}

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for PostgreSQL (XXHASH32 computed in Polars)."""
        return HashAlgorithm.XXHASH32

    def _validate_hash_algorithm_support(self) -> None:
        """Validate hash algorithm against PolarsVersioningEngine's supported set."""
        from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError

        supported = PolarsVersioningEngine._HASH_FUNCTION_MAP
        if self.hash_algorithm not in supported:
            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm '{self.hash_algorithm.value}' not supported. "
                f"Supported algorithms: {', '.join(a.value for a in supported)}"
            )

    def _get_struct_columns_for_jsonb(self, schema: pl.Schema) -> list[str]:
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
        Stored SQL type depends on target table schema (JSON/JSONB vs TEXT).
        """
        # Materialize to Polars and identify struct columns
        pl_df = collect_to_polars(df)
        struct_columns = self._get_struct_columns_for_jsonb(pl_df.schema)

        # Convert struct to JSON string using Polars' json_encode()
        if struct_columns:
            transforms = [pl.col(col_name).struct.json_encode().alias(col_name) for col_name in struct_columns]
            pl_df = pl_df.with_columns(transforms)

        return nw.from_native(pl_df)

    def _get_json_columns_for_struct(self, ibis_schema: "IbisSchema") -> list[str]:
        """Identify which JSONB/TEXT columns should be converted back to Structs.

        Metaxy system columns (stored as TEXT) are always converted.
        When auto_cast_struct_for_jsonb is True, only dt.JSON columns are
        additionally converted for user columns. User struct data may be stored
        as JSON/JSONB or TEXT depending on table schema; this method only auto-
        decodes user columns when backend type information indicates JSON.
        """
        json_columns = []
        for col_name, dtype in ibis_schema.items():
            is_metaxy_column = col_name in _METAXY_STRUCT_COLUMNS
            if is_metaxy_column and isinstance(dtype, (dt.JSON, dt.String)):
                json_columns.append(col_name)
            elif self.auto_cast_struct_for_jsonb and isinstance(dtype, dt.JSON):
                json_columns.append(col_name)
        return json_columns

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
        if feature_key in graph.feature_definitions_by_key:
            definition = graph.feature_definitions_by_key[feature_key]
            field_names = [f.key.to_struct_key() for f in definition.spec.fields]

        # Build parse expressions and apply in one pass
        transforms: list[pl.Expr] = []
        for col_name in json_columns:
            if col_name not in pl_df.columns:
                continue
            if isinstance(pl_df.schema[col_name], pl.Struct):
                continue  # Already a struct

            # For metaxy system columns with known schema
            if col_name in _METAXY_STRUCT_COLUMNS and field_names:
                struct_schema = pl.Struct([pl.Field(name, pl.Utf8) for name in field_names])
                if len(pl_df) == 0 or pl_df[col_name].null_count() == len(pl_df):
                    # Empty/null data: construct struct with known schema
                    transforms.append(pl.lit(None, dtype=struct_schema).alias(col_name))
                else:
                    transforms.append(pl.col(col_name).str.json_decode(dtype=struct_schema).alias(col_name))
            elif _column_contains_json_objects(pl_df[col_name]):
                inferred_dtype = pl_df[col_name].str.json_decode().dtype
                transforms.append(pl.col(col_name).str.json_decode(dtype=inferred_dtype).alias(col_name))

        if transforms:
            pl_df = pl_df.with_columns(transforms)

        return pl_df

    def _read_feature(
        self,
        feature: CoercibleToFeatureKey,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read from PostgreSQL, materialize to Polars, and parse JSONB to Structs.

        Overrides the Ibis base to return a Polars-backed LazyFrame, which is
        required by PolarsVersioningEngine.keep_latest_by_group.
        SQL-level filters are applied before materialization.
        """
        feature_key = self._resolve_feature_key(feature)
        table_name = self.get_table_name(feature_key)

        # Delegate to Ibis base (applies SQL filters, column selection, transform_after_read)
        ibis_lazy_frame = super()._read_feature(
            feature, feature_version=feature_version, filters=filters, columns=columns, **kwargs
        )
        if ibis_lazy_frame is None:
            return None

        # Identify JSON-compatible columns from original schema before parse
        original_schema = self.conn.table(table_name).schema()
        json_columns_to_parse = self._get_json_columns_for_struct(original_schema)

        # Materialize to Polars and parse JSONB strings back to Structs
        pl_df = collect_to_polars(ibis_lazy_frame)
        pl_df = self._parse_json_to_struct_columns(pl_df, feature_key, json_columns_to_parse)

        return nw.from_native(pl_df.lazy())
