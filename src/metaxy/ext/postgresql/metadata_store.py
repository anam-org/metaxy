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

import json
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field

from metaxy._decorators import experimental, public
from metaxy.metadata_store.exceptions import TableNotFoundError
from metaxy.metadata_store.ibis import IbisMetadataStore, IbisMetadataStoreConfig
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.utils import collect_to_polars
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    import ibis
    from ibis.expr.schema import Schema as IbisSchema

    from metaxy.metadata_store.base import MetadataStore

# Metaxy struct columns that need JSON-string ↔ Struct conversion
_METAXY_STRUCT_COLUMNS = frozenset({METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD})


@public
@experimental
class PostgreSQLMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for PostgreSQLMetadataStore.

    Inherits connection_string, connection_params, table_prefix, auto_create_tables from IbisMetadataStoreConfig.
    """

    auto_cast_struct_for_jsonb: bool = Field(
        default=True,
        description="Whether to encode/decode Struct columns to/from JSON on writes/reads. Metaxy system columns are always converted.",
    )


@public
@experimental
class PostgreSQLMetadataStore(IbisMetadataStore):
    """PostgreSQL metadata store with storage/compute separation.

    Uses PostgreSQL for storage and Polars for versioning.
    Filters push down to SQL WHERE clauses, then data materializes to Polars.

    Example:
        <!-- skip next -->
        ```python
        from metaxy.ext.postgresql import PostgreSQLMetadataStore

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
            auto_cast_struct_for_jsonb: If True, JSON-encode all Struct columns to strings on write
                (Metaxy system Struct columns are always converted). The actual SQL column type
                (e.g., JSON, JSONB, or TEXT) is determined by the table schema, not this flag.
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

    @contextmanager
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[PolarsVersioningEngine]:
        """Create PostgreSQL versioning engine.

        PostgreSQLMetadataStore always uses PolarsVersioningEngine.
        """
        yield self.versioning_engine_cls(plan=plan)  # ty: ignore[invalid-yield]

    def _create_hash_functions(self):
        """Not needed — hashing is handled by PolarsVersioningEngine."""
        return {}

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for PostgreSQL (XXHASH32 computed in Polars)."""
        return HashAlgorithm.XXHASH32

    def _validate_hash_algorithm_support(self) -> None:
        """Validate hash algorithm against PolarsVersioningEngine's supported set."""
        from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError

        supported = PolarsVersioningEngine.supported_hash_algorithms()
        if self.hash_algorithm not in supported:
            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm '{self.hash_algorithm.value}' not supported. "
                f"Supported algorithms: {', '.join(a.value for a in sorted(supported, key=lambda a: a.value))}"
            )

    def _get_struct_columns_for_jsonb(self, schema: pl.Schema) -> list[str]:
        """Identify which Struct columns should be JSON-string serialized on write.

        The method name is historical: this step only chooses Struct columns to
        pass through ``struct.json_encode()`` before insert. The destination SQL
        type (``JSON``, ``JSONB``, ``TEXT``, etc.) is determined by the table
        schema, not by this method.

        Args:
            schema: Polars schema dict (column_name -> dtype)

        Returns:
            List of Struct column names to JSON-string serialize before insert.
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

    def _encode_struct_columns(self, pl_df: pl.DataFrame) -> pl.DataFrame:
        """JSON-encode configured Struct columns using Polars."""
        struct_columns = self._get_struct_columns_for_jsonb(pl_df.schema)
        if not struct_columns:
            return pl_df

        transforms = [pl.col(col_name).struct.json_encode().alias(col_name) for col_name in struct_columns]
        return pl_df.with_columns(transforms)

    def _build_auto_create_schema(
        self,
        original_pl_df: pl.DataFrame,
        transformed_pl_df: pl.DataFrame,
    ) -> "IbisSchema | None":
        """Override auto-created table types so encoded Struct columns become JSONB."""
        import ibis
        import ibis.expr.datatypes as dt

        jsonb_columns = set(self._get_struct_columns_for_jsonb(original_pl_df.schema))
        if not jsonb_columns:
            return None

        inferred_schema = ibis.memtable(transformed_pl_df).schema()
        return ibis.schema(
            {
                col_name: dt.JSON(binary=True) if col_name in jsonb_columns else dtype
                for col_name, dtype in inferred_schema.items()
            }
        )

    def _warn_auto_create_table(self, table_name: str) -> None:
        """Emit the standard auto-create warning."""
        if not self._should_warn_auto_create_tables:
            return

        warnings.warn(
            f"AUTO_CREATE_TABLES is enabled - automatically creating table '{table_name}'. "
            "Do not use in production! "
            "Use proper database migration tools like Alembic for production deployments.",
            UserWarning,
            stacklevel=4,
        )

    def _insert_feature_rows(
        self,
        table_name: str,
        pl_df: pl.DataFrame,
        target_schema: "IbisSchema",
    ) -> None:
        """Insert rows, adapting JSON and JSONB columns through psycopg when needed."""
        import ibis.expr.datatypes as dt
        from psycopg import sql
        from psycopg.types.json import Json, Jsonb

        json_wrappers = {
            col_name: Jsonb if dtype.binary else Json
            for col_name, dtype in target_schema.items()
            if col_name in pl_df.columns and isinstance(dtype, dt.JSON)
        }
        if not json_wrappers:
            self.conn.insert(table_name, obj=pl_df)  # ty: ignore[invalid-argument-type]
            return

        column_names = list(pl_df.columns)
        rows: list[tuple[Any, ...]] = []
        for row in pl_df.iter_rows(named=True):
            adapted_row: list[Any] = []
            for col_name in column_names:
                value = row[col_name]
                wrapper = json_wrappers.get(col_name)
                if wrapper is None or value is None:
                    adapted_row.append(value)
                    continue

                if isinstance(value, str):
                    adapted_row.append(wrapper(json.loads(value)))
                else:
                    adapted_row.append(wrapper(value))
            rows.append(tuple(adapted_row))

        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Identifier(col_name) for col_name in column_names),
            sql.SQL(", ").join(sql.Placeholder() for _ in column_names),
        )
        raw_connection = getattr(self.conn, "con")
        with raw_connection.cursor() as cursor:
            cursor.executemany(query, rows)

    def transform_before_write(self, df: Frame, feature_key: FeatureKey, table_name: str) -> Frame:
        """Convert Struct columns to JSON strings using Polars.

        Uses Polars' struct.json_encode() to serialize structs to JSON strings.
        Stored SQL type depends on target table schema (JSON/JSONB vs TEXT).
        """
        pl_df = collect_to_polars(df)
        return nw.from_native(self._encode_struct_columns(pl_df))

    def _write_feature(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Write feature metadata, preserving JSONB for auto-created Struct columns."""
        table_name = self.get_table_name(feature_key)
        original_pl_df = collect_to_polars(df)
        transformed_pl_df = self._encode_struct_columns(original_pl_df)

        if table_name not in self.conn.list_tables():
            if not self.auto_create_tables:
                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist for feature {feature_key.to_string()}. "
                    "Enable auto_create_tables=True to automatically create tables, "
                    "or use proper database migration tools like Alembic to create the table first."
                )

            auto_create_schema = self._build_auto_create_schema(original_pl_df, transformed_pl_df)
            self._warn_auto_create_table(table_name)
            if auto_create_schema is None:
                self.conn.create_table(table_name, obj=transformed_pl_df)
                return

            self.conn.create_table(table_name, schema=auto_create_schema)
            self._insert_feature_rows(table_name, transformed_pl_df, auto_create_schema)
            return

        target_schema = self.conn.table(table_name).schema()
        self._insert_feature_rows(table_name, transformed_pl_df, target_schema)

    def _get_json_columns_for_struct(self, ibis_schema: "IbisSchema") -> list[str]:
        """Identify which JSON/TEXT columns should be parsed back to Structs.

        Metaxy system columns are always candidates when the SQL type is JSON/TEXT.
        When ``auto_cast_struct_for_jsonb`` is enabled, user JSON/JSONB columns are
        also candidates and are parsed opportunistically in Polars using inference.
        User TEXT columns are excluded to avoid mutating arbitrary string columns
        that happen to contain JSON-looking payloads.
        """
        import ibis.expr.datatypes as dt

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
        """Parse JSON string columns back to Polars Structs on a per-column basis.

        Args:
            pl_df: Polars DataFrame with JSON string columns
            feature_key: Feature key for Metaxy system column schema lookup
            json_columns: List of column names to parse from JSON to Struct

        Returns:
            DataFrame where decodable object-shaped JSON columns are converted to Struct
        """
        if not json_columns:
            return pl_df

        for col_name in json_columns:
            if col_name not in pl_df.columns:
                continue
            if isinstance(pl_df.schema[col_name], pl.Struct):
                continue  # Already a struct

            col_values = pl_df[col_name]
            non_null_values = col_values.drop_nulls()
            if non_null_values.len() == 0:
                continue

            normalized = non_null_values.cast(pl.Utf8).str.strip_chars()
            is_object_json = normalized.str.starts_with("{") & normalized.str.ends_with("}")
            is_json_null_literal = normalized == "null"

            # Avoid decoding scalar/list/partially encoded payloads.
            if not (is_object_json | is_json_null_literal).all():
                continue

            try:
                decoded_col = col_values.str.json_decode(infer_schema_length=None)
            except pl.exceptions.PolarsError:
                continue

            if isinstance(decoded_col.dtype, pl.Struct):
                pl_df = pl_df.with_columns(decoded_col.alias(col_name))

        return pl_df

    def _cast_empty_system_struct_columns(
        self,
        pl_df: pl.DataFrame,
        feature_key: FeatureKey,
        json_columns: list[str],
    ) -> pl.DataFrame:
        """Restore known Metaxy Struct schemas for empty result sets.

        PostgreSQL stores Metaxy struct columns as JSON/TEXT. When a read returns
        zero rows, there is no payload available for JSON schema inference, so
        Polars leaves those columns as strings. For feature-backed reads we can
        still recover the expected struct schema from the active graph.
        """
        if pl_df.height != 0 or self._is_system_table(feature_key):
            return pl_df

        try:
            plan = self._resolve_feature_plan(feature_key)
        except (KeyError, RuntimeError):
            return pl_df

        field_names = [field_spec.key.to_struct_key() for field_spec in plan.feature.fields]
        expected_struct_dtype = pl.Struct({field_name: pl.String for field_name in field_names})

        casts = []
        for col_name in _METAXY_STRUCT_COLUMNS:
            if col_name not in json_columns or col_name not in pl_df.columns:
                continue
            if isinstance(pl_df.schema[col_name], pl.Struct):
                continue
            casts.append(pl.col(col_name).cast(expected_struct_dtype).alias(col_name))

        if not casts:
            return pl_df

        return pl_df.with_columns(casts)

    def _validate_required_system_struct_columns(
        self,
        pl_df: pl.DataFrame,
        feature_key: FeatureKey,
        json_columns: list[str],
        *,
        require_all_system_columns: bool = False,
    ) -> None:
        """Ensure required Metaxy system JSON columns decoded to Struct when payloads exist."""
        if require_all_system_columns:
            required_system_columns = list(_METAXY_STRUCT_COLUMNS)
        else:
            required_system_columns = [col_name for col_name in _METAXY_STRUCT_COLUMNS if col_name in json_columns]
        if require_all_system_columns:
            missing_required_columns = [
                col_name for col_name in required_system_columns if col_name not in pl_df.columns
            ]
            if missing_required_columns:
                missing_columns_csv = ", ".join(sorted(missing_required_columns))
                raise ValueError(
                    "Failed to decode or validate required Metaxy system JSON columns for "
                    f"feature {feature_key.to_string()}: {missing_columns_csv}. "
                    "Required system columns are missing from the result set."
                )

        # Empty result sets should not fail required-column schema inference.
        if pl_df.height == 0:
            return

        required_columns = [col_name for col_name in required_system_columns if col_name in pl_df.columns]

        def _has_no_struct_payload(col_name: str) -> bool:
            col_values = pl_df[col_name]
            non_null_values = col_values.drop_nulls()
            if non_null_values.len() == 0:
                return True
            if pl_df.schema.get(col_name) == pl.String:
                normalized = non_null_values.cast(pl.Utf8).str.strip_chars()
                return (normalized == "null").all()
            return False

        # All-null required system columns, including JSON literal `null`, represent no payload to validate.
        if required_columns and all(_has_no_struct_payload(col_name) for col_name in required_columns):
            return

        invalid_columns: list[str] = []

        for col_name in required_columns:
            dtype = pl_df.schema.get(col_name)
            if not isinstance(dtype, pl.Struct):
                invalid_columns.append(col_name)
                continue

            if len(dtype.fields) == 0:
                invalid_columns.append(col_name)

        if invalid_columns:
            invalid_columns_csv = ", ".join(sorted(set(invalid_columns)))
            raise ValueError(
                "Failed to decode or validate required Metaxy system JSON columns for "
                f"feature {feature_key.to_string()}: {invalid_columns_csv}. "
                "Required system columns must contain decodable JSON object payloads."
            )

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

        SQL-level filters are applied before materialization.

        Note:
            PostgreSQL JSON decoding requires eager materialization to Polars.
            This method therefore executes the SQL query immediately, parses JSON
            columns eagerly, and then returns a Polars-backed LazyFrame over that
            in-memory snapshot (not a deferred database query).
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
        pl_df = self._cast_empty_system_struct_columns(pl_df, feature_key, json_columns_to_parse)
        self._validate_required_system_struct_columns(
            pl_df,
            feature_key,
            json_columns_to_parse,
            require_all_system_columns=columns is None and not self._is_system_table(feature_key),
        )

        return nw.from_native(pl_df.lazy())
