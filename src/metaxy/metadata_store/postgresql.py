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

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

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
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    import ibis
    from ibis.expr.schema import Schema as IbisSchema

    from metaxy.metadata_store.base import MetadataStore

# Metaxy struct columns that need JSON-string ↔ Struct conversion
_METAXY_STRUCT_COLUMNS = frozenset({METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD})


@public
class PostgreSQLMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for PostgreSQLMetadataStore.

    Inherits connection_string, connection_params, table_prefix, auto_create_tables from IbisMetadataStoreConfig.
    """

    auto_cast_struct_for_jsonb: bool = Field(
        default=True,
        description="Auto-convert DataFrame Struct columns to JSON strings on write. Metaxy system columns are always converted.",
    )
    decode_user_text_json_columns: bool = Field(
        default=False,
        description="Backward-compatibility toggle: when enabled, auto-cast user TEXT columns containing JSON objects back to Struct on read.",
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
        decode_user_text_json_columns: bool = False,
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
            decode_user_text_json_columns: If True, user TEXT columns are also considered
                JSON decode candidates on read (legacy behavior). Default False keeps
                user TEXT values as strings.
            **kwargs: Additional arguments passed to IbisMetadataStore
        """
        if connection_string is None and connection_params is None:
            raise ValueError("Must provide either connection_string or connection_params for PostgreSQL")

        self.auto_cast_struct_for_jsonb = auto_cast_struct_for_jsonb
        self.decode_user_text_json_columns = decode_user_text_json_columns

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
        yield self.versioning_engine_cls(plan=plan)

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
        """Identify which JSON/TEXT columns should be parsed back to Structs.

        Metaxy system columns are always candidates when the SQL type is JSON/TEXT.
        When ``auto_cast_struct_for_jsonb`` is enabled, user JSON/JSONB columns are
        also candidates and are parsed opportunistically in Polars using inference.
        User TEXT columns are excluded by default to avoid mutating arbitrary string
        columns that happen to contain JSON-looking payloads, but can be re-enabled
        via ``decode_user_text_json_columns`` for backward compatibility.
        """
        import ibis.expr.datatypes as dt

        json_columns = []
        for col_name, dtype in ibis_schema.items():
            is_metaxy_column = col_name in _METAXY_STRUCT_COLUMNS
            if is_metaxy_column and isinstance(dtype, (dt.JSON, dt.String)):
                json_columns.append(col_name)
            elif self.auto_cast_struct_for_jsonb and isinstance(dtype, dt.JSON):
                json_columns.append(col_name)
            elif (
                self.auto_cast_struct_for_jsonb and self.decode_user_text_json_columns and isinstance(dtype, dt.String)
            ):
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

            target_dtype = self._get_target_struct_dtype_for_column(feature_key, col_name)
            col_values = pl_df[col_name]
            non_null_values = col_values.drop_nulls()
            if non_null_values.len() == 0:
                if target_dtype is not None:
                    pl_df = pl_df.with_columns(
                        pl.Series(
                            name=col_name,
                            values=[None] * len(pl_df),
                            dtype=target_dtype,
                        )
                    )
                continue

            normalized = non_null_values.cast(pl.Utf8).str.strip_chars()
            is_object_json = normalized.str.starts_with("{") & normalized.str.ends_with("}")
            is_json_null_literal = normalized == "null"

            # Avoid decoding scalar/list/partially encoded payloads.
            if not (is_object_json | is_json_null_literal).all():
                continue

            try:
                if target_dtype is None:
                    decoded_col = col_values.str.json_decode(infer_schema_length=None)
                else:
                    decoded_col = col_values.str.json_decode(dtype=target_dtype, infer_schema_length=None)
            except pl.exceptions.PolarsError:
                continue

            if isinstance(decoded_col.dtype, pl.Struct):
                pl_df = pl_df.with_columns(decoded_col.alias(col_name))
                continue
            if decoded_col.dtype == pl.Null and target_dtype is not None:
                pl_df = pl_df.with_columns(
                    pl.Series(
                        name=col_name,
                        values=[None] * len(pl_df),
                        dtype=target_dtype,
                    )
                )

        return pl_df

    def _get_expected_system_struct_field_names(self, feature_key: FeatureKey) -> list[str] | None:
        """Get expected system struct field names for a feature when graph context is available."""
        try:
            from metaxy.models.feature import current_graph

            definition = current_graph().get_feature_definition(feature_key)
        except (KeyError, RuntimeError):
            return None

        return [field.key.to_struct_key() for field in definition.spec.fields]

    def _get_target_struct_dtype_for_column(self, feature_key: FeatureKey, column_name: str) -> pl.Struct | None:
        """Get target Struct dtype for JSON decode fallback paths.

        For Metaxy system struct columns, use feature fields from the active graph when
        available so all-NULL columns preserve expected field names. For user columns (or
        missing graph context), return None so read-path decoding does not force a schema.
        """
        if column_name not in _METAXY_STRUCT_COLUMNS:
            return None

        field_names = self._get_expected_system_struct_field_names(feature_key)
        if not field_names:
            return None

        return pl.Struct([pl.Field(field_name, pl.Utf8) for field_name in field_names])

    def _validate_required_system_struct_columns(
        self,
        pl_df: pl.DataFrame,
        feature_key: FeatureKey,
        json_columns: list[str],
        *,
        require_all_system_columns: bool = False,
    ) -> None:
        """Ensure required Metaxy system JSON columns were decoded to Struct with valid fields."""
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

        # All-null required system columns also represent "no payload to validate".
        if required_columns and all(pl_df[col_name].drop_nulls().len() == 0 for col_name in required_columns):
            return

        expected_field_names = self._get_expected_system_struct_field_names(feature_key)
        expected_field_name_set = frozenset(expected_field_names) if expected_field_names is not None else None
        schema_inference_failed = False

        if expected_field_names is None:
            inferred_field_name_set: frozenset[str] | None = None
            for col_name in required_columns:
                dtype = pl_df.schema.get(col_name)
                if not isinstance(dtype, pl.Struct) or len(dtype.fields) == 0:
                    schema_inference_failed = True
                    break

                current_field_name_set = frozenset(field.name for field in dtype.fields)
                if inferred_field_name_set is None:
                    inferred_field_name_set = current_field_name_set
                elif current_field_name_set != inferred_field_name_set:
                    schema_inference_failed = True
                    break

            if not schema_inference_failed and inferred_field_name_set is not None:
                # Normalize to a canonical set so struct-field order never impacts validation.
                expected_field_name_set = frozenset(inferred_field_name_set)
            else:
                schema_inference_failed = True

        invalid_columns: list[str] = []

        for col_name in required_columns:
            dtype = pl_df.schema.get(col_name)
            if not isinstance(dtype, pl.Struct):
                invalid_columns.append(col_name)
                continue

            actual_field_name_set = frozenset(field.name for field in dtype.fields)
            if not actual_field_name_set:
                invalid_columns.append(col_name)
                continue

            if expected_field_name_set is not None:
                missing_expected_fields = expected_field_name_set - actual_field_name_set
                unexpected_fields = actual_field_name_set - expected_field_name_set
                if missing_expected_fields or unexpected_fields:
                    invalid_columns.append(col_name)

        if schema_inference_failed:
            invalid_columns.extend(col_name for col_name in required_columns if col_name not in invalid_columns)

        if invalid_columns:
            invalid_columns_csv = ", ".join(sorted(set(invalid_columns)))
            reason = (
                "Could not determine expected required-field schema from feature definition "
                "or decoded system-column payloads."
                if schema_inference_failed
                else "Required system columns must contain decodable JSON object payloads with the expected field keys."
            )
            raise ValueError(
                "Failed to decode or validate required Metaxy system JSON columns for "
                f"feature {feature_key.to_string()}: {invalid_columns_csv}. "
                f"{reason}"
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
        self._validate_required_system_struct_columns(
            pl_df,
            feature_key,
            json_columns_to_parse,
            require_all_system_columns=columns is None and not self._is_system_table(feature_key),
        )

        return nw.from_native(pl_df.lazy())
