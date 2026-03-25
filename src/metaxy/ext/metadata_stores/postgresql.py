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

from __future__ import annotations

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
from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.exceptions import (
    HashAlgorithmNotSupportedError,
    TableNotFoundError,
)
from metaxy.metadata_store.ibis import IbisMetadataStoreConfig
from metaxy.metadata_store.ibis_compute_engine import (
    IbisComputeEngine,
    IbisSQLHandler,
    IbisStorageConfig,
    IbisStoreBackcompat,
)
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.system.keys import METAXY_SYSTEM_KEY_PREFIX
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.feature import current_graph
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey
from metaxy.utils import collect_to_polars
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    import ibis
    from ibis.backends.sql import SQLBackend
    from ibis.expr.schema import Schema as IbisSchema

# Metaxy struct columns that need JSON-string <-> Struct conversion
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


class PostgreSQLSQLHandler(IbisSQLHandler):
    """SQL storage handler for PostgreSQL with JSON/Struct round-tripping.

    Uses PostgreSQL for storage and Polars for versioning.

    Example:
        <!-- skip next -->
        ```python
        from metaxy.ext.metadata_stores.postgresql import PostgreSQLMetadataStore

        store = PostgreSQLMetadataStore(connection_string="postgresql://user:pass@localhost:5432/metaxy")

        with store:
            increment = store.resolve_update(MyFeature)
            store.write(MyFeature, increment.added)
        ```
    """

    def __init__(self, *, auto_create_tables: bool = False, auto_cast_struct_for_jsonb: bool = True) -> None:
        super().__init__(auto_create_tables=auto_create_tables)
        self.auto_cast_struct_for_jsonb = auto_cast_struct_for_jsonb

    # --- read ----------------------------------------------------------------

    def read(
        self,
        conn: SQLBackend,
        storage_config: StorageConfig,
        key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read from PostgreSQL, materialize to Polars, and parse JSONB to Structs.

        SQL-level filters are applied before materialization. PostgreSQL JSON
        decoding requires eager materialization to Polars, so the returned
        LazyFrame wraps an in-memory snapshot rather than a deferred query.
        """
        table_name = self._resolve_table_name(storage_config, key)
        nw_lazy = self._get_filtered_ibis_lazy(conn, table_name, key, filters=filters)
        if nw_lazy is None:
            return None

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        # Identify JSON-compatible columns from the original table schema
        original_schema = conn.table(table_name).schema()
        json_columns_to_parse = self._get_json_columns_for_struct(original_schema)

        # Materialize to Polars and parse JSONB strings back to Structs
        pl_df = collect_to_polars(nw_lazy)
        pl_df = self._parse_json_to_struct_columns(pl_df, key, json_columns_to_parse)
        pl_df = self._cast_empty_system_struct_columns(pl_df, key, json_columns_to_parse)

        is_system_table = len(key) >= 1 and key[0] == METAXY_SYSTEM_KEY_PREFIX
        self._validate_required_system_struct_columns(
            pl_df,
            key,
            json_columns_to_parse,
            require_all_system_columns=columns is None and not is_system_table,
        )

        return nw.from_native(pl_df.lazy())

    # --- write ---------------------------------------------------------------

    def write(
        self,
        conn: SQLBackend,
        storage_config: StorageConfig,
        key: FeatureKey,
        df: Frame,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Write feature metadata, preserving JSONB for auto-created Struct columns."""
        table_name = self._resolve_table_name(storage_config, key)
        original_pl_df = collect_to_polars(df)
        transformed_pl_df = self._encode_struct_columns(original_pl_df)

        if table_name not in conn.list_tables():
            if not self.auto_create_tables:
                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist for feature {key.to_string()}. "
                    "Enable auto_create_tables=True to automatically create tables, "
                    "or use proper database migration tools like Alembic to create the table first."
                )

            auto_create_schema = self._build_auto_create_schema(original_pl_df, transformed_pl_df)
            self._warn_auto_create_table(table_name)
            if auto_create_schema is None:
                conn.create_table(table_name, obj=transformed_pl_df)
                return

            conn.create_table(table_name, schema=auto_create_schema)
            self._insert_feature_rows(conn, table_name, transformed_pl_df, auto_create_schema)
            return

        target_schema = conn.table(table_name).schema()
        self._insert_feature_rows(conn, table_name, transformed_pl_df, target_schema)

    # --- transform hooks -----------------------------------------------------

    def transform_after_read(
        self,
        conn: SQLBackend,
        table: ibis.Table,
        table_name: str,
        key: FeatureKey,  # noqa: ARG002
    ) -> ibis.Table:
        """Cast JSONB columns to String so Polars can parse them back to Structs."""
        schema = table.schema()
        json_columns = self._get_json_columns_for_struct(schema)
        if not json_columns:
            return table

        mutations = {col_name: table[col_name].cast("string") for col_name in json_columns}
        return table.mutate(**mutations)

    def transform_before_write(
        self,
        conn: SQLBackend,
        df: Frame,
        table_name: str,
        key: FeatureKey,  # noqa: ARG002
    ) -> Frame:
        """Convert Struct columns to JSON strings using Polars."""
        return nw.from_native(self._encode_struct_columns(collect_to_polars(df)))

    # --- JSON/Struct helpers (write side) ------------------------------------

    def _get_struct_columns_for_jsonb(self, schema: pl.Schema) -> list[str]:
        """Identify Struct columns to JSON-string serialize before insert."""
        if self.auto_cast_struct_for_jsonb:
            return [col_name for col_name, col_dtype in schema.items() if isinstance(col_dtype, pl.Struct)]
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
    ) -> IbisSchema | None:
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
        conn: SQLBackend,
        table_name: str,
        pl_df: pl.DataFrame,
        target_schema: IbisSchema,
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
            conn.insert(table_name, obj=pl_df)  # ty: ignore[invalid-argument-type]
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
        raw_connection = getattr(conn, "con")
        with raw_connection.cursor() as cursor:
            cursor.executemany(query, rows)

    # --- JSON/Struct helpers (read side) -------------------------------------

    def _get_json_columns_for_struct(self, ibis_schema: IbisSchema) -> list[str]:
        """Identify which JSON/TEXT columns should be parsed back to Structs."""
        import ibis.expr.datatypes as dt

        json_columns: list[str] = []
        for col_name, dtype in ibis_schema.items():
            is_metaxy_column = col_name in _METAXY_STRUCT_COLUMNS
            if is_metaxy_column and isinstance(dtype, (dt.JSON, dt.String)):
                json_columns.append(col_name)
            elif self.auto_cast_struct_for_jsonb and isinstance(dtype, dt.JSON):
                json_columns.append(col_name)
        return json_columns

    def _parse_json_to_struct_columns(
        self, pl_df: pl.DataFrame, key: FeatureKey, json_columns: list[str]
    ) -> pl.DataFrame:
        """Parse JSON string columns back to Polars Structs on a per-column basis."""
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
        key: FeatureKey,
        json_columns: list[str],
    ) -> pl.DataFrame:
        """Restore known Metaxy Struct schemas for empty result sets."""
        is_system_table = len(key) >= 1 and key[0] == METAXY_SYSTEM_KEY_PREFIX
        if pl_df.height != 0 or is_system_table:
            return pl_df

        try:
            graph = current_graph()
            plan = graph.get_feature_plan(key)
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
        key: FeatureKey,
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
                    f"feature {key.to_string()}: {missing_columns_csv}. "
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
                return bool((normalized == "null").all())
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
                f"feature {key.to_string()}: {invalid_columns_csv}. "
                "Required system columns must contain decodable JSON object payloads."
            )


@public
@experimental
class PostgreSQLMetadataStore(IbisStoreBackcompat):
    """PostgreSQL metadata store factory.

    Returns a ``MetadataStore`` composed with a ``PostgreSQLEngine`` and
    ``PostgreSQLSQLHandler`` for JSON/Struct round-tripping.

    Example:
        <!-- skip next -->
        ```python
        from metaxy.ext.metadata_stores.postgresql import PostgreSQLMetadataStore

        store = PostgreSQLMetadataStore(connection_string="postgresql://user:pass@localhost:5432/metaxy")

        with store:
            increment = store.resolve_update(MyFeature)
            store.write(MyFeature, increment.added)
        ```
    """

    def __init__(
        self,
        connection_string: str | None = None,  # noqa: ARG002
        *,
        connection_params: dict[str, Any] | None = None,  # noqa: ARG002
        fallback_stores: list[MetadataStore] | None = None,  # noqa: ARG002
        auto_cast_struct_for_jsonb: bool = True,  # noqa: ARG002
        table_prefix: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass  # __new__ already initialized via MetadataStore.__init__

    def __new__(
        cls,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        auto_cast_struct_for_jsonb: bool = True,
        table_prefix: str | None = None,
        **kwargs: Any,
    ) -> MetadataStore:
        if connection_string is None and connection_params is None:
            raise ValueError("Must provide either connection_string or connection_params")

        auto_create_tables = kwargs.pop("auto_create_tables", None)
        if auto_create_tables is None:
            from metaxy.config import MetaxyConfig

            auto_create_tables = MetaxyConfig.get().auto_create_tables

        handler = PostgreSQLSQLHandler(
            auto_create_tables=auto_create_tables,
            auto_cast_struct_for_jsonb=auto_cast_struct_for_jsonb,
        )
        engine = PostgreSQLEngine(
            connection_string=connection_string,
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
            handler=handler,
        )

        prefix = table_prefix or ""
        location = connection_string or "postgresql"
        storage = [IbisStorageConfig(format="postgresql", location=location, table_prefix=prefix)]

        instance = IbisStoreBackcompat.__new__(cls)
        MetadataStore.__init__(
            instance,
            engine=engine,
            storage=storage,
            fallback_stores=fallback_stores,
            auto_create_tables=auto_create_tables,
            **kwargs,
        )
        return instance

    @classmethod
    def config_model(cls) -> type[PostgreSQLMetadataStoreConfig]:
        """Return the config model for this metadata store."""
        return PostgreSQLMetadataStoreConfig

    # -- backcompat properties (deprecated, will be removed in 0.2.0) --

    @property
    def _pg_handler(self) -> PostgreSQLSQLHandler:
        handler = self._engine.handler if isinstance(self._engine.handler, PostgreSQLSQLHandler) else None
        assert handler is not None
        return handler

    @property
    def _conn(self) -> Any:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("_conn")
        return self._ibis_engine._conn

    @property
    def auto_cast_struct_for_jsonb(self) -> bool:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("auto_cast_struct_for_jsonb")
        return self._pg_handler.auto_cast_struct_for_jsonb

    @auto_cast_struct_for_jsonb.setter
    def auto_cast_struct_for_jsonb(self, value: bool) -> None:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("auto_cast_struct_for_jsonb")
        self._pg_handler.auto_cast_struct_for_jsonb = value

    def transform_before_write(self, df: Any, feature_key: Any, table_name: str) -> Any:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("transform_before_write")
        return self._pg_handler.transform_before_write(self._ibis_engine.conn, df, table_name, feature_key)

    def _get_json_columns_for_struct(self, ibis_schema: Any) -> list[str]:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("_get_json_columns_for_struct")
        return self._pg_handler._get_json_columns_for_struct(ibis_schema)

    def _parse_json_to_struct_columns(self, pl_df: Any, feature_key: Any, json_columns: list[str]) -> Any:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("_parse_json_to_struct_columns")
        return self._pg_handler._parse_json_to_struct_columns(pl_df, feature_key, json_columns)

    def _validate_required_system_struct_columns(
        self,
        pl_df: Any,
        feature_key: Any,
        json_columns: list[str],
        *,
        require_all_system_columns: bool = False,
    ) -> None:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("_validate_required_system_struct_columns")
        self._pg_handler._validate_required_system_struct_columns(
            pl_df,
            feature_key,
            json_columns,
            require_all_system_columns=require_all_system_columns,
        )


class PostgreSQLEngine(IbisComputeEngine):
    """Compute engine for PostgreSQL backends using Ibis with Polars versioning."""

    versioning_engine_cls = PolarsVersioningEngine

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        auto_create_tables: bool = False,
        handler: PostgreSQLSQLHandler | None = None,
    ) -> None:
        if connection_string is None and connection_params is None:
            raise ValueError("Must provide either connection_string or connection_params for PostgreSQL")

        super().__init__(
            connection_string=connection_string,
            backend="postgres" if connection_string is None else None,
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
            handler=handler,
        )

    def _create_hash_functions(self) -> dict:
        return {}

    def get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXHASH32

    @contextmanager
    def create_versioning_engine(self, plan: FeaturePlan) -> Iterator[PolarsVersioningEngine]:
        yield PolarsVersioningEngine(plan=plan)

    def validate_hash_algorithm_support(self, algorithm: HashAlgorithm) -> None:
        supported = PolarsVersioningEngine.supported_hash_algorithms()
        if algorithm not in supported:
            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm '{algorithm.value}' not supported. "
                f"Supported algorithms: {', '.join(a.value for a in sorted(supported, key=lambda a: a.value))}"
            )

    def display(self) -> str:
        from metaxy.metadata_store.utils import sanitize_uri

        location = self.connection_string or "postgresql"
        return f"PostgreSQLEngine(connection={sanitize_uri(location)})"

    @classmethod
    def config_model(cls) -> type[PostgreSQLMetadataStoreConfig]:
        return PostgreSQLMetadataStoreConfig
