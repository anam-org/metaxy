"""PostgreSQL SQL handler with JSON/Struct round-tripping."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl
from narwhals.typing import Frame

from metaxy.ext.ibis.engine import IbisSQLHandler
from metaxy.metadata_store.exceptions import TableNotFoundError
from metaxy.metadata_store.system.keys import METAXY_SYSTEM_KEY_PREFIX
from metaxy.metadata_store.table_ref import SQLTableIdentifier
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.feature import current_graph
from metaxy.utils import collect_to_polars

if TYPE_CHECKING:
    import ibis
    from ibis.backends.sql import SQLBackend
    from ibis.expr.schema import Schema as IbisSchema

_METAXY_STRUCT_COLUMNS = frozenset({METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD})


class PostgreSQLSQLHandler(IbisSQLHandler):
    """SQL storage handler for PostgreSQL with JSON/Struct round-tripping."""

    def __init__(self, *, auto_create_tables: bool = False, auto_cast_struct_for_jsonb: bool = True) -> None:
        super().__init__(auto_create_tables=auto_create_tables)
        self.auto_cast_struct_for_jsonb = auto_cast_struct_for_jsonb

    def read(
        self,
        conn: SQLBackend,
        table_id: SQLTableIdentifier,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        table_name = table_id.table_name
        nw_lazy = self._get_filtered_ibis_lazy(conn, table_name, filters=filters)
        if nw_lazy is None:
            return None

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        original_schema = conn.table(table_name).schema()
        json_columns_to_parse = self._get_json_columns_for_struct(original_schema)

        pl_df = collect_to_polars(nw_lazy)
        pl_df = self._parse_json_to_struct_columns(pl_df, json_columns_to_parse)
        pl_df = self._cast_empty_system_struct_columns(pl_df, table_name, json_columns_to_parse)

        is_system_table = table_name.startswith(METAXY_SYSTEM_KEY_PREFIX)
        self._validate_required_system_struct_columns(
            pl_df,
            table_name,
            json_columns_to_parse,
            require_all_system_columns=columns is None and not is_system_table,
        )

        return nw.from_native(pl_df.lazy())

    def write(
        self,
        conn: SQLBackend,
        table_id: SQLTableIdentifier,
        df: Frame,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        table_name = table_id.table_name
        original_pl_df = collect_to_polars(df)
        transformed_pl_df = self._encode_struct_columns(original_pl_df)

        if table_name not in conn.list_tables():
            if not self.auto_create_tables:
                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist. "
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

    def transform_after_read(
        self,
        conn: SQLBackend,
        table: ibis.Table,
        table_name: str,
    ) -> ibis.Table:
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
    ) -> Frame:
        return nw.from_native(self._encode_struct_columns(collect_to_polars(df)))

    def _get_struct_columns_for_jsonb(self, schema: pl.Schema) -> list[str]:
        if self.auto_cast_struct_for_jsonb:
            return [col_name for col_name, col_dtype in schema.items() if isinstance(col_dtype, pl.Struct)]
        return [
            col_name
            for col_name in _METAXY_STRUCT_COLUMNS
            if col_name in schema and isinstance(schema[col_name], pl.Struct)
        ]

    def _encode_struct_columns(self, pl_df: pl.DataFrame) -> pl.DataFrame:
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

    def _get_json_columns_for_struct(self, ibis_schema: IbisSchema) -> list[str]:
        import ibis.expr.datatypes as dt

        json_columns: list[str] = []
        for col_name, dtype in ibis_schema.items():
            is_metaxy_column = col_name in _METAXY_STRUCT_COLUMNS
            if is_metaxy_column and isinstance(dtype, (dt.JSON, dt.String)):
                json_columns.append(col_name)
            elif self.auto_cast_struct_for_jsonb and isinstance(dtype, dt.JSON):
                json_columns.append(col_name)
        return json_columns

    def _parse_json_to_struct_columns(self, pl_df: pl.DataFrame, json_columns: list[str]) -> pl.DataFrame:
        if not json_columns:
            return pl_df

        for col_name in json_columns:
            if col_name not in pl_df.columns:
                continue
            if isinstance(pl_df.schema[col_name], pl.Struct):
                continue

            col_values = pl_df[col_name]
            non_null_values = col_values.drop_nulls()
            if non_null_values.len() == 0:
                continue

            normalized = non_null_values.cast(pl.Utf8).str.strip_chars()
            is_object_json = normalized.str.starts_with("{") & normalized.str.ends_with("}")
            is_json_null_literal = normalized == "null"

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
        table_name: str,
        json_columns: list[str],
    ) -> pl.DataFrame:
        is_system_table = table_name.startswith(METAXY_SYSTEM_KEY_PREFIX)
        if pl_df.height != 0 or is_system_table:
            return pl_df

        try:
            graph = current_graph()
            # Reverse-lookup: find the feature key whose table_name matches
            matching_key = next(
                (k for k in graph.feature_definitions_by_key if k.table_name == table_name),
                None,
            )
            if matching_key is None:
                return pl_df
            plan = graph.get_feature_plan(matching_key)
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
        table_name: str,
        json_columns: list[str],
        *,
        require_all_system_columns: bool = False,
    ) -> None:
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
                    f"table '{table_name}': {missing_columns_csv}. "
                    "Required system columns are missing from the result set."
                )

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
                f"table '{table_name}': {invalid_columns_csv}. "
                "Required system columns must contain decodable JSON object payloads."
            )
