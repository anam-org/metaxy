from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ibis.expr.datatypes as dt
import narwhals as nw
from narwhals.typing import Frame

if TYPE_CHECKING:
    import ibis
    from ibis.backends.sql import SQLBackend
    from ibis.expr.schema import Schema as IbisSchema

from metaxy.metadata_store.ibis_compute_engine import IbisSQLHandler
from metaxy.models.types import FeatureKey


class ClickHouseSQLHandler(IbisSQLHandler):
    """Handles ClickHouse-specific Map/Struct and JSON type transforms.

    On read, converts JSON columns to String and Map columns to Struct
    (metaxy system columns use field names from the feature spec; user columns
    discover keys from the data when ``auto_cast_struct_for_map`` is enabled).

    On write, converts Struct columns back to Map format when the target
    ClickHouse column is Map type.
    """

    def __init__(self, *, auto_create_tables: bool = False, auto_cast_struct_for_map: bool = True) -> None:
        super().__init__(auto_create_tables=auto_create_tables)
        self.auto_cast_struct_for_map = auto_cast_struct_for_map
        self._ch_schema_cache: dict[str, IbisSchema] = {}

    # --- transform hooks -----------------------------------------------------

    def transform_after_read(
        self,
        conn: SQLBackend,
        table: ibis.Table,
        table_name: str,
        key: FeatureKey,
    ) -> ibis.Table:
        """Transform ClickHouse-specific column types for PyArrow compatibility.

        Handles JSON to String cast. When ``enable_map_datatype`` is off, also
        converts metaxy Map(String, String) columns to named Struct.
        """
        from metaxy.config import MetaxyConfig

        schema = table.schema()
        mutations: dict[str, Any] = {}

        for col_name, dtype in schema.items():
            if isinstance(dtype, dt.JSON):
                mutations[col_name] = table[col_name].cast("string")

            elif isinstance(dtype, dt.Map) and not MetaxyConfig.get().enable_map_datatype:
                from metaxy.models.constants import (
                    METAXY_DATA_VERSION_BY_FIELD,
                    METAXY_PROVENANCE_BY_FIELD,
                )

                if col_name in {METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD}:
                    mutations[col_name] = self._map_to_struct_expr(table, col_name, key)

        if not mutations:
            return table

        return table.mutate(**mutations)

    def transform_before_write(
        self,
        conn: SQLBackend,
        df: Frame,
        table_name: str,
        key: FeatureKey,  # noqa: ARG002
    ) -> Frame:
        """Transform Struct columns to Map format for ClickHouse Map columns.

        When ``enable_map_datatype`` is set, the base write path handles
        Struct→Map conversion via Arrow, so this is a no-op.
        """
        from metaxy.config import MetaxyConfig

        if MetaxyConfig.get().enable_map_datatype:
            return df

        if table_name not in conn.list_tables():
            return df

        ch_schema = self._get_cached_schema(conn, table_name)
        return self._transform_struct_to_map(df, ch_schema)

    def ibis_type_to_polars(self, ibis_type: Any) -> Any:
        """Convert Ibis data type to Polars, with ClickHouse Map support.

        Handles ``Map(K, V)`` as Polar's canonical map representation.
        """
        import polars as pl

        if isinstance(ibis_type, dt.Map):
            key_pl = self.ibis_type_to_polars(ibis_type.key_type)
            value_pl = self.ibis_type_to_polars(ibis_type.value_type)
            return pl.List(pl.Struct({"key": key_pl, "value": value_pl}))

        return super().ibis_type_to_polars(ibis_type)

    # --- helpers --------------------------------------------------------------

    def _get_cached_schema(self, conn: SQLBackend, table_name: str) -> IbisSchema:
        """Get ClickHouse table schema, cached per handler instance."""
        if table_name not in self._ch_schema_cache:
            self._ch_schema_cache[table_name] = conn.table(table_name).schema()
        return self._ch_schema_cache[table_name]

    def _map_to_struct_expr(
        self,
        table: ibis.Table,
        col_name: str,
        key: FeatureKey,
    ) -> Any:
        """Convert a Map column to a named Struct using field names from the feature spec."""
        import ibis

        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()
        definition = graph.feature_definitions_by_key.get(key)
        if definition is None:
            return table[col_name].cast("string")

        field_names = [f.key.to_struct_key() for f in definition.spec.fields]
        if not field_names:
            return table[col_name].cast("string")

        map_col = table[col_name]
        struct_dict = {name: map_col.get(name, "") for name in field_names}
        return ibis.struct(struct_dict)

    def _transform_struct_to_map(self, df: Frame, ch_schema: IbisSchema) -> Frame:
        """Transform Struct columns to Map-compatible format for ClickHouse.

        When ``auto_cast_struct_for_map`` is True, transforms ALL DataFrame Struct
        columns whose target ClickHouse column is Map type. Otherwise only transforms
        metaxy system columns.
        """

        from metaxy.models.constants import (
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_PROVENANCE_BY_FIELD,
        )

        metaxy_struct_columns = {METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD}

        if self.auto_cast_struct_for_map:
            map_columns = {name for name, dtype in ch_schema.items() if isinstance(dtype, dt.Map)}
        else:
            map_columns = {
                name for name, dtype in ch_schema.items() if isinstance(dtype, dt.Map) and name in metaxy_struct_columns
            }

        if not map_columns:
            return df

        if df.implementation == nw.Implementation.IBIS:
            return self._transform_ibis_struct_to_map(df, map_columns, ch_schema)

        return self._transform_polars_struct_to_map(df, map_columns, ch_schema)

    def _transform_ibis_struct_to_map(
        self,
        df: Frame,
        map_columns: set[str],
        ch_schema: IbisSchema,
    ) -> Frame:
        """Transform Ibis Struct columns to Map format."""
        from typing import cast as typing_cast

        import ibis

        ibis_table = typing_cast("ibis.Table", df.to_native())
        schema = ibis_table.schema()

        mutations: dict[str, ibis.Expr] = {}
        for col_name in map_columns:
            if col_name not in schema:
                continue

            col_dtype = schema[col_name]
            if not isinstance(col_dtype, dt.Struct):
                continue

            field_names = list(col_dtype.names)  # ty: ignore[invalid-argument-type]
            ch_map_dtype = ch_schema[col_name]
            target_value_type = ch_map_dtype.value_type  # ty: ignore[unresolved-attribute]

            if not field_names:
                mutations[col_name] = ibis.literal(
                    {},
                    type=dt.Map(dt.String(), target_value_type),  # ty: ignore[invalid-argument-type, missing-argument]
                )
                continue

            keys = ibis.array([ibis.literal(name) for name in field_names])
            values = ibis.array([ibis_table[col_name][name].cast(target_value_type) for name in field_names])
            mutations[col_name] = ibis.map(keys, values)

        if not mutations:
            return df

        result_table = ibis_table.mutate(**mutations)  # ty: ignore[invalid-argument-type]
        return nw.from_native(result_table, eager_only=False)

    def _transform_polars_struct_to_map(
        self,
        df: Frame,
        map_columns: set[str],
        ch_schema: IbisSchema,
    ) -> Frame:
        """Transform Polars Struct columns to List[Struct{key, value}] for ClickHouse Map."""
        import polars as pl

        from metaxy._utils import collect_to_polars

        pl_df = collect_to_polars(df)

        cols_to_transform: list[tuple[str, list[str], pl.DataType]] = []
        for col_name in map_columns:
            if col_name not in pl_df.columns:
                continue
            col_dtype = pl_df.schema[col_name]
            if not isinstance(col_dtype, pl.Struct):
                continue
            field_names = [f.name for f in col_dtype.fields]
            ch_map_dtype = ch_schema[col_name]
            target_pl_type = self.ibis_type_to_polars(
                ch_map_dtype.value_type  # ty: ignore[unresolved-attribute]
            )
            cols_to_transform.append((col_name, field_names, target_pl_type))

        if not cols_to_transform:
            return df

        transformations = []
        for col_name, field_names, target_type in cols_to_transform:
            if not field_names:
                empty_list_type = pl.List(pl.Struct({"key": pl.Utf8, "value": target_type}))
                transformations.append(pl.lit([], dtype=empty_list_type).alias(col_name))
                continue

            key_value_structs = [
                pl.when(pl.col(col_name).struct.field(field_name).is_not_null())
                .then(
                    pl.struct(
                        [
                            pl.lit(field_name).alias("key"),
                            pl.col(col_name).struct.field(field_name).cast(target_type).alias("value"),
                        ]
                    )
                )
                .otherwise(None)
                for field_name in field_names
            ]
            transformations.append(pl.concat_list(key_value_structs).list.drop_nulls().alias(col_name))

        pl_df = pl_df.with_columns(transformations)
        return nw.from_native(pl_df)
