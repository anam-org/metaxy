"""ClickHouse-specific SQL handler for Map/Struct and JSON type transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from metaxy.ext.ibis.engine import IbisSQLHandler

if TYPE_CHECKING:
    import ibis
    from ibis.backends.sql import SQLBackend


class ClickHouseSQLHandler(IbisSQLHandler):
    """Handles ClickHouse-specific JSON type transforms.

    On read, casts JSON columns to String for PyArrow compatibility. Map columns
    are read as their canonical `polars_map.Map` representation.
    """

    def __init__(self, *, auto_create_tables: bool = False, auto_cast_struct_for_map: bool = True) -> None:  # noqa: ARG002
        super().__init__(auto_create_tables=auto_create_tables)

    @staticmethod
    def _is_table_not_found_error(e: Exception) -> bool:
        import ibis.common.exceptions

        if isinstance(e, ibis.common.exceptions.TableNotFound):
            return True
        try:
            from clickhouse_connect.driver.exceptions import DatabaseError
        except ImportError:
            return False
        return isinstance(e, DatabaseError) and "UNKNOWN_TABLE" in str(e)

    # --- transform hooks -----------------------------------------------------

    def transform_after_read(
        self,
        conn: SQLBackend,
        table: ibis.Table,
        table_name: str,
    ) -> ibis.Table:
        """Cast JSON columns to String for PyArrow compatibility."""
        import ibis.expr.datatypes as dt

        schema = table.schema()
        mutations: dict[str, Any] = {}

        for col_name, dtype in schema.items():
            if isinstance(dtype, dt.JSON):
                mutations[col_name] = table[col_name].cast("string")

        if not mutations:
            return table

        return table.mutate(**mutations)

    def ibis_type_to_polars(self, ibis_type: Any) -> Any:
        """Convert an Ibis data type to Polars, mapping ``Map(K, V)`` to ``List(Struct{key, value})``."""
        import ibis.expr.datatypes as dt
        import polars as pl

        if isinstance(ibis_type, dt.Map):
            key_pl = self.ibis_type_to_polars(ibis_type.key_type)
            value_pl = self.ibis_type_to_polars(ibis_type.value_type)
            return pl.List(pl.Struct({"key": key_pl, "value": value_pl}))

        return super().ibis_type_to_polars(ibis_type)
