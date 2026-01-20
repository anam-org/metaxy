"""ClickHouse-specific implementation of VersioningEngine.

ClickHouse has issues with Ibis's group_concat().over(window) which generates
invalid SQL with CASE WHEN ... END OVER (...). Instead, we use:
  arrayStringConcat(groupArray(col) OVER (...), separator)
"""

from typing import cast

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.versioning.ibis import IbisVersioningEngine


class ClickHouseVersioningEngine(IbisVersioningEngine):
    """Versioning engine for ClickHouse backend.

    Overrides concat_strings_over_groups to use ClickHouse-compatible
    syntax with collect() (groupArray) + arrayStringConcat.
    """

    def concat_strings_over_groups(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        group_by_columns: list[str],
        order_by_columns: list[str],
        separator: str = "|",
    ) -> FrameT:
        """Concatenate string values within groups using ClickHouse window functions.

        Uses collect() (groupArray) + arrayStringConcat instead of group_concat().over()
        which generates invalid SQL for ClickHouse.
        """
        import ibis
        import ibis.expr.datatypes as dt
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Define ClickHouse arrayStringConcat function
        @ibis.udf.scalar.builtin
        def arrayStringConcat(arr: dt.Array[dt.String], sep: str) -> str:  # ty: ignore[invalid-return-type]
            """ClickHouse arrayStringConcat() function."""
            ...

        # Create window spec with ordering for deterministic results
        # Fall back to group_by columns for ordering if no explicit order_by columns
        effective_order_by = order_by_columns if order_by_columns else group_by_columns
        window = ibis.window(
            group_by=group_by_columns,
            order_by=[ibis_table[col] for col in effective_order_by],
        )

        # Use collect() (groupArray in ClickHouse) over window, then arrayStringConcat
        arr_expr = ibis_table[source_column].cast("string").collect().over(window)
        concat_expr = arrayStringConcat(arr_expr, separator)

        ibis_table = ibis_table.mutate(**{target_column: concat_expr})

        return cast(FrameT, nw.from_native(ibis_table))
