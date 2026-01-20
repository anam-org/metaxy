"""PostgreSQL-specific versioning engine overrides."""

from typing import cast

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.versioning.flat_engine import IbisFlatVersioningEngine


class PostgresVersioningEngine(IbisFlatVersioningEngine):
    """Versioning engine for PostgreSQL using window functions for latest-by-group."""

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_column: nw.Expr | str,
    ) -> FrameT:
        """Keep only the latest row per group using row_number windowing.

        Args:
            df: Narwhals DataFrame/LazyFrame backed by Ibis
            group_columns: Columns to group by (typically ID columns)
            timestamp_column: Expression or column name to use for determining "latest"

        Returns:
            Narwhals DataFrame/LazyFrame with only the latest row per group
        """
        import ibis
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS

        # Convert string column name to expression if needed
        if isinstance(timestamp_column, str):
            timestamp_expr = nw.col(timestamp_column)
        else:
            timestamp_expr = timestamp_column

        # Create a temporary column for ordering expression
        columns = df.collect_schema().names()
        temp_column = "__metaxy_ordering_timestamp"
        suffix = 0
        while temp_column in columns:
            suffix += 1
            temp_column = f"__metaxy_ordering_timestamp_{suffix}"
        df = df.with_columns(  # ty: ignore[invalid-argument-type]
            timestamp_expr.alias(temp_column)
        )

        ibis_table: ibis.expr.types.Table = cast("ibis.expr.types.Table", df.to_native())

        group_exprs = [ibis_table[col] for col in group_columns]
        order_exprs = [ibis_table[temp_column].desc()]
        window = ibis.window(
            group_by=group_exprs,
            order_by=order_exprs,
        )
        # Note: Ibis row_number() starts at 0, unlike standard SQL which starts at 1
        ranked = ibis_table.mutate(_metaxy_rn=ibis.row_number().over(window))
        result_table = ranked.filter(ranked["_metaxy_rn"] == 0).drop("_metaxy_rn", temp_column)
        return cast(FrameT, nw.from_native(result_table))
