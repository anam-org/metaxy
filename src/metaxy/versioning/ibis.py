"""Ibis implementation of VersioningEngine.

CRITICAL: This implementation NEVER materializes lazy expressions.
All operations stay in the lazy Ibis world for SQL execution.
"""

from typing import Any, Protocol, cast

import ibis
import ibis.expr.types as ibis_types
import narwhals as nw
from ibis import Expr as IbisExpr
from narwhals.typing import FrameT

from metaxy.models.plan import FeaturePlan
from metaxy.versioning.engine import VersioningEngine
from metaxy.versioning.types import HashAlgorithm


class IbisHashFn(Protocol):
    def __call__(self, expr: IbisExpr) -> IbisExpr: ...


class IbisVersioningEngine(VersioningEngine):
    """Provenance engine using Ibis for SQL databases.

    !!!info
        This implementation never leaves the lazy world.
        All operations stay as Ibis expressions and eventually get compiled to SQL.
    """

    def __init__(
        self,
        plan: FeaturePlan,
        hash_functions: dict[HashAlgorithm, IbisHashFn],
    ) -> None:
        """Initialize the Ibis engine.

        Args:
            plan: Feature plan to track provenance for
            backend: Ibis backend instance (e.g., ibis.duckdb.connect())
            hash_functions: Mapping from HashAlgorithm to Ibis hash functions.
                Each function takes an Ibis expression and returns an Ibis expression.
        """
        super().__init__(plan)
        self.hash_functions: dict[HashAlgorithm, IbisHashFn] = hash_functions

    @classmethod
    def implementation(cls) -> nw.Implementation:
        return nw.Implementation.IBIS

    def hash_string_column(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        hash_algo: HashAlgorithm,
        truncate_length: int | None = None,
    ) -> FrameT:
        """Hash a string column using Ibis hash functions.

        Args:
            df: Narwhals DataFrame backed by Ibis
            source_column: Name of string column to hash
            target_column: Name for the new column containing the hash
            hash_algo: Hash algorithm to use
            truncate_length: Optional length to truncate hash to. If None, no truncation.

        Returns:
            Narwhals DataFrame with new hashed column added, backed by Ibis.
            The source column remains unchanged.
        """
        if hash_algo not in self.hash_functions:
            raise ValueError(
                f"Hash algorithm {hash_algo} not supported by this Ibis backend. "
                f"Supported: {list(self.hash_functions.keys())}"
            )

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis_types.Table = df.to_native()

        # Get hash function
        hash_fn = self.hash_functions[hash_algo]

        # Apply hash to source column
        # Hash functions are responsible for returning strings
        hashed = hash_fn(ibis_table[source_column])

        # Apply truncation if specified
        if truncate_length is not None:
            hashed = hashed[0:truncate_length]

        # Add new column with the hash
        result_table = ibis_table.mutate(**{target_column: hashed})

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def build_struct_column(
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Build a struct column from existing columns.

        Args:
            df: Narwhals DataFrame backed by Ibis
            struct_name: Name for the new struct column
            field_columns: Mapping from struct field names to column names

        Returns:
            Narwhals DataFrame with new struct column added, backed by Ibis.
            The source columns remain unchanged.
        """
        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis_types.Table = df.to_native()

        # Build struct expression - reference columns by name
        struct_expr = ibis.struct(
            {
                field_name: ibis_table[col_name]
                for field_name, col_name in field_columns.items()
            }
        )

        # Add struct column
        result_table = ibis_table.mutate(**{struct_name: struct_expr})

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))

    def concat_strings_over_groups(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        group_by_columns: list[str],
        order_by_columns: list[str],
        separator: str = "|",
    ) -> FrameT:
        """Concatenate string values within groups using Ibis window functions.

        Uses group_concat with ordering to concatenate values in deterministic order.
        All rows in the same group receive identical concatenated values.
        """
        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = df.to_native()

        # Create window spec with ordering for deterministic results
        # Fall back to group_by columns for ordering if no explicit order_by columns
        effective_order_by = order_by_columns if order_by_columns else group_by_columns
        window = ibis.window(
            group_by=group_by_columns,
            order_by=[ibis_table[col] for col in effective_order_by],
        )

        # Use group_concat over window to concatenate values
        concat_expr = (
            ibis_table[source_column]
            .cast("string")
            .group_concat(sep=separator)
            .over(window)
        )
        ibis_table = ibis_table.mutate(**{target_column: concat_expr})

        return cast(FrameT, nw.from_native(ibis_table))

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        order_by_columns: list[str],
    ) -> FrameT:
        """Keep only the latest row per group based on ordered columns.

        Args:
            df: Narwhals DataFrame/LazyFrame backed by Ibis
            group_columns: Columns to group by (typically ID columns)
            order_by_columns: Columns to order by (highest value wins)

        Returns:
            Narwhals DataFrame/LazyFrame with only the latest row per group

        Raises:
            ValueError: If no order_by_columns exist in df
        """
        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )

        if not order_by_columns:
            raise ValueError("order_by_columns must contain at least one column")

        ibis_table: ibis.expr.types.Table = df.to_native()

        # Get column names without triggering LazyFrame performance warning
        # Use cast to satisfy type checker - collect_schema exists on both DataFrame and LazyFrame
        from typing import cast as typing_cast

        df_any: Any = df
        df_columns = typing_cast(Any, df_any).collect_schema().names()

        present_order_cols = [col for col in order_by_columns if col in df_columns]
        if not present_order_cols:
            raise ValueError(
                f"None of the order_by_columns {order_by_columns} found in DataFrame. "
                f"Available columns: {df_columns}"
            )

        order_exprs = [ibis_table[col].desc() for col in present_order_cols]
        window = ibis.window(group_by=group_columns, order_by=order_exprs)

        ranked = ibis_table.mutate(_metaxy_row_num=ibis.row_number().over(window))
        filter_expr: ibis.Expr = ranked["_metaxy_row_num"] == ibis.literal(0)
        latest = ranked.filter(filter_expr)
        result_table = latest.drop("_metaxy_row_num")

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))
