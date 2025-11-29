"""Ibis implementation of VersioningEngine."""

from abc import ABC
from typing import TYPE_CHECKING, Protocol, cast

import narwhals as nw
from ibis import Expr as IbisExpr
from narwhals.typing import FrameT

from metaxy.models.plan import FeaturePlan
from metaxy.utils.constants import TEMP_TABLE_NAME
from metaxy.versioning.engine import VersioningEngine
from metaxy.versioning.struct_adapter import StructFieldAccessor
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    import ibis.expr.types


class IbisHashFn(Protocol):
    def __call__(self, expr: IbisExpr) -> IbisExpr: ...


class BaseIbisVersioningEngine(VersioningEngine, ABC):
    """Shared Ibis-specific plumbing for provenance engines.

    !!! info
        This implementation never leaves the lazy world.
        All operations stay as Ibis expressions and eventually get compiled to SQL.

    Handles hashing, grouping helpers, and keeps all operations lazy in Ibis.
    Concrete subclasses decide how to represent field-level structs (true structs vs. flattened columns).
    """

    def __init__(
        self,
        plan: FeaturePlan,
        hash_functions: dict[HashAlgorithm, IbisHashFn],
    ) -> None:
        """Initialize the Ibis engine.

        Args:
            plan: Feature plan to track provenance for
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
        """Hash a string column using Ibis hash functions."""
        if hash_algo not in self.hash_functions:
            raise ValueError(
                f"Hash algorithm {hash_algo} not supported by this Ibis backend. "
                f"Supported: {list(self.hash_functions.keys())}"
            )

        # Import ibis lazily (module-level import restriction)
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Apply hash function to source column
        hash_fn = self.hash_functions[hash_algo]
        hashed = hash_fn(ibis_table[source_column])

        # Apply truncation if specified
        if truncate_length is not None:
            hashed = hashed[0:truncate_length]

        result_table = ibis_table.mutate(**{target_column: hashed})

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
        """Concatenate string values within groups using Ibis window functions."""
        import ibis
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Create window spec with ordering for deterministic results
        effective_order_by = order_by_columns if order_by_columns else group_by_columns
        window = ibis.window(
            group_by=group_by_columns,
            order_by=[ibis_table[col] for col in effective_order_by],
        )

        # Use group_concat over window to concatenate values
        concat_expr = ibis_table[source_column].cast("string").group_concat(sep=separator).over(window)
        result_table = ibis_table.mutate(**{target_column: concat_expr})
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_columns: list[str],
    ) -> FrameT:
        """Keep only the latest row per group based on timestamp columns.

        Uses argmax aggregation to get the value from each column where the
        timestamp is maximum. This is simpler and more semantically clear than
        window functions.

        Args:
            df: Narwhals DataFrame/LazyFrame backed by Ibis
            group_columns: Columns to group by (typically ID columns)
            timestamp_columns: Column names to coalesce for ordering (uses first non-null value)

        Returns:
            Narwhals DataFrame/LazyFrame with only the latest row per group
        """
        # Import ibis lazily
        import ibis
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"

        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Create a temporary column for ordering using coalesce
        ordering_expr = ibis.coalesce(*[ibis_table[col] for col in timestamp_columns])
        ibis_table = ibis_table.mutate(**{TEMP_TABLE_NAME: ordering_expr})

        # Use argmax aggregation: for each column, get the value where timestamp is maximum
        all_columns = set(ibis_table.columns)
        non_group_columns = all_columns - set(group_columns) - {TEMP_TABLE_NAME}

        # Build aggregation dict: for each non-group column, use argmax(timestamp)
        agg_exprs = {col: ibis_table[col].argmax(ibis_table[TEMP_TABLE_NAME]) for col in non_group_columns}

        result_table = ibis_table.group_by(group_columns).aggregate(**agg_exprs)
        # Note: TEMP_TABLE_NAME is not in result_table because we excluded it from aggregation

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))


class IbisVersioningEngine(StructFieldAccessor, BaseIbisVersioningEngine):
    """Provenance engine using Ibis for SQL databases with native struct support.

    CRITICAL: This implementation NEVER leaves the lazy world.
    All operations stay as Ibis expressions that compile to SQL.
    """

    def record_field_versions(
        self,
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Persist field-level versions using a struct column."""
        import ibis
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS
        ibis_table: ibis.expr.types.Table = cast("ibis.expr.types.Table", df.to_native())

        # Build struct expression - reference columns by name
        struct_expr = ibis.struct({field_name: ibis_table[col_name] for field_name, col_name in field_columns.items()})

        result_table = ibis_table.mutate(**{struct_name: struct_expr})
        return cast(FrameT, nw.from_native(result_table))
