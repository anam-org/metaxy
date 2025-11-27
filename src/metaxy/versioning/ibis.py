"""Ibis implementation of VersioningEngine.

CRITICAL: This implementation NEVER materializes lazy expressions.
All operations stay in the lazy Ibis world for SQL execution.
"""

from abc import ABC
from typing import Protocol, cast

import narwhals as nw
from ibis import Expr as IbisExpr
from narwhals.typing import FrameT

from metaxy.models.plan import FeaturePlan
from metaxy.versioning.engine import VersioningEngine
from metaxy.versioning.types import HashAlgorithm


class IbisHashFn(Protocol):
    def __call__(self, expr: IbisExpr) -> IbisExpr: ...


class BaseIbisVersioningEngine(VersioningEngine, ABC):
    """Shared Ibis-specific plumbing for provenance engines.

    Handles hashing, grouping helpers, and keeps all operations lazy in Ibis.
    Concrete subclasses decide how to represent field-level structs (true structs vs. flattened columns).
    """

    def __init__(
        self,
        plan: FeaturePlan,
        hash_functions: dict[HashAlgorithm, IbisHashFn],
    ) -> None:
        """Initialize the Ibis engine."""
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
    ) -> FrameT:
        """Hash a string column using Ibis hash functions."""
        if hash_algo not in self.hash_functions:
            raise ValueError(
                f"Hash algorithm {hash_algo} not supported by this Ibis backend. "
                f"Supported: {list(self.hash_functions.keys())}"
            )

        import ibis
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        hash_fn = self.hash_functions[hash_algo]
        hashed = hash_fn(ibis_table[source_column])

        result_table = ibis_table.mutate(**{target_column: hashed})  # pyright: ignore[reportArgumentType]
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def aggregate_with_string_concat(
        df: FrameT,
        group_by_columns: list[str],
        concat_column: str,
        concat_separator: str,
        exclude_columns: list[str],
    ) -> FrameT:
        """Aggregate DataFrame by grouping and concatenating strings."""
        import ibis
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        agg_exprs = {}
        agg_exprs[concat_column] = ibis_table[concat_column].group_concat(
            concat_separator
        )

        all_columns = set(ibis_table.columns)
        columns_to_aggregate = (
            all_columns - set(group_by_columns) - {concat_column} - set(exclude_columns)
        )

        for col in columns_to_aggregate:
            agg_exprs[col] = ibis_table[col].arbitrary()

        result_table = ibis_table.group_by(group_by_columns).aggregate(**agg_exprs)
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_column: str,
    ) -> FrameT:
        """Keep only the latest row per group based on a timestamp column."""
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )

        if timestamp_column not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_column}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )

        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        all_columns = set(ibis_table.columns)
        non_group_columns = all_columns - set(group_columns)

        agg_exprs = {
            col: ibis_table[col].argmax(ibis_table[timestamp_column])
            for col in non_group_columns
        }

        result_table = ibis_table.group_by(group_columns).aggregate(**agg_exprs)
        return cast(FrameT, nw.from_native(result_table))


class IbisVersioningEngine(BaseIbisVersioningEngine):
    """Provenance engine using Ibis for SQL databases with native struct support."""

    @staticmethod
    def record_field_versions(
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Persist field-level versions using a struct column."""
        import ibis
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        struct_expr = ibis.struct(
            {
                field_name: ibis_table[col_name]
                for field_name, col_name in field_columns.items()
            }
        )

        result_table = ibis_table.mutate(**{struct_name: struct_expr})
        return cast(FrameT, nw.from_native(result_table))
