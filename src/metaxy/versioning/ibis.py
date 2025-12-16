"""Ibis implementation of VersioningEngine.

CRITICAL: This implementation NEVER materializes lazy expressions.
All operations stay in the lazy Ibis world for SQL execution.
"""

from typing import Protocol, cast

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

    Only implements hash_string_column and build_struct_column.
    All logic lives in the base class.

    CRITICAL: This implementation NEVER leaves the lazy world.
    All operations stay as Ibis expressions that compile to SQL.
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
    ) -> FrameT:
        """Hash a string column using Ibis hash functions.

        Args:
            df: Narwhals DataFrame backed by Ibis
            source_column: Name of string column to hash
            target_column: Name for the new column containing the hash
            hash_algo: Hash algorithm to use

        Returns:
            Narwhals DataFrame with new hashed column added, backed by Ibis.
            The source column remains unchanged.
        """
        if hash_algo not in self.hash_functions:
            raise ValueError(
                f"Hash algorithm {hash_algo} not supported by this Ibis backend. "
                f"Supported: {list(self.hash_functions.keys())}"
            )

        # Import ibis lazily (module-level import restriction)
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Get hash function
        hash_fn = self.hash_functions[hash_algo]

        # Apply hash to source column
        # Hash functions are responsible for returning strings
        hashed = hash_fn(ibis_table[source_column])

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
        # Import ibis lazily
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

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

    @staticmethod
    def aggregate_with_string_concat(
        df: FrameT,
        group_by_columns: list[str],
        concat_column: str,
        concat_separator: str,
        exclude_columns: list[str],
    ) -> FrameT:
        """Aggregate DataFrame by grouping and concatenating strings.

        Args:
            df: Narwhals DataFrame backed by Ibis
            group_by_columns: Columns to group by
            concat_column: Column containing strings to concatenate within groups
            concat_separator: Separator to use when concatenating strings
            exclude_columns: Columns to exclude from aggregation

        Returns:
            Narwhals DataFrame with one row per group.
        """
        # Import ibis lazily
        import ibis
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Build aggregation expressions
        agg_exprs = {}

        # Concatenate the concat_column with separator
        agg_exprs[concat_column] = ibis_table[concat_column].group_concat(
            concat_separator
        )

        # Take first value for all other columns (except group_by and exclude)
        all_columns = set(ibis_table.columns)
        columns_to_aggregate = (
            all_columns - set(group_by_columns) - {concat_column} - set(exclude_columns)
        )

        for col in columns_to_aggregate:
            agg_exprs[col] = ibis_table[
                col
            ].arbitrary()  # Take any value (like first())

        # Perform groupby and aggregate
        result_table = ibis_table.group_by(group_by_columns).aggregate(**agg_exprs)

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_column: str,
    ) -> FrameT:
        """Keep only the latest row per group based on a timestamp column.

        Uses argmax aggregation to get the value from each column where the
        timestamp is maximum. This is simpler and more semantically clear than
        window functions.

        Args:
            df: Narwhals DataFrame/LazyFrame backed by Ibis
            group_columns: Columns to group by (typically ID columns)
            timestamp_column: Column to use for determining "latest" (typically metaxy_created_at)

        Returns:
            Narwhals DataFrame/LazyFrame with only the latest row per group

        Raises:
            ValueError: If timestamp_column doesn't exist in df
        """
        # Import ibis lazily
        import ibis.expr.types

        # Convert to Ibis table
        assert df.implementation == nw.Implementation.IBIS, (
            "Only Ibis DataFrames are accepted"
        )

        # Check if timestamp_column exists
        if timestamp_column not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_column}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )

        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        # Use argmax aggregation: for each column, get the value where timestamp is maximum
        # This directly expresses "get the row with the latest timestamp per group"
        all_columns = set(ibis_table.columns)
        non_group_columns = all_columns - set(group_columns)

        # Build aggregation dict: for each non-group column, use argmax(timestamp)
        agg_exprs = {
            col: ibis_table[col].argmax(ibis_table[timestamp_column])
            for col in non_group_columns
        }

        # Perform groupby and aggregate
        result_table = ibis_table.group_by(group_columns).aggregate(**agg_exprs)

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result_table))
