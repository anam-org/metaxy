"""Polars implementation of ProvenanceTracker."""

from collections.abc import Callable
from typing import cast

import narwhals as nw
import polars as pl
import polars_hash  # noqa: F401  # Registers .nchash and .chash namespaces
from narwhals.typing import FrameT

from metaxy.provenance.tracker import ProvenanceTracker
from metaxy.provenance.types import HashAlgorithm

# narwhals DataFrame backed by either a lazy or an eager frame
# PolarsFrame = TypeVar("PolarsFrame", pl.DataFrame, pl.LazyFrame)


class PolarsProvenanceTracker(ProvenanceTracker):
    """Provenance tracker using Polars and polars_hash plugin.

    Only implements hash_string_column and build_struct_column.
    All logic lives in the base class.
    """

    # Map HashAlgorithm enum to polars-hash functions
    _HASH_FUNCTION_MAP: dict[HashAlgorithm, Callable[[pl.Expr], pl.Expr]] = {
        HashAlgorithm.XXHASH64: lambda expr: expr.nchash.xxhash64(),  # pyright: ignore[reportAttributeAccessIssue]
        HashAlgorithm.XXHASH32: lambda expr: expr.nchash.xxhash32(),  # pyright: ignore[reportAttributeAccessIssue]
        HashAlgorithm.WYHASH: lambda expr: expr.nchash.wyhash(),  # pyright: ignore[reportAttributeAccessIssue]
        HashAlgorithm.SHA256: lambda expr: expr.chash.sha2_256(),  # pyright: ignore[reportAttributeAccessIssue]
        HashAlgorithm.MD5: lambda expr: expr.nchash.md5(),  # pyright: ignore[reportAttributeAccessIssue]
    }

    def hash_string_column(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        hash_algo: HashAlgorithm,
    ) -> FrameT:
        """Hash a string column using polars_hash.

        Args:
            df: Narwhals DataFrame backed by Polars
            source_column: Name of string column to hash
            target_column: Name for the new column containing the hash
            hash_algo: Hash algorithm to use

        Returns:
            Narwhals DataFrame with new hashed column added, backed by Polars.
            The source column remains unchanged.
        """
        if hash_algo not in self._HASH_FUNCTION_MAP:
            raise ValueError(
                f"Hash algorithm {hash_algo} not supported. "
                f"Supported: {list(self._HASH_FUNCTION_MAP.keys())}"
            )

        assert df.implementation == nw.Implementation.POLARS, (
            "Only Polars DataFrames are accepted"
        )
        df_pl = cast(pl.DataFrame | pl.LazyFrame, df.to_native())

        # Apply hash
        hash_fn = self._HASH_FUNCTION_MAP[hash_algo]
        hashed = hash_fn(polars_hash.col(source_column)).cast(pl.Utf8)

        # Add new column with the hash
        df_pl = df_pl.with_columns(hashed.alias(target_column))

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(df_pl))

    def build_struct_column(
        self,
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Build a struct column from existing columns.

        Args:
            df: Narwhals DataFrame backed by Polars
            struct_name: Name for the new struct column
            field_columns: Mapping from struct field names to column names

        Returns:
            Narwhals DataFrame with new struct column added, backed by Polars.
            The source columns remain unchanged.
        """
        assert df.implementation == nw.Implementation.POLARS, (
            "Only Polars DataFrames are accepted"
        )
        df_pl = cast(pl.DataFrame | pl.LazyFrame, df.to_native())

        # Build struct expression
        struct_expr = pl.struct(
            [
                pl.col(col_name).alias(field_name)
                for field_name, col_name in field_columns.items()
            ]
        )

        # Add struct column
        df_pl = df_pl.with_columns(struct_expr.alias(struct_name))

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(df_pl))

    def aggregate_with_string_concat(
        self,
        df: FrameT,
        group_by_columns: list[str],
        concat_column: str,
        concat_separator: str,
        exclude_columns: list[str],
    ) -> FrameT:
        """Aggregate DataFrame by grouping and concatenating strings.

        Args:
            df: Narwhals DataFrame backed by Polars
            group_by_columns: Columns to group by
            concat_column: Column containing strings to concatenate within groups
            concat_separator: Separator to use when concatenating strings
            exclude_columns: Columns to exclude from aggregation

        Returns:
            Narwhals DataFrame with one row per group.
        """
        assert df.implementation == nw.Implementation.POLARS, (
            "Only Polars DataFrames are accepted"
        )
        df_pl = cast(pl.DataFrame | pl.LazyFrame, df.to_native())

        # Group and aggregate: concatenate concat_column, take first for others
        grouped = df_pl.group_by(group_by_columns).agg(
            [
                pl.col(concat_column).str.join(concat_separator),
                pl.exclude(
                    group_by_columns + [concat_column] + exclude_columns
                ).first(),
            ]
        )

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(grouped))

    def keep_latest_by_group(
        self,
        df: FrameT,
        group_columns: list[str],
        timestamp_column: str,
    ) -> FrameT:
        """Keep only the latest row per group by sorting and aggregating.

        Uses native Polars operations to sort by timestamp and take the last row per group.

        Args:
            df: Narwhals DataFrame/LazyFrame
            group_columns: Columns to group by (typically ID columns)
            timestamp_column: Column to use for determining "latest" (typically metaxy_created_at)

        Returns:
            Narwhals DataFrame/LazyFrame with only latest row per group

        Raises:
            ValueError: If timestamp_column does not exist in the DataFrame
        """
        # Validate timestamp column exists
        cols = df.collect_schema().names()
        if timestamp_column not in cols:
            raise ValueError(
                f"Timestamp column '{timestamp_column}' not found in DataFrame. "
                f"Available columns: {cols}"
            )

        # Convert to native Polars
        df_pl = cast(pl.DataFrame | pl.LazyFrame, df.to_native())

        # Ensure we're working with LazyFrame for consistent typing
        if isinstance(df_pl, pl.DataFrame):
            df_pl = df_pl.lazy()

        # Sort by group columns + timestamp (ascending), then take last row per group
        # Using maintain_order=True preserves the sort order
        result: pl.LazyFrame = (
            df_pl.sort([*group_columns, timestamp_column])
            .group_by(group_columns, maintain_order=True)
            .agg(pl.col("*").last())
        )

        # Convert back to Narwhals
        return cast(
            FrameT,
            nw.from_native(result, eager_only=False),
        )
