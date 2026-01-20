"""Polars implementation of VersioningEngine."""

from collections.abc import Callable
from typing import cast

import narwhals as nw
import polars as pl
import polars_hash  # noqa: F401  # Registers .nchash and .chash namespaces
from narwhals.typing import FrameT

from metaxy.utils.constants import TEMP_TABLE_NAME
from metaxy.versioning.engine import VersioningEngine
from metaxy.versioning.types import HashAlgorithm

# narwhals DataFrame backed by either a lazy or an eager frame
# PolarsFrame = TypeVar("PolarsFrame", pl.DataFrame, pl.LazyFrame)


class PolarsVersioningEngine(VersioningEngine):
    """Provenance engine using Polars and polars_hash plugin.

    !!! info
        This implementation never leaves the lazy world.
    """

    # Map HashAlgorithm enum to polars-hash functions
    _HASH_FUNCTION_MAP: dict[HashAlgorithm, Callable[[pl.Expr], pl.Expr]] = {
        HashAlgorithm.XXHASH64: lambda expr: expr.nchash.xxhash64(),
        HashAlgorithm.XXHASH32: lambda expr: expr.nchash.xxhash32(),
        HashAlgorithm.WYHASH: lambda expr: expr.nchash.wyhash(),
        HashAlgorithm.SHA256: lambda expr: expr.chash.sha2_256(),
        HashAlgorithm.MD5: lambda expr: expr.nchash.md5(),
    }

    @classmethod
    def implementation(cls) -> nw.Implementation:
        return nw.Implementation.POLARS

    def hash_string_column(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        hash_algo: HashAlgorithm,
        truncate_length: int | None = None,
    ) -> FrameT:
        """Hash a string column using polars_hash.

        Args:
            df: Narwhals DataFrame backed by Polars
            source_column: Name of string column to hash
            target_column: Name for the new column containing the hash
            hash_algo: Hash algorithm to use
            truncate_length: Optional length to truncate hash to. If None, no truncation.

        Returns:
            Narwhals DataFrame with new hashed column added, backed by Polars.
            The source column remains unchanged.
        """
        if hash_algo not in self._HASH_FUNCTION_MAP:
            raise ValueError(
                f"Hash algorithm {hash_algo} not supported. Supported: {list(self._HASH_FUNCTION_MAP.keys())}"
            )

        assert df.implementation == nw.Implementation.POLARS, "Only Polars DataFrames are accepted"
        df_pl = cast(pl.DataFrame | pl.LazyFrame, df.to_native())  # ty: ignore[invalid-argument-type]

        # Apply hash
        hash_fn = self._HASH_FUNCTION_MAP[hash_algo]
        hashed = hash_fn(polars_hash.col(source_column)).cast(pl.Utf8)

        # Apply truncation if specified
        if truncate_length is not None:
            hashed = hashed.str.slice(0, truncate_length)

        # Add new column with the hash
        df_pl = df_pl.with_columns(hashed.alias(target_column))

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(df_pl))

    @staticmethod
    def build_struct_column(
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
        assert df.implementation == nw.Implementation.POLARS, "Only Polars DataFrames are accepted"
        df_pl = cast(pl.DataFrame | pl.LazyFrame, df.to_native())  # ty: ignore[invalid-argument-type]

        # Build struct expression
        struct_expr = pl.struct([pl.col(col_name).alias(field_name) for field_name, col_name in field_columns.items()])

        # Add struct column
        df_pl = df_pl.with_columns(struct_expr.alias(struct_name))

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(df_pl))

    def concat_strings_over_groups(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        group_by_columns: list[str],
        order_by_columns: list[str],
        separator: str = "|",
    ) -> FrameT:
        """Concatenate string values within groups using Polars window functions.

        Uses sort_by + str.join over window to concatenate values in deterministic order.
        All rows in the same group receive identical concatenated values.
        """
        assert df.implementation == nw.Implementation.POLARS, "Only Polars DataFrames are accepted"
        df_pl = cast(pl.DataFrame | pl.LazyFrame, df.to_native())  # ty: ignore[invalid-argument-type]

        # Use sort_by within the window to ensure deterministic ordering
        # then join all values with the separator
        # Fall back to group_by columns for ordering if no explicit order_by columns
        effective_order_by = order_by_columns if order_by_columns else group_by_columns
        concat_expr = pl.col(source_column).sort_by(*effective_order_by).str.join(separator).over(group_by_columns)
        df_pl = df_pl.with_columns(concat_expr.alias(target_column))

        return cast(FrameT, nw.from_native(df_pl))

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_columns: list[str],
    ) -> FrameT:
        """Keep only the latest row per group based on timestamp columns.

        Args:
            df: Narwhals DataFrame/LazyFrame backed by Polars
            group_columns: Columns to group by (typically ID columns)
            timestamp_columns: Column names to coalesce for ordering (uses first non-null value)

        Returns:
            Narwhals DataFrame/LazyFrame with only the latest row per group
        """
        assert df.implementation == nw.Implementation.POLARS, "Only Polars DataFrames are accepted"

        df_pl = cast(pl.DataFrame | pl.LazyFrame, df.to_native())  # ty: ignore[invalid-argument-type]

        # Create a temporary column for ordering using coalesce
        ordering_expr = pl.coalesce([pl.col(col) for col in timestamp_columns])
        df_pl = df_pl.with_columns(ordering_expr.alias(TEMP_TABLE_NAME))

        result = df_pl.group_by(group_columns).agg(pl.col("*").sort_by(TEMP_TABLE_NAME).last())
        result = result.drop(TEMP_TABLE_NAME)

        # Convert back to Narwhals
        return cast(FrameT, nw.from_native(result))
