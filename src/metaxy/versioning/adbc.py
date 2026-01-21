"""ADBC implementation of VersioningEngine.

This engine works with Arrow data from ADBC stores, converting to Polars for
computations and then back to the appropriate format. It reuses Polars hash
functions for consistency across backends.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import narwhals as nw
import polars as pl
import polars_hash  # noqa: F401  # Registers .nchash and .chash namespaces

from metaxy.models.plan import FeaturePlan
from metaxy.versioning.engine import FrameT, VersioningEngine
from metaxy.versioning.types import HashAlgorithm


class ADBCVersioningEngine(VersioningEngine):
    """Versioning engine for ADBC stores using Polars for computations.

    ADBC stores work with Arrow data natively, but this engine converts to Polars
    for hash computation and transformations to ensure consistency with other backends.
    The PolarsVersioningEngine's hash functions are reused.

    Note:
        This implementation converts Arrow -> Polars -> operations -> Polars.
        The ADBC store will handle final Arrow conversion when writing.
    """

    # Map HashAlgorithm enum to polars-hash functions (same as PolarsVersioningEngine)
    _HASH_FUNCTION_MAP: dict[HashAlgorithm, Callable[[pl.Expr], pl.Expr]] = {
        HashAlgorithm.XXHASH64: lambda expr: expr.nchash.xxhash64(),
        HashAlgorithm.XXHASH32: lambda expr: expr.nchash.xxhash32(),
        HashAlgorithm.WYHASH: lambda expr: expr.nchash.wyhash(),
        HashAlgorithm.SHA256: lambda expr: expr.chash.sha2_256(),
        HashAlgorithm.MD5: lambda expr: expr.nchash.md5(),
    }

    def __init__(self, plan: FeaturePlan) -> None:
        """Initialize the ADBC versioning engine.

        Args:
            plan: Feature plan to track provenance for
        """
        super().__init__(plan)

    @classmethod
    def implementation(cls) -> nw.Implementation:
        """Return Polars implementation since we use Polars for computations."""
        return nw.Implementation.POLARS

    def _to_polars(self, df: nw.DataFrame[Any] | nw.LazyFrame[Any]) -> pl.DataFrame | pl.LazyFrame:
        """Convert Narwhals frame to Polars, handling Arrow tables."""
        # Collect if lazy
        if hasattr(df, "collect"):
            df = df.collect()  # type: ignore[union-attr]

        native_df = df.to_native()

        # Handle Arrow tables by converting to Polars
        if hasattr(native_df, "to_polars"):
            return native_df.to_polars()  # type: ignore[union-attr,return-value]
        elif isinstance(native_df, (pl.DataFrame, pl.LazyFrame)):
            return native_df
        else:
            # Try converting via Arrow if it's an Arrow table
            try:
                import pyarrow as pa

                if isinstance(native_df, pa.Table):
                    # pl.from_arrow returns DataFrame, not Series
                    return pl.from_arrow(native_df)  # type: ignore[return-value]
            except ImportError:
                pass

            raise ValueError(f"Cannot convert {type(native_df)} to Polars")

    def hash_string_column(
        self,
        df: nw.DataFrame[Any] | nw.LazyFrame[Any],
        source_column: str,
        target_column: str,
        hash_algo: HashAlgorithm,
        truncate_length: int | None = None,
    ) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
        """Hash a string column using polars_hash.

        Args:
            df: Narwhals DataFrame (may be backed by Arrow or Polars)
            source_column: Name of string column to hash
            target_column: Name for the new column containing the hash
            hash_algo: Hash algorithm to use
            truncate_length: Optional length to truncate hash to

        Returns:
            Narwhals DataFrame with new hashed column added
        """
        if hash_algo not in self._HASH_FUNCTION_MAP:
            raise ValueError(
                f"Hash algorithm {hash_algo} not supported. Supported: {list(self._HASH_FUNCTION_MAP.keys())}"
            )

        # Convert to Polars
        df_pl = self._to_polars(df)

        # Apply hash using polars_hash
        hash_fn = self._HASH_FUNCTION_MAP[hash_algo]
        hashed = hash_fn(polars_hash.col(source_column)).cast(pl.Utf8)

        # Apply truncation if specified
        if truncate_length is not None:
            hashed = hashed.str.slice(0, truncate_length)

        # Add new column
        df_pl = df_pl.with_columns(hashed.alias(target_column))

        # Convert back to Narwhals
        return nw.from_native(df_pl)  # type: ignore[return-value]

    def record_field_versions(
        self,
        df: nw.DataFrame[Any] | nw.LazyFrame[Any],
        struct_name: str,
        field_columns: dict[str, str],
    ) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
        """Persist field-level versions using a Polars struct column.

        Args:
            df: Narwhals DataFrame
            struct_name: Name for the new struct column
            field_columns: Mapping from struct field names to column names

        Returns:
            Narwhals DataFrame with the struct column added
        """
        # Convert to Polars
        df_pl = self._to_polars(df)

        # Build struct expression
        struct_expr = pl.struct([pl.col(col_name).alias(field_name) for field_name, col_name in field_columns.items()])

        # Add struct column
        df_pl = df_pl.with_columns(struct_expr.alias(struct_name))

        # Convert back to Narwhals
        return nw.from_native(df_pl)  # type: ignore[return-value]

    def concat_strings_over_groups(
        self,
        df: nw.DataFrame[Any] | nw.LazyFrame[Any],
        source_column: str,
        target_column: str,
        group_by_columns: list[str],
        order_by_columns: list[str],
        separator: str = "|",
    ) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
        """Concatenate string values within groups using Polars window functions.

        Args:
            df: Narwhals DataFrame
            source_column: Column to concatenate
            target_column: Name for the new concatenated column
            group_by_columns: Columns to group by
            order_by_columns: Columns to order by within groups
            separator: Separator for concatenation

        Returns:
            Narwhals DataFrame with concatenated column
        """
        # Convert to Polars
        df_pl = self._to_polars(df)

        # Use Polars window function
        effective_order_by = order_by_columns if order_by_columns else group_by_columns
        concat_expr = pl.col(source_column).sort_by(*effective_order_by).str.join(separator).over(group_by_columns)

        df_pl = df_pl.with_columns(concat_expr.alias(target_column))

        return nw.from_native(df_pl)  # type: ignore[return-value]

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_columns: list[str],
    ) -> FrameT:
        """Keep only the latest row per group based on timestamp columns.

        Args:
            df: Narwhals DataFrame or LazyFrame
            group_columns: Columns to group by (typically ID columns)
            timestamp_columns: Column names to coalesce for ordering (uses first non-null value)

        Returns:
            Narwhals DataFrame or LazyFrame (same type as input)
        """
        # Delegate to PolarsVersioningEngine implementation since we're using Polars
        from metaxy.versioning.polars import PolarsVersioningEngine

        return PolarsVersioningEngine.keep_latest_by_group(df, group_columns, timestamp_columns)  # type: ignore[arg-type]
