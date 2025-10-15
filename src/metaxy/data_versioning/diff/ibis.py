"""Ibis implementation of metadata diff resolver.

Note: This implementation converts to Polars for diff computation
since it provides cleaner semantics. For production SQL databases,
consider implementing native SQL diff logic.
"""

from typing import TYPE_CHECKING

import polars as pl

from metaxy.data_versioning.diff.base import DiffResult, MetadataDiffResolver

if TYPE_CHECKING:
    import ibis.expr.types as ir


class IbisDiffResolver(MetadataDiffResolver):
    """Identifies changed rows using Ibis tables via Polars conversion.

    Strategy:
    - Converts Ibis tables to Polars
    - Uses Polars operations for diff (efficient)
    - Returns Polars DataFrames in DiffResult
    """

    def find_changes(
        self,
        target_versions: "ir.Table",
        current_metadata: "ir.Table | None",
    ) -> DiffResult:
        """Find all changes between target and current.

        Args:
            target_versions: Ibis table with calculated data_versions
            current_metadata: Ibis table with current metadata, or None

        Returns:
            DiffResult with three Polars DataFrames
        """
        # Convert to Polars for efficient diff computation
        target_pl = target_versions.to_polars().lazy()

        # Select only sample_id and data_version from target_versions
        # (it may have intermediate joined columns from upstream)
        target_pl = target_pl.select(["sample_id", "data_version"])

        if current_metadata is None:
            # No existing metadata - all target rows are new
            empty_df = pl.DataFrame({"sample_id": [], "data_version": []})
            return DiffResult(
                added=target_pl.collect(),
                changed=empty_df,
                removed=empty_df,
            )

        current_pl = current_metadata.to_polars().lazy()

        # Keep only sample_id and data_version from current for comparison
        current_comparison = current_pl.select(
            ["sample_id", pl.col("data_version").alias("__current_data_version")]
        )

        # LEFT JOIN target with current
        compared = target_pl.join(
            current_comparison,
            on="sample_id",
            how="left",
        )

        # Build lazy queries for each category
        added_lazy = (
            compared.filter(pl.col("__current_data_version").is_null())
            .drop("__current_data_version")
            .select(["sample_id", "data_version"])
        )

        changed_lazy = (
            compared.filter(
                pl.col("__current_data_version").is_not_null()
                & (pl.col("data_version") != pl.col("__current_data_version"))
            )
            .drop("__current_data_version")
            .select(["sample_id", "data_version"])
        )

        removed_lazy = current_pl.join(
            target_pl.select("sample_id"),
            on="sample_id",
            how="anti",
        ).select(["sample_id", "data_version"])

        # Collect all three at once
        added_df, changed_df, removed_df = pl.collect_all(
            [added_lazy, changed_lazy, removed_lazy]
        )

        return DiffResult(
            added=added_df,
            changed=changed_df,
            removed=removed_df,
        )
