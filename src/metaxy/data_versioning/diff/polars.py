"""Polars implementation of metadata diff resolver."""

import polars as pl

from metaxy.data_versioning.diff.base import DiffResult, MetadataDiffResolver


class PolarsDiffResolver(MetadataDiffResolver):
    """Identifies changed rows using Polars operations.

    Strategy:
    - Categorizes changes into added, changed, and removed
    - Uses LEFT/RIGHT JOINs to identify each category
    - Materializes once and splits into three DataFrames (efficient)
    """

    def find_changes(
        self,
        target_versions: pl.LazyFrame,
        current_metadata: pl.LazyFrame | None,
    ) -> DiffResult:
        """Find all changes between target and current.

        Args:
            target_versions: LazyFrame with calculated data_versions
            current_metadata: LazyFrame with current metadata, or None

        Returns:
            DiffResult with three materialized DataFrames
        """
        # Select only sample_id and data_version from target_versions
        # (it may have intermediate joined columns from upstream)
        target_versions = target_versions.select(["sample_id", "data_version"])

        if current_metadata is None:
            # No existing metadata - all target rows are new, nothing removed/changed
            empty_df = pl.DataFrame({"sample_id": [], "data_version": []})
            return DiffResult(
                added=target_versions.collect(),
                changed=empty_df,
                removed=empty_df,
            )

        # Join target with current to categorize changes
        # Keep only sample_id and data_version from current for comparison
        current_comparison = current_metadata.select(
            ["sample_id", pl.col("data_version").alias("__current_data_version")]
        )

        # LEFT JOIN target with current
        compared = target_versions.join(
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

        removed_lazy = current_metadata.join(
            target_versions.select("sample_id"),
            on="sample_id",
            how="anti",
        ).select(["sample_id", "data_version"])

        # Collect all three at once (single execution for shared computations)
        added_df, changed_df, removed_df = pl.collect_all(
            [added_lazy, changed_lazy, removed_lazy]
        )

        return DiffResult(
            added=added_df,
            changed=changed_df,
            removed=removed_df,
        )
