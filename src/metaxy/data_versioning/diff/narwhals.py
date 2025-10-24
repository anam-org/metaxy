"""Narwhals implementation of metadata diff resolver.

Unified diff resolver that works with any backend (Polars, Ibis/SQL) through Narwhals.
"""

from typing import TYPE_CHECKING

import narwhals as nw

from metaxy.data_versioning.diff.base import (
    LazyDiffResult,
    MetadataDiffResolver,
)

if TYPE_CHECKING:
    pass


class NarwhalsDiffResolver(MetadataDiffResolver):
    """Identifies changed rows using Narwhals operations.

    Uses Narwhals LazyFrames (works with Polars, Ibis, Pandas, PyArrow)

    Strategy:
    - Categorizes changes into added, changed, and removed
    - Uses LEFT/RIGHT JOINs to identify each category
    - Materializes once and splits into three DataFrames (efficient)
    - Backend-agnostic: same code works for in-memory and SQL backends

    The underlying backend (Polars vs Ibis) determines execution:
    - Polars backend → operations happen in-memory
    - Ibis backend → operations happen in SQL database
    """

    def find_changes(
        self,
        target_versions: nw.LazyFrame,
        current_metadata: nw.LazyFrame | None,
    ) -> LazyDiffResult:
        """Find all changes between target and current.

        Args:
            target_versions: Narwhals LazyFrame with calculated data_versions
            current_metadata: Narwhals LazyFrame with current metadata, or None.
                Should be pre-filtered by feature_version at caller level if needed.

        Returns:
            LazyDiffResult with three lazy Narwhals frames (caller materializes if needed)
        """
        # Select only sample_id and data_version from target_versions
        # (it may have intermediate joined columns from upstream)
        target_versions = target_versions.select(["sample_id", "data_version"])

        if current_metadata is None:
            # No existing metadata - all target rows are new
            # Create empty LazyFrame with proper schema
            import polars as pl

            empty_lazy = nw.from_native(
                pl.LazyFrame({"sample_id": [], "data_version": []})
            )

            return LazyDiffResult(
                added=target_versions,
                changed=empty_lazy,
                removed=empty_lazy,
            )

        # Keep only sample_id and data_version from current for comparison
        current_comparison = current_metadata.select(
            "sample_id", nw.col("data_version").alias("__current_data_version")
        )

        # LEFT JOIN target with current
        compared = target_versions.join(
            current_comparison,
            on="sample_id",
            how="left",
        )

        # Build lazy queries for each category
        added_lazy = (
            compared.filter(nw.col("__current_data_version").is_null())
            .drop("__current_data_version")
            .select("sample_id", "data_version")
        )

        changed_lazy = (
            compared.filter(
                ~nw.col("__current_data_version").is_null()
                & (nw.col("data_version") != nw.col("__current_data_version"))
            )
            .drop("__current_data_version")
            .select("sample_id", "data_version")
        )

        removed_lazy = current_metadata.join(
            target_versions.select("sample_id"),
            on="sample_id",
            how="anti",
        ).select("sample_id", "data_version")

        # Return lazy frames - caller will materialize if needed
        return LazyDiffResult(
            added=added_lazy,
            changed=changed_lazy,
            removed=removed_lazy,
        )
