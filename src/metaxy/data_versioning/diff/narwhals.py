"""Narwhals implementation of metadata diff resolver.

Unified diff resolver that works with any backend (Polars, Ibis/SQL) through Narwhals.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw

from metaxy.data_versioning.diff.base import (
    LazyIncrement,
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
        target_provenance: nw.LazyFrame[Any],
        current_metadata: nw.LazyFrame[Any] | None,
        id_columns: Sequence[str],
    ) -> LazyIncrement:
        """Find all changes between target and current.

        Args:
            target_provenance: Narwhals LazyFrame with calculated field_provenance
            current_metadata: Narwhals LazyFrame with current metadata, or None.
                Should be pre-filtered by feature_version at caller level if needed.
            id_columns: ID columns to use for comparison (required - from feature spec)

        Returns:
            LazyIncrement with three lazy Narwhals frames (caller materializes if needed)
        """
        id_columns = list(id_columns)

        # id_columns must be explicitly provided from the feature spec
        if not id_columns:
            raise ValueError(
                "id_columns must be provided to find_changes. "
                "These should come from the feature spec's id_columns property."
            )

        # Select only ID columns and provenance_by_field from target_provenance
        # (it may have intermediate joined columns from upstream)
        target_provenance = target_provenance.select(
            id_columns + ["provenance_by_field"]
        )

        if current_metadata is None:
            # No existing metadata - all target rows are new
            # Create empty LazyFrame with proper schema
            import polars as pl

            # Create empty schema with ID columns
            schema = {col: [] for col in id_columns}
            schema["provenance_by_field"] = []
            empty_lazy = nw.from_native(pl.LazyFrame(schema))

            return LazyIncrement(
                added=target_provenance,
                changed=empty_lazy,
                removed=empty_lazy,
            )

        # Keep only ID columns and provenance_by_field from current for comparison
        select_cols = id_columns + [
            nw.col("provenance_by_field").alias("__current_provenance_by_field")
        ]
        current_comparison = current_metadata.select(*select_cols)

        # LEFT JOIN target with current on ID columns
        compared = target_provenance.join(
            current_comparison,
            on=id_columns,
            how="left",
        )

        # Build lazy queries for each category
        added_lazy = (
            compared.filter(nw.col("__current_provenance_by_field").is_null())
            .drop("__current_provenance_by_field")
            .select(id_columns + ["provenance_by_field"])
        )

        changed_lazy = (
            compared.filter(
                ~nw.col("__current_provenance_by_field").is_null()
                & (
                    nw.col("provenance_by_field")
                    != nw.col("__current_provenance_by_field")
                )
            )
            .drop("__current_provenance_by_field")
            .select(id_columns + ["provenance_by_field"])
        )

        removed_lazy = current_metadata.join(
            target_provenance.select(id_columns),
            on=id_columns,
            how="anti",
        ).select(id_columns + ["provenance_by_field"])

        # Return lazy frames - caller will materialize if needed
        return LazyIncrement(
            added=added_lazy,
            changed=changed_lazy,
            removed=removed_lazy,
        )
