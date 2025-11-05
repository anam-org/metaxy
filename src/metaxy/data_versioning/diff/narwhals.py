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

        # Always preserve all columns from target_provenance
        # Column selection should be handled by FeatureDep.columns during join phase

        # Create empty LazyFrame with proper schema (used in multiple places)
        import polars as pl

        # Create empty schema with ID columns
        schema = {col: [] for col in id_columns}
        schema["provenance_by_field"] = []
        empty_lazy = nw.from_native(pl.LazyFrame(schema))

        if current_metadata is None:
            # No existing metadata - all target rows are new

            return LazyIncrement(
                added=target_provenance,
                changed=empty_lazy,
                removed=empty_lazy,
            )

        # AUTOMATIC AGGREGATION: Always apply group_by(id_columns).first() on current metadata
        # This ensures consistency with the joiner's automatic aggregation and handles all
        # relationship types automatically:
        # - 1->1: Query optimizer recognizes grouping on unique keys (no-op)
        # - 1->N: Groups child records by parent, taking first provenance
        # - N->1: Groups by target ID columns (future: aggregated provenance)
        # - M->N: Handles complex relationships naturally

        # Get available columns in both target and current metadata
        target_schema = target_provenance.collect_schema()
        target_cols = set(target_schema.names())
        available_cols = set(current_metadata.collect_schema().names())

        # IMPORTANT: For 1:N relationships, target (from parent) may not have all child ID columns
        # Only use ID columns that exist in both target and current for joining
        available_id_cols = [col for col in id_columns if col in target_cols]

        if len(available_id_cols) < len(id_columns):
            # This is likely a 1:N expansion relationship
            # Use only the parent ID columns that exist in target
            effective_id_columns = available_id_cols
        else:
            # All ID columns available - normal case
            effective_id_columns = list(id_columns)

        # Special case: For expansion relationships with no common ID columns,
        # we need to check if parent records already have child records.
        # Extract parent ID columns from child metadata to see which parents are covered
        if not effective_id_columns:
            # For expansion relationships, we need to find which parent IDs already have children
            # The target has parent IDs, current has child IDs
            # We want to find parent records that DON'T have children yet

            # Find common ID columns between target and current that might be parent IDs
            # Usually the parent ID columns are a subset of child ID columns
            # For example: parent has [video_id], child has [video_id, video_chunk_id]
            # Use a heuristic: only consider columns that look like IDs (end with "_id" or are explicitly named "id")
            parent_id_cols = []
            for col in target_cols:
                if col in available_cols and col not in [
                    "provenance_by_field",
                    "feature_version",
                    "snapshot_version",
                    "metaxy_provenance_by_field",
                ]:
                    # Check if this looks like an ID column
                    if (
                        col.endswith("_id")
                        or col == "id"
                        or col.endswith("_uid")
                        or col == "uid"
                    ):
                        # This column exists in both and looks like an ID - likely a parent ID
                        parent_id_cols.append(col)

            if parent_id_cols:
                # We found potential parent ID columns
                # Group current child records by parent IDs to get unique parent IDs that have children
                [nw.col(col).max().alias(col) for col in parent_id_cols]
                existing_parents = current_metadata.select(parent_id_cols).unique()

                # Anti-join: find target parent records that don't have children
                added_lazy = target_provenance.join(
                    existing_parents,
                    on=parent_id_cols,
                    how="anti",
                )

                # Changed: empty (can't determine changes without matching IDs)
                changed_lazy = empty_lazy

                # Removed: all current child records (they have child IDs we can't match)
                # This is expected - child records are preserved, we're only adding new parents
                removed_lazy = empty_lazy  # Don't remove existing children

                return LazyIncrement(
                    added=added_lazy,
                    changed=changed_lazy,
                    removed=removed_lazy,
                )
            else:
                # No common columns at all - treat all target as added
                # This shouldn't normally happen in well-designed expansion relationships
                return LazyIncrement(
                    added=target_provenance,
                    changed=empty_lazy,
                    removed=empty_lazy,  # Don't remove existing data
                )

        # Use .unique() instead of group_by().agg() to avoid order-dependent operations
        # This is more semantically correct: we want unique records by ID columns,
        # and all duplicate records should have identical provenance values

        # Track child-specific columns (defined at this scope for later use)
        child_specific_cols: list[str] = []

        if effective_id_columns:
            # Find any missing child-specific ID columns (for 1:N)
            for col in id_columns:
                if col not in effective_id_columns and col in available_cols:
                    child_specific_cols.append(col)

            # Select only the columns we need for comparison to avoid duplicates
            # We need: effective ID columns + provenance_by_field + child-specific columns
            cols_to_select = list(effective_id_columns) + ["provenance_by_field"]
            cols_to_select.extend(child_specific_cols)

            # Get unique records by effective ID columns
            # IMPORTANT: Select columns first, then unique to ensure we don't have duplicate columns
            current_comparison = current_metadata.select(cols_to_select).unique(
                subset=effective_id_columns
            )

            # Rename the provenance column for comparison
            current_comparison = current_comparison.with_columns(
                nw.col("provenance_by_field").alias("__current_provenance_by_field")
            )

            # Rename child-specific columns to avoid conflicts
            for col in child_specific_cols:
                current_comparison = current_comparison.with_columns(
                    nw.col(col).alias(f"__child_{col}")
                )
        else:
            # No effective ID columns - this shouldn't normally happen
            # but can occur in edge cases with expansion relationships
            # Just use the current metadata as-is with renamed provenance column
            current_comparison = current_metadata.with_columns(
                nw.col("provenance_by_field").alias("__current_provenance_by_field")
            )

        # Rename child-specific columns back to their original names
        if child_specific_cols:
            rename_dict = {f"__child_{col}": col for col in child_specific_cols}
            current_comparison = current_comparison.rename(rename_dict)

        # LEFT JOIN target with current on effective ID columns
        compared = target_provenance.join(
            current_comparison,
            on=effective_id_columns,
            how="left",
        )

        # Build lazy queries for each category
        # Always preserve all columns from target_provenance in added/changed results
        added_lazy = compared.filter(
            nw.col("__current_provenance_by_field").is_null()
        ).drop("__current_provenance_by_field")

        changed_lazy = compared.filter(
            ~nw.col("__current_provenance_by_field").is_null()
            & (nw.col("provenance_by_field") != nw.col("__current_provenance_by_field"))
        ).drop("__current_provenance_by_field")

        # For removed, we only have ID columns and provenance_by_field
        # since the other columns aren't available in current_metadata
        # For 1:N, use effective ID columns for anti-join
        removed_lazy = current_metadata.join(
            target_provenance.select(effective_id_columns),
            on=effective_id_columns,
            how="anti",
        )
        # Select columns carefully to avoid duplicates
        # All id_columns should already be present from current_metadata
        cols_to_select = list(id_columns) + ["provenance_by_field"]
        removed_lazy = removed_lazy.select(cols_to_select)

        # Return lazy frames - caller will materialize if needed
        return LazyIncrement(
            added=added_lazy,
            changed=changed_lazy,
            removed=removed_lazy,
        )
