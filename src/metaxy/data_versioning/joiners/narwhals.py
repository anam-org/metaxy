"""Narwhals implementation of upstream joiner.

Unified joiner that works with any backend (Polars, Ibis/SQL) through Narwhals.
"""

from typing import TYPE_CHECKING, Any

import narwhals as nw

from metaxy.data_versioning.joiners.base import UpstreamJoiner
from metaxy.models.constants import (
    DROPPABLE_SYSTEM_COLUMNS,
)

if TYPE_CHECKING:
    from metaxy.models.feature_spec import BaseFeatureSpec, IDColumns
    from metaxy.models.plan import FeaturePlan


class NarwhalsJoiner(UpstreamJoiner):
    """Joins upstream features using Narwhals LazyFrames.

    Type Parameters:
        TRef = nw.LazyFrame (works with Polars, Ibis, Pandas, PyArrow)

    Strategy:
    - Starts with first upstream feature
    - Sequentially inner joins remaining upstream features on configured ID columns
    - ID columns come from feature spec (default: ["sample_uid"])
    - Supports multi-column joins for composite keys
    - Renames data_version columns to avoid conflicts
    - All operations are lazy (no materialization until collect)
    - Backend-agnostic: same code works for in-memory and SQL backends

    The underlying backend (Polars vs Ibis) is determined by what's wrapped:
    - nw.from_native(pl.LazyFrame) → stays in Polars
    - nw.from_native(ibis.Table) → stays in SQL until collect()
    """

    def join_upstream(
        self,
        upstream_refs: dict[str, nw.LazyFrame[Any]],
        feature_spec: "BaseFeatureSpec[IDColumns]",
        feature_plan: "FeaturePlan",
        upstream_columns: dict[str, tuple[str, ...] | None] | None = None,
        upstream_renames: dict[str, dict[str, str] | None] | None = None,
    ) -> tuple[nw.LazyFrame[Any], dict[str, str]]:
        """Join upstream Narwhals LazyFrames together with column selection/renaming.

        Args:
            upstream_refs: Dict of upstream feature key -> Narwhals LazyFrame
            feature_spec: Feature specification (contains id_columns configuration)
            feature_plan: Feature plan
            upstream_columns: Optional column selection per upstream feature
            upstream_renames: Optional column renaming per upstream feature

        Returns:
            (joined Narwhals LazyFrame, column mapping)
        """
        # Get ID columns from feature spec (default to ["sample_uid"])
        id_columns = feature_spec.id_columns

        # Validate that all upstream features have the required ID columns
        for upstream_key, upstream_ref in upstream_refs.items():
            schema = upstream_ref.collect_schema()
            available_cols = set(schema.names())
            missing_cols = set(id_columns) - available_cols
            if missing_cols:
                raise ValueError(
                    f"Upstream feature '{upstream_key}' is missing required ID columns: {sorted(missing_cols)}. "
                    f"The target feature requires ID columns {id_columns} for joining, but upstream "
                    f"only has columns: {sorted(available_cols)}. "
                    f"Ensure all upstream features have the same ID columns as the target feature."
                )

        if not upstream_refs:
            # No upstream dependencies - root feature
            # Root features should always be provided with samples parameter
            # in resolve_update(), so this shouldn't happen in normal usage.
            # Return empty result for now (mainly for testing)
            import polars as pl

            # Create minimal empty LazyFrame with just ID columns (as Int64)
            empty_data = {col: [] for col in id_columns}
            empty_df = pl.LazyFrame(empty_data)
            return nw.from_native(empty_df), {}

        # Initialize parameters if not provided
        upstream_columns = upstream_columns or {}
        upstream_renames = upstream_renames or {}

        # Use imported constants for system columns
        # Essential columns now include id_columns and data_version
        id_columns_set = set(id_columns)
        system_cols = id_columns_set | {"data_version"}
        system_cols_to_drop = DROPPABLE_SYSTEM_COLUMNS

        # Track all column names to detect conflicts
        all_columns: dict[str, str] = {}  # column_name -> source_feature

        # Process and join upstream features
        upstream_keys = sorted(upstream_refs.keys())
        first_key = upstream_keys[0]
        upstream_mapping = {}

        # Process first upstream feature
        first_ref = upstream_refs[first_key]
        first_columns_spec = upstream_columns.get(first_key)
        first_renames_spec = upstream_renames.get(first_key) or {}

        # Get column names from first upstream
        # We need to collect schema to know available columns
        # Use lazy evaluation where possible
        first_schema = first_ref.collect_schema()
        available_cols = set(first_schema.names())

        # Determine columns to select
        if first_columns_spec is None:
            # Keep all columns (new default behavior) except problematic system columns
            cols_to_select = [c for c in available_cols if c not in system_cols_to_drop]
        elif first_columns_spec == ():
            # Keep only essential system columns
            cols_to_select = [c for c in available_cols if c in system_cols]
        else:
            # Keep specified columns plus essential system columns
            requested = set(first_columns_spec)
            # Filter out problematic system columns even if requested
            requested = requested - system_cols_to_drop
            cols_to_select = list(requested | (available_cols & system_cols))

            # Warn about missing columns
            missing = requested - available_cols
            if missing:
                import warnings

                warnings.warn(
                    f"Columns {missing} requested but not found in upstream feature {first_key}",
                    UserWarning,
                )

        # Build select expressions with renaming for first upstream
        select_exprs = []
        for col in cols_to_select:
            if col == "data_version":
                # Always rename data_version to avoid conflicts
                new_name = f"__upstream_{first_key}__data_version"
                select_exprs.append(nw.col(col).alias(new_name))
                upstream_mapping[first_key] = new_name
            elif col in first_renames_spec:
                # Apply user-specified rename
                new_name = first_renames_spec[col]
                if new_name in all_columns:
                    raise ValueError(
                        f"Column name conflict: '{new_name}' from {first_key} "
                        f"conflicts with column from {all_columns[new_name]}. "
                        f"Use the 'rename' parameter to resolve the conflict."
                    )
                select_exprs.append(nw.col(col).alias(new_name))
                all_columns[new_name] = first_key
            else:
                # Keep original name
                # Don't track ID columns for conflicts (they're expected to be the same)
                if col not in id_columns_set and col in all_columns:
                    raise ValueError(
                        f"Column name conflict: '{col}' appears in both "
                        f"{first_key} and {all_columns[col]}. "
                        f"Use the 'rename' parameter to resolve the conflict."
                    )
                select_exprs.append(nw.col(col))
                if col not in id_columns_set:
                    all_columns[col] = first_key

        joined = first_ref.select(select_exprs)

        # Join remaining upstream features
        for upstream_key in upstream_keys[1:]:
            upstream_ref = upstream_refs[upstream_key]
            columns_spec = upstream_columns.get(upstream_key)
            renames_spec = upstream_renames.get(upstream_key) or {}

            # Get available columns
            schema = upstream_ref.collect_schema()
            available_cols = set(schema.names())

            # Determine columns to select
            if columns_spec is None:
                # Keep all columns except problematic system columns
                cols_to_select = [
                    c for c in available_cols if c not in system_cols_to_drop
                ]
            elif columns_spec == ():
                # Keep only essential system columns
                cols_to_select = [c for c in available_cols if c in system_cols]
            else:
                # Keep specified columns plus essential system columns
                requested = set(columns_spec)
                # Filter out problematic system columns even if requested
                requested = requested - system_cols_to_drop
                cols_to_select = list(requested | (available_cols & system_cols))

                # Warn about missing columns
                missing = requested - available_cols
                if missing:
                    import warnings

                    warnings.warn(
                        f"Columns {missing} requested but not found in upstream feature {upstream_key}",
                        UserWarning,
                    )

            # Build select expressions with renaming
            select_exprs = []
            join_cols = []  # Columns to include in join (exclude ID columns)

            for col in cols_to_select:
                if col in id_columns_set:
                    # Always include ID columns for joining, but don't duplicate them
                    select_exprs.append(nw.col(col))
                elif col == "data_version":
                    # Always rename data_version to avoid conflicts
                    new_name = f"__upstream_{upstream_key}__data_version"
                    select_exprs.append(nw.col(col).alias(new_name))
                    join_cols.append(new_name)
                    upstream_mapping[upstream_key] = new_name
                elif col in renames_spec:
                    # Apply user-specified rename
                    new_name = renames_spec[col]
                    if new_name in all_columns:
                        raise ValueError(
                            f"Column name conflict: '{new_name}' from {upstream_key} "
                            f"conflicts with column from {all_columns[new_name]}. "
                            f"Use the 'rename' parameter to resolve the conflict."
                        )
                    select_exprs.append(nw.col(col).alias(new_name))
                    join_cols.append(new_name)
                    all_columns[new_name] = upstream_key
                else:
                    # Keep original name
                    if col in all_columns:
                        raise ValueError(
                            f"Column name conflict: '{col}' appears in both "
                            f"{upstream_key} and {all_columns[col]}. "
                            f"Use the 'rename' parameter to resolve the conflict."
                        )
                    select_exprs.append(nw.col(col))
                    join_cols.append(col)
                    all_columns[col] = upstream_key

            upstream_renamed = upstream_ref.select(select_exprs)

            # Join with existing data on all ID columns
            joined = joined.join(
                upstream_renamed,
                on=list(id_columns),  # Use configured ID columns (may be composite key)
                how="inner",  # Only rows present in ALL upstream with matching ID columns
            )

        return joined, upstream_mapping
