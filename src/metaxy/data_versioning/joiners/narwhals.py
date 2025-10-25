"""Narwhals implementation of upstream joiner.

Unified joiner that works with any backend (Polars, Ibis/SQL) through Narwhals.
"""

from typing import TYPE_CHECKING, Any

import narwhals as nw

from metaxy.data_versioning.joiners.base import UpstreamJoiner
from metaxy.models.constants import DROPPABLE_SYSTEM_COLUMNS, ESSENTIAL_SYSTEM_COLUMNS

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


class NarwhalsJoiner(UpstreamJoiner):
    """Joins upstream features using Narwhals LazyFrames.

    Type Parameters:
        TRef = nw.LazyFrame (works with Polars, Ibis, Pandas, PyArrow)

    Strategy:
    - Starts with first upstream feature
    - Sequentially inner joins remaining upstream features on sample_uid
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
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
        upstream_columns: dict[str, tuple[str, ...] | None] | None = None,
        upstream_renames: dict[str, dict[str, str] | None] | None = None,
    ) -> tuple[nw.LazyFrame[Any], dict[str, str]]:
        """Join upstream Narwhals LazyFrames together with column selection/renaming.

        Args:
            upstream_refs: Dict of upstream feature key -> Narwhals LazyFrame
            feature_spec: Feature specification
            feature_plan: Feature plan
            upstream_columns: Optional column selection per upstream feature
            upstream_renames: Optional column renaming per upstream feature

        Returns:
            (joined Narwhals LazyFrame, column mapping)
        """
        if not upstream_refs:
            # No upstream dependencies - source feature
            # Return empty LazyFrame with just sample_uid column (with proper type)
            import polars as pl

            # Create empty frame with explicit Int64 type for sample_uid
            # This ensures it's not NULL-typed which would fail with Ibis backends
            empty_df = pl.LazyFrame(
                {"sample_uid": pl.Series("sample_uid", [], dtype=pl.Int64)}
            )
            return nw.from_native(empty_df), {}

        # Initialize parameters if not provided
        upstream_columns = upstream_columns or {}
        upstream_renames = upstream_renames or {}

        # Use imported constants for system columns
        system_cols = ESSENTIAL_SYSTEM_COLUMNS
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
                if col != "sample_uid" and col in all_columns:
                    raise ValueError(
                        f"Column name conflict: '{col}' appears in both "
                        f"{first_key} and {all_columns[col]}. "
                        f"Use the 'rename' parameter to resolve the conflict."
                    )
                select_exprs.append(nw.col(col))
                if col != "sample_uid":
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
            join_cols = []  # Columns to include in join (exclude sample_uid)

            for col in cols_to_select:
                if col == "sample_uid":
                    # Always include sample_uid for joining, but don't duplicate it
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

            # Join with existing data
            joined = joined.join(
                upstream_renamed,
                on="sample_uid",
                how="inner",  # Only sample_uids present in ALL upstream
            )

        return joined, upstream_mapping
