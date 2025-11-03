"""Narwhals implementation of upstream joiner.

Unified joiner that works with any backend (Polars, Ibis/SQL) through Narwhals.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw

from metaxy.data_versioning.joiners.base import UpstreamJoiner
from metaxy.models.constants import (
    DROPPABLE_SYSTEM_COLUMNS,
)

if TYPE_CHECKING:
    from metaxy.models.feature_spec import BaseFeatureSpec
    from metaxy.models.plan import FeaturePlan

# Constants for upstream provenance column naming pattern
UPSTREAM_PROVENANCE_PREFIX = "__upstream_"
UPSTREAM_PROVENANCE_SUFFIX = "__provenance_by_field"


class NarwhalsJoiner(UpstreamJoiner):
    """Joins upstream features using Narwhals LazyFrames.

    Type Parameters:
        TRef = nw.LazyFrame (works with Polars, Ibis, Pandas, PyArrow)

    Strategy:
    - Starts with first upstream feature
    - Sequentially inner joins remaining upstream features on configured ID columns
    - ID columns come from feature spec (default: ["sample_uid"])
    - Supports multi-column joins for composite keys
    - Renames provenance_by_field columns to avoid conflicts
    - All operations are lazy (no materialization until collect)
    - Backend-agnostic: same code works for in-memory and SQL backends

    The underlying backend (Polars vs Ibis) is determined by what's wrapped:
    - nw.from_native(pl.LazyFrame) → stays in Polars
    - nw.from_native(ibis.Table) → stays in SQL until collect()
    """

    def _validate_id_columns_for_upstream(
        self,
        upstream_key: str,
        upstream_ref: nw.LazyFrame[Any],
        id_columns: Sequence[str],
        id_mapping: dict[str, str] | None,
    ) -> None:
        """Validate that upstream has required ID columns or mapped columns.

        Args:
            upstream_key: Key identifying the upstream feature
            upstream_ref: Upstream feature's LazyFrame
            id_columns: Target feature's ID columns
            id_mapping: Optional mapping of target ID columns to upstream columns

        Raises:
            ValueError: If required columns are missing or mapping is invalid
        """
        schema = upstream_ref.collect_schema()
        available_cols = set(schema.names())

        if id_mapping:
            # Validate that all mapped upstream columns exist
            required_upstream_cols = set(id_mapping.values())
            missing_cols = required_upstream_cols - available_cols
            if missing_cols:
                raise ValueError(
                    f"Upstream feature '{upstream_key}' is missing mapped ID columns: {sorted(missing_cols)}. "
                    f"The id_columns_mapping specifies upstream columns {sorted(required_upstream_cols)}, "
                    f"but upstream only has columns: {sorted(available_cols)}."
                )
            # Also validate that mapped target columns are valid
            mapped_target_cols = set(id_mapping.keys())
            invalid_target_cols = mapped_target_cols - set(id_columns)
            if invalid_target_cols:
                raise ValueError(
                    f"Invalid target ID columns in mapping for upstream '{upstream_key}': {sorted(invalid_target_cols)}. "
                    f"Target feature has ID columns {id_columns}, but mapping references {sorted(invalid_target_cols)}."
                )
        else:
            # No mapping provided - all ID columns must match exactly
            missing_cols = set(id_columns) - available_cols
            if missing_cols:
                raise ValueError(
                    f"Upstream feature '{upstream_key}' is missing required ID columns: {sorted(missing_cols)}. "
                    f"The target feature requires ID columns {id_columns} for joining, but upstream "
                    f"only has columns: {sorted(available_cols)}. "
                    f"Either ensure all upstream features have the same ID columns as the target feature, "
                    f"or use id_columns_mapping to specify the column correspondence."
                )

    def _determine_columns_to_select(
        self,
        upstream_key: str,
        available_cols: set[str],
        columns_spec: tuple[str, ...] | None,
        id_columns_set: set[str],
        system_cols: set[str],
        system_cols_to_drop: frozenset[str],
        id_mapping: dict[str, str] | None = None,
    ) -> list[str]:
        """Determine which columns to select from upstream.

        Args:
            upstream_key: Key identifying the upstream feature
            available_cols: Set of available column names in upstream
            columns_spec: Optional specification of columns to select
            id_columns_set: Set of ID column names
            system_cols: Set of essential system columns
            system_cols_to_drop: Set of system columns to drop
            id_mapping: Optional ID column mapping

        Returns:
            List of column names to select
        """
        if columns_spec is None:
            # Keep all columns (new default behavior) except problematic system columns
            cols_to_select = [c for c in available_cols if c not in system_cols_to_drop]
        elif columns_spec == ():
            # Keep only essential system columns
            cols_to_select = [c for c in available_cols if c in system_cols]
        else:
            # Keep specified columns plus essential system columns
            requested = set(columns_spec)
            # Filter out problematic system columns even if requested
            requested = requested - system_cols_to_drop

            # If id_mapping exists, ensure mapped upstream ID columns are included
            if id_mapping:
                # Add upstream ID columns that are mapped
                for target_col, upstream_col in id_mapping.items():
                    if upstream_col in available_cols:
                        requested.add(upstream_col)

            cols_to_select = list(requested | (available_cols & system_cols))

            # Warn about missing columns
            missing = requested - available_cols
            if missing:
                import warnings

                warnings.warn(
                    f"Columns {missing} requested but not found in upstream feature {upstream_key}",
                    UserWarning,
                )

        return cols_to_select

    def _build_select_expressions(
        self,
        upstream_key: str,
        cols_to_select: list[str],
        renames_spec: dict[str, str] | None,
        id_mapping: dict[str, str] | None,
        id_columns_set: set[str],
        all_columns: dict[str, str],
        is_first: bool = False,
    ) -> tuple[list[Any], str | None]:
        """Build select expressions with renaming for upstream columns.

        Args:
            upstream_key: Key identifying the upstream feature
            cols_to_select: List of columns to select
            renames_spec: Optional user-specified rename mapping
            id_mapping: Optional ID column mapping
            id_columns_set: Set of ID column names
            all_columns: Dict tracking all column names to detect conflicts
            is_first: Whether this is the first upstream (affects conflict tracking)

        Returns:
            Tuple of (select expressions, optional upstream provenance mapping)
        """
        select_exprs = []
        upstream_mapping = None

        # Ensure renames_spec is a dict
        renames_spec = renames_spec or {}

        # Build reverse mapping for ID column renaming if needed
        upstream_to_target = {}
        if id_mapping:
            upstream_to_target = {v: k for k, v in id_mapping.items()}

        for col in cols_to_select:
            if col == "provenance_by_field":
                # Always rename provenance_by_field to avoid conflicts
                new_name = f"{UPSTREAM_PROVENANCE_PREFIX}{upstream_key}{UPSTREAM_PROVENANCE_SUFFIX}"
                select_exprs.append(nw.col(col).alias(new_name))
                upstream_mapping = new_name
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
                all_columns[new_name] = upstream_key
            elif col in upstream_to_target:
                # This is a mapped ID column - rename to target name if different
                target_col = upstream_to_target[col]
                if col != target_col:
                    select_exprs.append(nw.col(col).alias(target_col))
                else:
                    select_exprs.append(nw.col(col))
                # ID columns don't count for conflicts
            else:
                # Keep original name
                # Don't track ID columns for conflicts (they're expected to be the same)
                if col not in id_columns_set:
                    if col in all_columns and not is_first:
                        raise ValueError(
                            f"Column name conflict: '{col}' appears in both "
                            f"{upstream_key} and {all_columns[col]}. "
                            f"Use the 'rename' parameter to resolve the conflict."
                        )
                    all_columns[col] = upstream_key
                select_exprs.append(nw.col(col))

        return select_exprs, upstream_mapping

    def _prepare_upstream_for_join(
        self,
        upstream_key: str,
        upstream_ref: nw.LazyFrame[Any],
        columns_spec: tuple[str, ...] | None,
        renames_spec: dict[str, str] | None,
        id_mapping: dict[str, str] | None,
        id_columns: Sequence[str],
        id_columns_set: set[str],
        system_cols: set[str],
        system_cols_to_drop: frozenset[str],
        all_columns: dict[str, str],
        is_first: bool = False,
    ) -> tuple[nw.LazyFrame[Any], str | None, list[str]]:
        """Prepare a single upstream for joining with proper column selection and renaming.

        Args:
            upstream_key: Key identifying the upstream feature
            upstream_ref: Upstream feature's LazyFrame
            columns_spec: Optional specification of columns to select
            renames_spec: Optional user-specified rename mapping
            id_mapping: Optional ID column mapping
            id_columns: Target feature's ID columns
            id_columns_set: Set of ID column names
            system_cols: Set of essential system columns
            system_cols_to_drop: Set of system columns to drop
            all_columns: Dict tracking all column names to detect conflicts
            is_first: Whether this is the first upstream

        Returns:
            Tuple of (prepared LazyFrame, upstream provenance mapping, join columns)
        """
        # Get available columns
        schema = upstream_ref.collect_schema()
        available_cols = set(schema.names())

        # Determine columns to select
        cols_to_select = self._determine_columns_to_select(
            upstream_key,
            available_cols,
            columns_spec,
            id_columns_set,
            system_cols,
            system_cols_to_drop,
            id_mapping,
        )

        # Build select expressions
        select_exprs, upstream_mapping = self._build_select_expressions(
            upstream_key,
            cols_to_select,
            renames_spec,
            id_mapping,
            id_columns_set,
            all_columns,
            is_first,
        )

        # Apply selections
        prepared = upstream_ref.select(select_exprs)

        # Determine join columns
        if id_mapping:
            # Use mapped columns for join
            join_cols = list(id_mapping.keys())
        else:
            # Use all ID columns
            join_cols = list(id_columns)  # Ensure it's always a list

        return prepared, upstream_mapping, join_cols

    def join_upstream(
        self,
        upstream_refs: dict[str, nw.LazyFrame[Any]],
        feature_spec: "BaseFeatureSpec",
        feature_plan: "FeaturePlan",
        upstream_columns: dict[str, tuple[str, ...] | None] | None = None,
        upstream_renames: dict[str, dict[str, str] | None] | None = None,
        upstream_id_mappings: dict[str, dict[str, str] | None] | None = None,
    ) -> tuple[nw.LazyFrame[Any], dict[str, str]]:
        """Join upstream Narwhals LazyFrames together with column selection/renaming and flexible ID mapping.

        Args:
            upstream_refs: Dict of upstream feature key -> Narwhals LazyFrame
            feature_spec: Feature specification (contains id_columns configuration)
            feature_plan: Feature plan
            upstream_columns: Optional column selection per upstream feature
            upstream_renames: Optional column renaming per upstream feature
            upstream_id_mappings: Optional ID column mappings per upstream feature

        Returns:
            (joined Narwhals LazyFrame, column mapping)
        """
        # Get ID columns from feature spec (default to ["sample_uid"])
        id_columns = feature_spec.id_columns

        # Initialize parameters if not provided
        upstream_id_mappings = upstream_id_mappings or {}
        upstream_columns = upstream_columns or {}
        upstream_renames = upstream_renames or {}

        # Validate all upstreams have required ID columns
        for upstream_key, upstream_ref in upstream_refs.items():
            id_mapping = upstream_id_mappings.get(upstream_key)
            self._validate_id_columns_for_upstream(
                upstream_key, upstream_ref, id_columns, id_mapping
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

        # Setup system column sets
        id_columns_set = set(id_columns)
        system_cols = id_columns_set | {"provenance_by_field"}
        system_cols_to_drop = DROPPABLE_SYSTEM_COLUMNS

        # Track all column names to detect conflicts
        all_columns: dict[str, str] = {}  # column_name -> source_feature

        # Process upstream features in sorted order for deterministic behavior
        upstream_keys = sorted(upstream_refs.keys())
        upstream_mapping = {}

        # Process all upstreams uniformly
        joined: nw.LazyFrame[Any] | None = None
        for idx, upstream_key in enumerate(upstream_keys):
            upstream_ref = upstream_refs[upstream_key]
            columns_spec = upstream_columns.get(upstream_key)
            renames_spec = upstream_renames.get(upstream_key)
            id_mapping = upstream_id_mappings.get(upstream_key)

            # Prepare upstream for joining
            prepared, provenance_mapping, join_cols = self._prepare_upstream_for_join(
                upstream_key=upstream_key,
                upstream_ref=upstream_ref,
                columns_spec=columns_spec,
                renames_spec=renames_spec,
                id_mapping=id_mapping,
                id_columns=id_columns,
                id_columns_set=id_columns_set,
                system_cols=system_cols,
                system_cols_to_drop=system_cols_to_drop,
                all_columns=all_columns,
                is_first=(idx == 0),
            )

            # Track upstream provenance mapping
            if provenance_mapping:
                upstream_mapping[upstream_key] = provenance_mapping

            if idx == 0:
                # First upstream becomes the base
                joined = prepared
            else:
                # Join with existing data (joined is guaranteed to be non-None here)
                assert joined is not None  # For type checker
                joined = joined.join(
                    prepared,
                    on=join_cols,  # Use join columns determined during preparation
                    how="inner",  # Only rows present in ALL upstreams
                )

        assert joined is not None  # We know we have at least one upstream
        return joined, upstream_mapping
