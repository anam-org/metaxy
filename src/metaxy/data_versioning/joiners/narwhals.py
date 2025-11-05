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

    def _aggregate_provenance_struct(self, col_name: str) -> Any:
        """Aggregate provenance struct fields by hashing values together.

        For N->1 relationships, this hashes together all parent provenances
        for each field in the struct. For 1->1 and 1->N, all values are
        identical so the result is the same as taking first.

        Args:
            col_name: Name of the provenance column (struct type)

        Returns:
            Aggregation expression that hashes struct fields together
        """
        # This method is not currently used since we handle aggregation inline
        # with backend-specific code. Keeping it for documentation purposes.
        # In group_by().agg() context, we need:
        #   pl.col(col_name).list.map_elements(...)
        # not:
        #   pl.col(col_name).list().map_elements(...)
        #
        # For now, just return a first() aggregation as a fallback
        import narwhals as nw

        return nw.col(col_name).first().alias(col_name)

    def _hash_struct_list(self, struct_list: list[Any]) -> dict[str, Any]:
        """Hash together struct fields from a list of structs.

        Args:
            struct_list: List of struct dictionaries

        Returns:
            Single struct with fields hashed together
        """
        if not struct_list:
            return {}

        # If only one struct, return as-is
        if len(struct_list) == 1:
            first = struct_list[0]
            # Handle Polars Series case
            if hasattr(first, "__len__") and not isinstance(first, (dict, str)):
                # It's likely a Series, extract the value
                return first if isinstance(first, dict) else {}
            # Ensure we only return dict, not str or other types
            if isinstance(first, dict):
                return first
            return {}

        # Multiple structs - hash each field's values together
        result = {}

        # Get field names from first non-null struct
        first_struct = None
        for s in struct_list:
            # Handle Series or other non-dict types
            if s is not None and not isinstance(s, (dict, str)):
                # Skip non-dict items that aren't None
                continue
            if s and isinstance(s, dict):
                first_struct = s
                break

        if not first_struct or not isinstance(first_struct, dict):
            return {}

        for field_name in first_struct.keys():
            # Collect all values for this field across structs
            values = []
            for struct in struct_list:
                # Handle None and non-dict types
                if struct is None:
                    continue
                if not isinstance(struct, dict):
                    continue
                if field_name in struct:
                    values.append(str(struct[field_name]))

            # Sort for deterministic ordering
            values.sort()

            if len(values) == 1:
                # Single value - use as-is
                # Find the first non-None struct that has this field
                for struct in struct_list:
                    if struct and isinstance(struct, dict) and field_name in struct:
                        result[field_name] = struct[field_name]
                        break
            else:
                # Multiple values - hash them together
                # Try xxhash first (preferred), fallback to hashlib
                try:
                    import xxhash  # pyright: ignore[reportMissingImports]

                    hasher = xxhash.xxh64()
                    for v in values:
                        hasher.update(v.encode())
                    result[field_name] = hasher.hexdigest()
                except ImportError:
                    # Fallback to standard library
                    import hashlib

                    hasher = hashlib.sha256()
                    for v in values:
                        hasher.update(v.encode())
                    # Use first 16 chars to match xxhash64 length
                    result[field_name] = hasher.hexdigest()[:16]

        return result

    def _validate_id_columns_for_upstream(
        self,
        upstream_key: str,
        upstream_ref: nw.LazyFrame[Any],
        id_columns: Sequence[str],
        id_mapping: dict[str, str] | None,
        all_mapped_columns: set[str],
        lineage_type: str | None = None,
        parent_id_columns: Sequence[str] | None = None,
        renames_spec: dict[str, str] | None = None,
    ) -> None:
        """Validate that upstream has required ID columns or mapped columns.

        Args:
            upstream_key: Key identifying the upstream feature
            upstream_ref: Upstream feature's LazyFrame
            id_columns: Target feature's ID columns
            id_mapping: Optional mapping of target ID columns to upstream columns
            all_mapped_columns: Set of all ID columns mapped by any upstream
            lineage_type: Type of lineage relationship (e.g., "1:N" for expansion)
            parent_id_columns: For expansion relationships, the parent ID columns to validate
            renames_spec: Optional rename mapping that will be applied

        Raises:
            ValueError: If required columns are missing or mapping is invalid
        """
        schema = upstream_ref.collect_schema()
        available_cols = set(schema.names())

        # Account for renames - these columns will be available after rename
        renames_spec = renames_spec or {}
        renamed_cols = set(renames_spec.values())

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
            # For expansion relationships (1:N), validate differently
            if lineage_type == "1:N":
                if parent_id_columns is not None:
                    # Explicit parent columns specified - validate they exist
                    missing_cols = set(parent_id_columns) - available_cols
                    if missing_cols:
                        raise ValueError(
                            f"Upstream feature '{upstream_key}' is missing parent ID columns: {sorted(missing_cols)}. "
                            f"The expansion relationship requires parent columns {sorted(parent_id_columns)}, "
                            f"but upstream only has columns: {sorted(available_cols)}."
                        )
                else:
                    # No explicit parent columns - infer from available columns
                    # For expansion relationships, the parent's ID columns often become regular fields
                    # in the child (e.g., video_id is an ID in Video but a regular field in VideoChunk).
                    # So we shouldn't validate at this point - the join will use whatever common
                    # columns exist after the child's load_input() is called.
                    # Just skip validation for inference-based expansion relationships.
                    pass
            else:
                # Default validation for non-expansion relationships
                # No mapping provided - only need unmapped ID columns
                # (columns that aren't mapped by any upstream)
                unmapped_id_cols = set(id_columns) - all_mapped_columns
                # Check both available columns and columns that will be available after rename
                available_after_rename = available_cols | renamed_cols
                # Find common ID columns - if there are none, it will be a cross join (allowed)
                common_id_cols = unmapped_id_cols & available_after_rename
                # Only fail if SOME but not ALL ID columns are present (partial match is an error)
                if common_id_cols and common_id_cols != unmapped_id_cols:
                    missing_cols = unmapped_id_cols - available_after_rename
                    raise ValueError(
                        f"Upstream feature '{upstream_key}' is missing some required ID columns: {sorted(missing_cols)}. "
                        f"The target feature requires ID columns {id_columns} for joining, but upstream "
                        f"only has columns: {sorted(available_cols)}. "
                        f"Either ensure all upstream features have the unmapped ID columns {sorted(unmapped_id_cols)}, "
                        f"or use the 'rename' parameter in FeatureDep to map column names."
                    )
                # If no common ID columns at all, it's a valid cross join - don't error

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
                # Sanitize upstream_key to replace / with _ for SQL compatibility
                sanitized_key = upstream_key.replace("/", "_")
                new_name = f"{UPSTREAM_PROVENANCE_PREFIX}{sanitized_key}{UPSTREAM_PROVENANCE_SUFFIX}"
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
    ) -> tuple[nw.LazyFrame[Any], str | None]:
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
            Tuple of (prepared LazyFrame, upstream provenance mapping)
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

        return prepared, upstream_mapping

    def join_upstream(
        self,
        upstream_refs: dict[str, nw.LazyFrame[Any]],
        feature_spec: "BaseFeatureSpec",
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

        # Initialize parameters if not provided
        upstream_columns = upstream_columns or {}
        upstream_renames = upstream_renames or {}

        # Collect all ID columns
        all_mapped_columns = set()

        # Get lineage type and parent columns for validation
        from metaxy.models.lineage import ExpansionRelationship

        lineage_type = None
        parent_id_columns = None
        if hasattr(feature_spec.lineage, "relationship"):
            rel = feature_spec.lineage.relationship
            if isinstance(rel, ExpansionRelationship):
                lineage_type = "1:N"
                parent_id_columns = rel.on

        # Validate all upstreams have required ID columns
        for upstream_key, upstream_ref in upstream_refs.items():
            # Get renames for this upstream to check if missing columns will be provided via rename
            renames_spec = upstream_renames.get(upstream_key)
            self._validate_id_columns_for_upstream(
                upstream_key,
                upstream_ref,
                id_columns,
                None,
                all_mapped_columns,
                lineage_type,
                parent_id_columns,
                renames_spec,
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

            # Prepare upstream for joining
            prepared, provenance_mapping = self._prepare_upstream_for_join(
                upstream_key=upstream_key,
                upstream_ref=upstream_ref,
                columns_spec=columns_spec,
                renames_spec=renames_spec,
                id_mapping=None,  # No longer using id_mapping - use rename instead
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
                # Dynamically determine join columns based on common columns
                assert joined is not None  # For type checker

                # Get columns from both dataframes
                joined_schema = joined.collect_schema()
                prepared_schema = prepared.collect_schema()
                joined_cols = set(joined_schema.names())
                prepared_cols = set(prepared_schema.names())

                # Find common ID columns (intersection of ID columns in both frames)
                common_id_cols = (joined_cols & prepared_cols) & id_columns_set

                if not common_id_cols:
                    # No common ID columns - this will be a cross join
                    # Create a dummy column for joining (will produce cross product)
                    joined = joined.with_columns(nw.lit(1).alias("__dummy_join_key__"))
                    prepared = prepared.with_columns(
                        nw.lit(1).alias("__dummy_join_key__")
                    )
                    join_cols = ["__dummy_join_key__"]
                    # Join and then drop the dummy column
                    joined = joined.join(
                        prepared,
                        on=join_cols,
                        how="inner",
                    ).drop("__dummy_join_key__")
                else:
                    # Use common ID columns for joining
                    join_cols = sorted(common_id_cols)
                    joined = joined.join(
                        prepared,
                        on=join_cols,
                        how="inner",  # Only rows present in ALL upstreams
                    )

        assert joined is not None  # We know we have at least one upstream

        # EXPLICIT AGGREGATION: Check the lineage type to determine if aggregation is needed
        # - Identity: No aggregation (1:1 relationship)
        # - Aggregation: Group by specified columns (N:1 relationship)
        # - Expansion: No aggregation here (1:N handled by load_input)

        aggregation_columns = feature_spec.lineage.get_aggregation_columns(id_columns)

        if aggregation_columns is not None:
            # Aggregation relationship - perform grouping
            # Get all columns for aggregation
            schema = joined.collect_schema()
            all_columns_set = set(schema.names())

            # Only group by ID columns that actually exist in the joined result
            # (some ID columns may be generated later by the feature's load_input)
            existing_aggregation_columns = [
                col for col in aggregation_columns if col in all_columns_set
            ]

            if existing_aggregation_columns:
                # Check if we're using Polars backend - need to decide upfront
                use_polars = False
                pl = None  # Initialize to avoid unbound variable
                try:
                    underlying_native = joined.to_native()
                    if hasattr(underlying_native, "_ldf"):  # Polars LazyFrame
                        import polars as pl

                        use_polars = True
                except (ImportError, AttributeError):
                    pass

                # Build aggregation expressions for non-grouping columns
                agg_exprs = []
                for col in all_columns_set:
                    if col not in existing_aggregation_columns:
                        # For provenance columns, aggregate by hashing all values together
                        # This correctly handles N->1 relationships where multiple parents
                        # contribute to a single child's provenance
                        if "provenance" in col.lower() and "by_field" in col.lower():
                            # Aggregate struct fields by hashing each field's values together
                            # For N->1: hash all parent provenances per field
                            # For 1->N: all values are identical, so result is same

                            if use_polars and pl is not None:
                                # Use Polars-specific aggregation for proper N->1 support
                                # In aggregation, use pl.col().implode() to collect values
                                # The alias is applied to the entire expression
                                agg_expr = (
                                    pl.col(col).implode().alias(col)
                                )  # implode() collects into list
                                agg_exprs.append(agg_expr)
                            else:
                                # Non-Polars backend - use first()
                                agg_exprs.append(nw.col(col).first().alias(col))
                        else:
                            # For non-provenance columns, take first value
                            # In valid relationships, these should be identical within groups
                            if use_polars and pl is not None:
                                # Use Polars expression
                                agg_exprs.append(pl.col(col).first().alias(col))
                            else:
                                # Use Narwhals expression
                                agg_exprs.append(nw.col(col).first().alias(col))

                # Apply group_by on existing ID columns
                if agg_exprs:  # Only group if there are columns to aggregate
                    if use_polars:
                        # Use native Polars for aggregation with Polars expressions
                        underlying_native = joined.to_native()
                        native_result = underlying_native.group_by(
                            sorted(existing_aggregation_columns)
                        ).agg(agg_exprs)
                        # Sort by ID columns to maintain deterministic ordering
                        native_result = native_result.sort(existing_aggregation_columns)

                        # Post-process: Convert list columns back to single structs
                        for col in native_result.columns:
                            if (
                                "provenance" in col.lower()
                                and "by_field" in col.lower()
                            ):
                                # This column contains lists of structs, convert to single structs
                                def hash_list(lst) -> dict[str, Any]:
                                    """Hash a list of structs into a single struct."""
                                    # When map_elements is called on a list column,
                                    # it might pass a Series containing the list
                                    if hasattr(lst, "to_list"):
                                        # It's a Series, convert to list
                                        actual_list = lst.to_list()
                                    else:
                                        actual_list = lst
                                    return self._hash_struct_list(actual_list)

                                # Apply the function to convert lists to single structs
                                # Don't specify return_dtype, let Polars infer it
                                if pl is not None:
                                    native_result = native_result.with_columns(
                                        pl.col(col).map_elements(hash_list).alias(col)
                                    )

                        joined = nw.from_native(native_result)
                    else:
                        # Use Narwhals API for aggregation
                        joined = joined.group_by(
                            sorted(existing_aggregation_columns)
                        ).agg(agg_exprs)
                        joined = joined.sort(existing_aggregation_columns)

        # Note: We're NOT filtering columns after aggregation anymore.
        # The aggregation has already handled duplicate values by taking .first() or aggregating.
        # Filtering columns would remove user data that tests expect to be present.

        # Ensure joined is never None when returning
        if joined is None:
            # This should never happen given the logic above, but type checker needs this
            raise ValueError("Internal error: joined result is None")

        return joined, upstream_mapping
