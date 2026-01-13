from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey
from metaxy.versioning.renamed_df import RenamedDataFrame


@dataclass(frozen=True)
class IdColumnTracker:
    """Track ID columns through transformation stages: original -> renamed -> selected -> output."""

    original: tuple[str, ...]
    renamed: tuple[str, ...]
    selected: tuple[str, ...]
    output: tuple[str, ...]

    @classmethod
    def from_upstream_spec(
        cls,
        upstream_id_columns: Sequence[str],
        renames: dict[str, str],
        selected_columns: Sequence[str] | None,
        output_columns: Sequence[str] | None = None,
    ) -> "IdColumnTracker":
        """Create from upstream spec and transformation info."""
        original = tuple(upstream_id_columns)
        renamed = tuple(renames.get(col, col) for col in original)

        if selected_columns is None:
            # No column filtering - all renamed ID columns are present
            selected = renamed
        else:
            # Only include ID columns that are in the selection
            selected_set = set(selected_columns)
            selected = tuple(col for col in renamed if col in selected_set)

        # Output defaults to selected, but can be overridden for aggregation
        output = tuple(output_columns) if output_columns is not None else selected

        return cls(
            original=original,
            renamed=renamed,
            selected=selected,
            output=output,
        )

    def with_output(self, output_columns: Sequence[str]) -> "IdColumnTracker":
        """Return a new tracker with updated output columns."""
        return IdColumnTracker(
            original=self.original,
            renamed=self.renamed,
            selected=self.selected,
            output=tuple(output_columns),
        )


class FeatureDepTransformer:
    """Transforms upstream DataFrames based on FeatureDep configuration.

    Applies the transformations defined in a FeatureDep to an upstream DataFrame:
    - Filters: Static and runtime filters to reduce rows
    - Renames: Column renaming including automatic renaming of metaxy system columns
    - Column selection: Limiting columns to those specified plus required columns

    Also tracks ID column mappings through transformations to support lineage
    relationships that change the granularity (aggregation, expansion).
    """

    METAXY_COLUMNS_TO_LOAD = (
        METAXY_PROVENANCE_BY_FIELD,
        METAXY_PROVENANCE,
        METAXY_DATA_VERSION_BY_FIELD,
        METAXY_DATA_VERSION,
    )

    def __init__(self, dep: FeatureDep, plan: FeaturePlan):
        self.plan = plan
        self.dep = dep

    @cached_property
    def upstream_feature_key(self) -> FeatureKey:
        return self.dep.feature

    @cached_property
    def upstream_feature_spec(self) -> FeatureSpec:
        return self.plan.parent_features_by_key[self.dep.feature]

    @cached_property
    def is_optional(self) -> bool:
        """Whether this dependency uses left join (optional) or inner join (required)."""
        return self.dep.optional

    def _get_flattened_column_rename(
        self,
        col: str,
        base_name: str,
        target_struct: str,
    ) -> str | None:
        """Return renamed column if col is a flattened struct column, else None.

        Flattened columns follow the pattern `{base_name}__{field}`. This method
        renames them to `{target_struct}__{field}` to avoid collisions when joining
        multiple upstream features.

        Args:
            col: Column name to check.
            base_name: The base struct name prefix (e.g., metaxy_provenance_by_field).
            target_struct: The target struct name including the upstream feature suffix.

        Returns:
            Renamed column string if col matches the pattern, otherwise None.
        """
        prefix = f"{base_name}__"
        if col.startswith(prefix):
            field_suffix = col[len(prefix) :]
            return f"{target_struct}__{field_suffix}"
        return None

    def _build_flattened_column_renames(self, columns: Sequence[str]) -> dict[str, str]:
        """Build rename mapping for flattened provenance/data_version columns.

        Renames columns like `metaxy_provenance_by_field__field1` to include the
        upstream feature key, avoiding collisions when joining multiple upstream features.
        """
        suffix = self.upstream_feature_key.to_column_suffix()
        prov_target = f"{METAXY_PROVENANCE_BY_FIELD}{suffix}"
        data_target = f"{METAXY_DATA_VERSION_BY_FIELD}{suffix}"

        renames: dict[str, str] = {}
        for col in columns:
            renamed = self._get_flattened_column_rename(col, METAXY_PROVENANCE_BY_FIELD, prov_target)
            if renamed is None:
                renamed = self._get_flattened_column_rename(col, METAXY_DATA_VERSION_BY_FIELD, data_target)
            if renamed is not None:
                renames[col] = renamed
        return renames

    def transform(self, df: FrameT, filters: Sequence[nw.Expr] | None = None) -> RenamedDataFrame[FrameT]:
        """Apply FeatureDep transformations to an upstream DataFrame.

        Transforms the upstream DataFrame by:
        1. Applying column renames (user-specified, metaxy system columns, and flattened columns)
        2. Filtering with combined static and runtime filters
        3. Selecting specified columns plus required ID and metaxy columns

        Args:
            df: Raw upstream DataFrame.
            filters: Optional runtime filters to apply in addition to static filters.

        Returns:
            RenamedDataFrame containing the transformed data and ID column tracker.
        """
        # Combine static and runtime filters
        combined_filters: list[nw.Expr] = []
        if self.dep.filters is not None:
            combined_filters.extend(self.dep.filters)
        if filters:
            combined_filters.extend(filters)

        # Build rename mapping including flattened column renames
        columns = df.collect_schema().names()  # ty: ignore[invalid-argument-type]
        flattened_renames = self._build_flattened_column_renames(columns)
        renames = {**self.renames, **flattened_renames}

        # Apply renames
        renamed_df = df.rename(renames) if renames else df  # ty: ignore[invalid-argument-type]

        # Determine columns to select (include flattened columns if selecting)
        select_cols = [*self.renamed_columns, *flattened_renames.values()] if self.renamed_columns is not None else None

        # Track ID columns through transformation
        id_tracker = IdColumnTracker.from_upstream_spec(
            upstream_id_columns=self.upstream_feature_spec.id_columns,
            renames=renames,
            selected_columns=self.renamed_columns,
        )

        return (
            RenamedDataFrame(
                df=renamed_df,  # ty: ignore[invalid-argument-type]
                id_column_tracker=id_tracker,
            )
            .filter(combined_filters if combined_filters else None)
            .select(select_cols)
        )

    def rename_upstream_metaxy_column(self, column_name: str) -> str:
        """Add upstream feature key suffix to a column name."""
        return f"{column_name}{self.upstream_feature_key.to_column_suffix()}"

    @cached_property
    def renamed_provenance_col(self) -> str:
        return self.rename_upstream_metaxy_column(METAXY_PROVENANCE)

    @cached_property
    def renamed_provenance_by_field_col(self) -> str:
        return self.rename_upstream_metaxy_column(METAXY_PROVENANCE_BY_FIELD)

    @cached_property
    def renamed_data_version_by_field_col(self) -> str:
        return self.rename_upstream_metaxy_column(METAXY_DATA_VERSION_BY_FIELD)

    @cached_property
    def renamed_data_version_col(self) -> str:
        return self.rename_upstream_metaxy_column(METAXY_DATA_VERSION)

    @cached_property
    def renamed_metaxy_cols(self) -> list[str]:
        return list(map(self.rename_upstream_metaxy_column, self.METAXY_COLUMNS_TO_LOAD))

    @cached_property
    def renames(self) -> dict[str, str]:
        """Column rename mapping including user renames and metaxy column renames."""
        return {
            **(self.dep.rename or {}),
            **{col: self.rename_upstream_metaxy_column(col) for col in self.METAXY_COLUMNS_TO_LOAD},
        }

    @cached_property
    def renamed_id_columns(self) -> list[str]:
        """All upstream ID columns after rename (regardless of column selection)."""
        return [self.renames.get(col, col) for col in self.upstream_feature_spec.id_columns]

    @cached_property
    def selected_id_columns(self) -> list[str]:
        """Upstream ID columns that are actually selected after column filtering."""
        return list(self.id_column_tracker.selected)

    @cached_property
    def id_column_tracker(self) -> IdColumnTracker:
        """ID column tracker for this dependency."""
        output_columns = self.plan.get_input_id_columns_for_dep(self.dep)

        return IdColumnTracker.from_upstream_spec(
            upstream_id_columns=self.upstream_feature_spec.id_columns,
            renames=self.renames,
            selected_columns=self.renamed_columns,
            output_columns=output_columns,
        )

    @cached_property
    def _lineage_on_columns(self) -> list[str]:
        """Columns required by lineage relationship (before rename)."""
        from metaxy.versioning.lineage_handler import get_lineage_required_columns

        return list(get_lineage_required_columns(self.dep))

    @cached_property
    def _join_required_id_columns(self) -> list[str]:
        """Upstream ID columns required for joining (before rename)."""
        if self._lineage_on_columns:
            return []
        return list(self.upstream_feature_spec.id_columns)

    def _apply_rename(self, column: str) -> str:
        return self.renames.get(column, column)

    @cached_property
    def _user_requested_columns(self) -> list[str]:
        """User-requested columns (after rename)."""
        if self.dep.columns is None:
            return []
        return [self._apply_rename(col) for col in self.dep.columns]

    @cached_property
    def _lineage_required_columns(self) -> list[str]:
        """Lineage-required columns (after rename) not already in user selection."""
        already_selected = set(self._user_requested_columns)
        return [
            self._apply_rename(col)
            for col in self._lineage_on_columns
            if self._apply_rename(col) not in already_selected
        ]

    @cached_property
    def _join_required_columns(self) -> list[str]:
        """Join-required ID columns (after rename) not already selected."""
        already_selected = set(self._user_requested_columns) | set(self._lineage_required_columns)
        return [
            self._apply_rename(col)
            for col in self._join_required_id_columns
            if self._apply_rename(col) not in already_selected
        ]

    @cached_property
    def renamed_columns(self) -> list[str] | None:
        """Columns to select, or None to select all."""
        if self.dep.columns is None:
            return None

        return [
            *self._user_requested_columns,
            *self._lineage_required_columns,
            *self._join_required_columns,
            *self.renamed_metaxy_cols,
        ]
