from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.models.constants import (
    _COLUMNS_TO_DROP_BEFORE_JOIN,
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

    def __init__(self, dep: FeatureDep, plan: FeaturePlan):
        self.plan = plan
        self.dep = dep

        self.metaxy_columns_to_load = [
            METAXY_PROVENANCE_BY_FIELD,
            METAXY_PROVENANCE,
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_DATA_VERSION,
        ]

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

    def transform(self, df: FrameT, filters: Sequence[nw.Expr] | None = None) -> RenamedDataFrame[FrameT]:
        """Apply FeatureDep transformations to an upstream DataFrame.

        Transforms the upstream DataFrame by:
        1. Dropping columns that shouldn't be carried through joins
        2. Applying column renames (user-specified and metaxy system columns)
        3. Filtering with combined static and runtime filters
        4. Selecting specified columns plus required ID and metaxy columns

        Args:
            df: Raw upstream DataFrame.
            filters: Optional runtime filters to apply in addition to static filters.

        Returns:
            RenamedDataFrame containing the transformed data and ID column tracker.
        """
        combined_filters: list[nw.Expr] = []
        if self.dep.filters is not None:
            combined_filters.extend(self.dep.filters)
        if filters:
            combined_filters.extend(filters)

        # Drop columns that should not be carried through joins
        # (e.g., metaxy_created_at, metaxy_materialization_id, metaxy_feature_version)
        # These are recalculated for the downstream feature and would cause column name
        # conflicts when joining 3+ upstream features
        existing_columns = set(df.collect_schema().names())  # ty: ignore[invalid-argument-type]
        columns_to_drop = [col for col in _COLUMNS_TO_DROP_BEFORE_JOIN if col in existing_columns]
        if columns_to_drop:
            df = df.drop(*columns_to_drop)  # ty: ignore[invalid-argument-type]

        # Apply rename
        renamed_df = df.rename(self.renames) if self.renames else df  # ty: ignore[invalid-argument-type]

        # Apply filter
        if combined_filters:
            renamed_df = renamed_df.filter(*combined_filters)  # ty: ignore[invalid-argument-type]

        # Apply select
        if self.renamed_columns:
            renamed_df = renamed_df.select(*self.renamed_columns)  # ty: ignore[invalid-argument-type]

        # Create RenamedDataFrame with the complete ID column tracker
        return RenamedDataFrame(
            df=renamed_df,  # ty: ignore[invalid-argument-type]
            id_column_tracker=self.id_column_tracker,
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
        return list(map(self.rename_upstream_metaxy_column, self.metaxy_columns_to_load))

    @cached_property
    def renames(self) -> dict[str, str]:
        """Column rename mapping including user renames and metaxy column renames."""
        return {
            **(self.dep.rename or {}),
            **{col: self.rename_upstream_metaxy_column(col) for col in self.metaxy_columns_to_load},
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
