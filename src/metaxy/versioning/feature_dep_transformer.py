from collections.abc import Sequence
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


class FeatureDepTransformer:
    def __init__(self, dep: FeatureDep, plan: FeaturePlan):
        """A class responsible for applying transformations that live on the [metaxy.models.feature_spec.FeatureDep][]:

            - Filters (from FeatureDep.filters)
            - Renames
            - Selections

        This is supposed to always run before the upstream metadata is joined.

        Will also inject Metaxy system columns.
        """
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

    def transform(
        self, df: FrameT, filters: Sequence[nw.Expr] | None = None
    ) -> RenamedDataFrame[FrameT]:
        """Apply the transformation specified by the feature dependency.

        Args:
            df: The dataframe to transform, it's expected to represent the raw upstream feature metadata
                as it resides in the metadata store.
            filters: Optional sequence of additional filters to apply to the dataframe **after renames**.
                These are combined with the static filters from FeatureDep.filters.

        Returns:
            The transformed dataframe coupled with the renamed ID columns

        """
        # Combine static filters from FeatureDep with any additional filters passed as arguments
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
        columns_to_drop = [
            col for col in _COLUMNS_TO_DROP_BEFORE_JOIN if col in existing_columns
        ]
        if columns_to_drop:
            df = df.drop(*columns_to_drop)  # ty: ignore[invalid-argument-type]

        return (
            RenamedDataFrame(
                df=df,  # ty: ignore[invalid-argument-type]
                id_columns=tuple(self.upstream_feature_spec.id_columns),
            )
            .rename(self.renames)
            .filter(combined_filters if combined_filters else None)
            .select(self.renamed_columns)
        )

    def rename_upstream_metaxy_column(self, column_name: str) -> str:
        """Insert the upstream feature key suffix into the column name.

        Is typically applied to Metaxy's system columns since they have to be loaded and do not have user-defined renames."""
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
        return list(
            map(self.rename_upstream_metaxy_column, self.metaxy_columns_to_load)
        )

    @cached_property
    def renames(self) -> dict[str, str]:
        """Get column renames for an upstream feature.

        Returns:
            Dictionary of column renames
        """
        return {
            **(self.dep.rename or {}),
            **{
                col: self.rename_upstream_metaxy_column(col)
                for col in self.metaxy_columns_to_load
            },
        }

    @cached_property
    def renamed_id_columns(self) -> list[str]:
        """All upstream ID columns after rename (regardless of column selection)."""
        return [
            self.renames.get(col, col) for col in self.upstream_feature_spec.id_columns
        ]

    @cached_property
    def selected_id_columns(self) -> list[str]:
        """Upstream ID columns that are actually selected (after column filtering).

        When columns= is specified, only ID columns that are explicitly requested
        or required for lineage/joins are included. When columns= is None, all
        upstream ID columns are selected.
        """
        if self.renamed_columns is None:
            # No column filtering, all ID columns are present
            return self.renamed_id_columns

        # Return only ID columns that are in the selected columns
        selected_set = set(self.renamed_columns)
        return [col for col in self.renamed_id_columns if col in selected_set]

    @cached_property
    def _lineage_on_columns(self) -> list[str]:
        """Get the 'on' columns from lineage relationship (before rename).

        For aggregation and expansion lineage, these columns are required for
        the lineage transformation and must be included in the selected columns.

        Returns:
            List of original column names from lineage.on, or empty list if not applicable.
        """
        from metaxy.models.lineage import AggregationRelationship, ExpansionRelationship

        relationship = self.dep.lineage.relationship
        if isinstance(relationship, (AggregationRelationship, ExpansionRelationship)):
            return list(relationship.on) if relationship.on else []
        return []

    @cached_property
    def _join_required_id_columns(self) -> list[str]:
        """Get upstream ID columns required for joining (before rename).

        For identity lineage: all upstream ID columns are needed for the join.
        For aggregation/expansion: only the `on=` columns are needed (already in _lineage_on_columns).

        Returns:
            List of original column names that must be included for joins to work.
        """
        from metaxy.models.lineage import AggregationRelationship, ExpansionRelationship

        relationship = self.dep.lineage.relationship
        # For aggregation/expansion, the on= columns are the join keys (handled by _lineage_on_columns)
        if isinstance(relationship, (AggregationRelationship, ExpansionRelationship)):
            return []
        # For identity lineage, all upstream ID columns are needed
        return list(self.upstream_feature_spec.id_columns)

    def _apply_rename(self, column: str) -> str:
        """Apply rename mapping to a single column name."""
        return self.renames.get(column, column)

    @cached_property
    def _user_requested_columns(self) -> list[str]:
        """Get user-requested columns (after rename).

        Returns columns explicitly specified in FeatureDep.columns, with renames applied.
        """
        if self.dep.columns is None:
            return []

        return [self._apply_rename(col) for col in self.dep.columns]

    @cached_property
    def _lineage_required_columns(self) -> list[str]:
        """Get lineage-required columns (after rename) not already selected.

        Returns lineage 'on' columns that aren't already in user-requested columns.
        These are auto-included for aggregation/expansion lineage.
        """
        already_selected = set(self._user_requested_columns)
        return [
            self._apply_rename(col)
            for col in self._lineage_on_columns
            if self._apply_rename(col) not in already_selected
        ]

    @cached_property
    def _join_required_columns(self) -> list[str]:
        """Get join-required ID columns (after rename) not already selected.

        For identity lineage, this includes all upstream ID columns.
        For aggregation/expansion, this is empty (handled by _lineage_required_columns).
        """
        already_selected = set(self._user_requested_columns) | set(
            self._lineage_required_columns
        )
        return [
            self._apply_rename(col)
            for col in self._join_required_id_columns
            if self._apply_rename(col) not in already_selected
        ]

    @cached_property
    def renamed_columns(self) -> list[str] | None:
        """Get columns to select from an upstream feature.

        When `columns=` is specified in FeatureDep, this returns only:
        - User-requested columns (explicit selection)
        - Lineage-required columns (for aggregation/expansion `on=` keys)
        - Join-required columns (upstream ID columns for identity lineage)
        - Metaxy system columns

        When `columns=` is None (select all), returns None to indicate no filtering.

        For identity lineage, upstream ID columns are always included (needed for joins).
        For aggregation/expansion lineage, only the `on=` columns are auto-included,
        not other upstream ID columns. This allows column selection to avoid collisions
        when multiple upstreams share column names that aren't part of the join key.

        Returns:
            List of column names to select, or None to select all columns.
        """
        if self.dep.columns is None:
            return None

        return [
            *self._user_requested_columns,
            *self._lineage_required_columns,
            *self._join_required_columns,
            *self.renamed_metaxy_cols,
        ]
