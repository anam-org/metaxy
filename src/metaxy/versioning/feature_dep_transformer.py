from collections.abc import Sequence
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

        return (
            RenamedDataFrame(
                df=df,  # ty: ignore[invalid-argument-type]
                id_columns=list(self.upstream_feature_spec.id_columns),
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
        # TODO: potentially include more system columns here?
        return {
            **(self.dep.rename or {}),
            **{
                col: self.rename_upstream_metaxy_column(col)
                for col in self.metaxy_columns_to_load
            },
        }

    @cached_property
    def renamed_id_columns(self) -> list[str]:
        return [
            self.renames.get(col, col) for col in self.upstream_feature_spec.id_columns
        ]

    @cached_property
    def renamed_columns(
        self,
    ) -> list[str] | None:
        """Get columns to select from an upstream feature.

        There include both original and metaxy-injected columns, all already renamed.
        Users are expected to use renamed column names in their columns specification.

        Returns:
            List of column names to select, or None to select all columns
        """

        # If no specific columns requested (None), return None to keep all columns
        # If empty tuple, return only ID columns and system columns
        if self.dep.columns is None:
            return None
        else:
            # Apply renames to the selected columns since selection happens after renaming
            renamed_selected_cols = [
                self.renames.get(col, col) for col in self.dep.columns
            ]
            return [
                *self.renamed_id_columns,
                *renamed_selected_cols,
                *self.renamed_metaxy_cols,
            ]
