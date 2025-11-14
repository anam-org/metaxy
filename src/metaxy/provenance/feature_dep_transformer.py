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
from metaxy.provenance.renamed_df import RenamedDataFrame


class FeatureDepTransformer:
    def __init__(self, dep: FeatureDep, plan: FeaturePlan):
        """A class responsible for applying transformations that live on the [metaxy.models.feature_spec.FeatureDep][]:

            - Renames
            - Selections
            - In the future, filters

        This is supposed to always run before the upstream metadata is joined.

        Will also inject Metaxy system columns.
        """
        self.plan = plan
        self.dep = dep

        # Use data_version columns (user-overrideable) instead of provenance columns
        # when computing downstream provenance
        self.metaxy_columns_to_load = [
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
        self, df: FrameT, filters: Sequence[nw.Expr] | None
    ) -> RenamedDataFrame[FrameT]:
        """Apply the transformation specified by the feature dependency.

        Args:
            df: The dataframe to transform, it's expected to represent the raw upstream feature metadata
                as it resides in the metadata store.
            filters: Optional sequence of filters to apply to the dataframe **after renames**.

        Returns:
            The transformed dataframe coupled with the renamed ID columns

        """
        # Backwards compatibility: add data_version columns if they don't exist
        # (defaulting to provenance columns)
        cols = df.collect_schema().names()
        if (
            METAXY_DATA_VERSION_BY_FIELD not in cols
            and METAXY_PROVENANCE_BY_FIELD in cols
        ):
            df = df.with_columns(
                nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD)
            )
        if METAXY_DATA_VERSION not in cols and METAXY_PROVENANCE in cols:
            df = df.with_columns(nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION))

        # Drop provenance columns if both provenance and data_version exist
        # (keeps only data_version which may be user-overridden)
        cols = df.collect_schema().names()
        cols_to_drop = []
        if METAXY_PROVENANCE in cols and METAXY_DATA_VERSION in cols:
            cols_to_drop.append(METAXY_PROVENANCE)
        if METAXY_PROVENANCE_BY_FIELD in cols and METAXY_DATA_VERSION_BY_FIELD in cols:
            cols_to_drop.append(METAXY_PROVENANCE_BY_FIELD)

        if cols_to_drop:
            df = df.drop(*cols_to_drop)

        return (
            RenamedDataFrame(
                df=df, id_columns=list(self.upstream_feature_spec.id_columns)
            )
            .rename(self.renames)
            .filter(filters)
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
    def renamed_data_version_col(self) -> str:
        return self.rename_upstream_metaxy_column(METAXY_DATA_VERSION)

    @cached_property
    def renamed_data_version_by_field_col(self) -> str:
        return self.rename_upstream_metaxy_column(METAXY_DATA_VERSION_BY_FIELD)

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
