"""Lineage transformation handlers for per-dependency provenance calculation.

Each dependency can have its own lineage relationship (identity, aggregation, expansion).
Lineage transformations are applied independently per-dependency before joining.

This module provides:
- LineageTransformer: Per-dependency transformation applied during upstream loading
- Comparison helpers for handling expansion relationships during diff resolution
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.models.lineage import LineageRelationshipType
from metaxy.utils.hashing import get_hash_truncation_length

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureDep
    from metaxy.models.plan import FeaturePlan
    from metaxy.versioning.engine import VersioningEngine
    from metaxy.versioning.types import HashAlgorithm


class LineageTransformer(ABC):
    """Base class for per-dependency lineage transformations.

    Each transformer handles a specific lineage relationship type and is applied
    independently to its dependency's data before joining with other dependencies.
    """

    def __init__(
        self,
        feature_dep: FeatureDep,
        plan: FeaturePlan,
        engine: VersioningEngine,
    ):
        """Initialize transformer for a specific dependency.

        Args:
            feature_dep: The dependency this transformer handles
            plan: The feature plan for the downstream feature
            engine: The versioning engine instance
        """
        self.dep = feature_dep
        self.plan = plan
        self.engine = engine

    @abstractmethod
    def transform_upstream(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> FrameT:
        """Transform upstream data according to lineage relationship.

        Applied per-dependency before joining with other dependencies.

        Args:
            df: Upstream DataFrame (already filtered, renamed, selected)
            hash_algorithm: Hash algorithm for any provenance computation

        Returns:
            Transformed DataFrame
        """
        pass

    def transform_current_for_comparison(
        self,
        current: FrameT,
        join_columns: list[str],
    ) -> FrameT:
        """Transform current (downstream) data for comparison with expected.

        Override in subclasses that need special handling during diff resolution.
        By default, returns the data unchanged.

        Args:
            current: Current downstream metadata from the store
            join_columns: Columns used for joining expected and current

        Returns:
            Transformed DataFrame suitable for comparison
        """
        return current

    @property
    def output_id_columns(self) -> list[str]:
        """ID columns after lineage transformation.

        For identity and expansion: same as upstream (after rename)
        For aggregation: the aggregation columns
        """
        return self.plan.get_input_id_columns_for_dep(self.dep)


class IdentityLineageTransformer(LineageTransformer):
    """Transformer for 1:1 identity lineage.

    No transformation needed - each upstream row maps to exactly one downstream row.
    """

    def transform_upstream(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> FrameT:
        """Identity transformation - return unchanged."""
        return df


class AggregationLineageTransformer(LineageTransformer):
    """Transformer for N:1 aggregation lineage.

    Multiple upstream rows aggregate to one downstream row.
    We group by aggregation columns and combine provenance values.
    """

    def transform_upstream(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> FrameT:
        """Aggregate upstream data by grouping columns.

        Groups by the aggregation columns specified in the lineage relationship,
        concatenates data_version values deterministically, and hashes the result.

        For aggregation, we only care about upstream `metaxy_data_version` (which is
        what downstream provenance computation reads from). The upstream provenance
        column is not used - data_version is either equal to it or customized by user.
        """
        from metaxy.versioning.feature_dep_transformer import FeatureDepTransformer

        agg_columns = self.output_id_columns
        upstream_spec = self.plan.parent_features_by_key[self.dep.feature]

        transformer = FeatureDepTransformer(dep=self.dep, plan=self.plan)
        renamed_data_version_col = transformer.rename_upstream_metaxy_column(
            "metaxy_data_version"
        )
        renamed_data_version_by_field_col = (
            transformer.renamed_data_version_by_field_col
        )
        # We still need to handle provenance columns for the output
        renamed_prov_col = transformer.renamed_provenance_col
        renamed_prov_by_field_col = transformer.renamed_provenance_by_field_col

        # Sort by ALL upstream ID columns for deterministic ordering within each group
        # Using just agg_columns would only sort by the group key, not within each group
        renamed_id_columns = transformer.renamed_id_columns
        df_sorted = df.sort(renamed_id_columns)  # ty: ignore[invalid-argument-type]

        # Verify data_version column exists
        df_columns = df_sorted.collect_schema().names()  # ty: ignore[invalid-argument-type]
        if renamed_data_version_col not in df_columns:
            raise ValueError(
                f"Column '{renamed_data_version_col}' not found in upstream data. "
                f"Available columns: {df_columns}"
            )

        # Aggregate by concatenating data_version values and hashing
        grouped = self.engine.aggregate_with_string_concat(
            df=df_sorted,
            group_by_columns=agg_columns,
            concat_column=renamed_data_version_col,
            concat_separator="|",
            exclude_columns=[
                renamed_prov_by_field_col,
                renamed_data_version_by_field_col,
                renamed_prov_col,
            ],
        )

        # Hash the concatenated values
        hashed = self.engine.hash_string_column(
            grouped, renamed_data_version_col, "__hashed_agg", hash_algorithm
        )
        hashed = hashed.with_columns(
            nw.col("__hashed_agg").str.slice(0, get_hash_truncation_length())
        )

        # The aggregated hash becomes both provenance and data_version
        hashed = hashed.drop(renamed_data_version_col).with_columns(
            nw.col("__hashed_agg").alias(renamed_prov_col),
            nw.col("__hashed_agg").alias(renamed_data_version_col),
        )
        hashed = hashed.drop("__hashed_agg")

        # Build struct columns with the aggregated hash for each field
        upstream_field_names = [f.key.to_struct_key() for f in upstream_spec.fields]
        field_map = {name: renamed_data_version_col for name in upstream_field_names}

        result = self.engine.build_struct_column(
            hashed, renamed_data_version_by_field_col, field_map
        )
        # Also set provenance_by_field to the same values
        result = self.engine.build_struct_column(
            result, renamed_prov_by_field_col, field_map
        )

        return result  # ty: ignore[invalid-return-type]


class ExpansionLineageTransformer(LineageTransformer):
    """Transformer for 1:N expansion lineage.

    One upstream row expands to many downstream rows.
    No transformation during loading - all expanded rows inherit parent provenance.
    The expansion itself happens in user code; Metaxy just tracks lineage.
    """

    def transform_upstream(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> FrameT:
        """Expansion transformation - return unchanged.

        The expansion happens in user compute code, not here.
        Downstream rows inherit their parent's provenance.
        """
        return df

    def transform_current_for_comparison(
        self,
        current: FrameT,
        join_columns: list[str],
    ) -> FrameT:
        """Collapse expanded rows to parent level for comparison.

        For expansion lineage, current has multiple rows per parent (one per child).
        Since all children from the same parent have the same provenance,
        collapse to one row per parent by picking any representative row.

        Args:
            current: Current downstream metadata with expanded rows
            join_columns: Parent ID columns to group by

        Returns:
            DataFrame with one row per parent
        """
        current_cols = current.collect_schema().names()  # ty: ignore[invalid-argument-type]
        non_key_cols = [c for c in current_cols if c not in join_columns]
        return current.group_by(*join_columns).agg(  # ty: ignore[invalid-argument-type]
            *[nw.col(c).any_value(ignore_nulls=True) for c in non_key_cols]
        )


def create_lineage_transformer(
    feature_dep: FeatureDep,
    plan: FeaturePlan,
    engine: VersioningEngine,
) -> LineageTransformer:
    """Factory function to create appropriate lineage transformer for a dependency.

    Args:
        feature_dep: The dependency to create a transformer for
        plan: The feature plan for the downstream feature
        engine: The versioning engine instance

    Returns:
        Appropriate LineageTransformer instance based on lineage type
    """
    relationship_type = feature_dep.lineage.relationship.type

    if relationship_type == LineageRelationshipType.IDENTITY:
        return IdentityLineageTransformer(feature_dep, plan, engine)
    elif relationship_type == LineageRelationshipType.AGGREGATION:
        return AggregationLineageTransformer(feature_dep, plan, engine)
    elif relationship_type == LineageRelationshipType.EXPANSION:
        return ExpansionLineageTransformer(feature_dep, plan, engine)
    else:
        raise ValueError(f"Unknown lineage relationship type: {relationship_type}")
