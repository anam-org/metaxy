"""Lineage handlers for different lineage relationship types."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from narwhals.typing import FrameT

from metaxy.models.lineage import (
    AggregationRelationship,
    ExpansionRelationship,
    LineageRelationshipType,
)

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureDep
    from metaxy.models.plan import FeaturePlan
    from metaxy.versioning.engine import VersioningEngine
    from metaxy.versioning.feature_dep_transformer import FeatureDepTransformer
    from metaxy.versioning.types import HashAlgorithm


class LineageHandler(ABC):
    """Base class for handling different lineage relationship types.

    Lineage handlers transform upstream data according to the relationship type
    (identity, aggregation, expansion) between the upstream and downstream features.
    Each handler implements transform methods for both upstream preparation and
    current data comparison during increment resolution.
    """

    def __init__(
        self,
        feature_dep: FeatureDep,
        plan: FeaturePlan,
        engine: VersioningEngine,
        dep_transformer: FeatureDepTransformer,
    ):
        self.dep = feature_dep
        self.plan = plan
        self.engine = engine
        self.dep_transformer = dep_transformer

    def transform_upstream(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> FrameT:
        """Transform upstream data according to the lineage relationship.

        Called during upstream preparation to apply lineage-specific transformations
        such as aggregating provenance values for N:1 relationships.

        Args:
            df: Upstream DataFrame after filtering, renaming, and selection.
            hash_algorithm: Hash algorithm for any hashing operations.

        Returns:
            Transformed DataFrame (default implementation returns input unchanged).
        """
        return df

    def transform_current_for_comparison(
        self,
        current: FrameT,
        join_columns: list[str],
    ) -> FrameT:
        """Transform current stored data for comparison with expected provenance.

        Called during increment resolution to prepare current metadata for comparison.
        For example, expansion lineage collapses child rows back to parent level.

        Args:
            current: Current metadata DataFrame from the store.
            join_columns: Columns used for joining expected and current data.

        Returns:
            Transformed DataFrame (default implementation returns input unchanged).
        """
        return current

    @property
    def output_id_columns(self) -> list[str]:
        """ID columns after lineage transformation."""
        return list(self.dep_transformer.id_column_tracker.output)

    @property
    def requires_aggregation(self) -> bool:
        """Whether this lineage type requires aggregating upstream values."""
        return False

    def get_required_columns(self) -> set[str]:
        """Return additional columns required by this lineage relationship.

        These are columns needed for lineage operations (e.g., aggregation 'on'
        columns) that must be included even if not explicitly selected.

        Returns:
            Set of column names (before rename) required by this lineage type.
        """
        return set()


class IdentityLineageHandler(LineageHandler):
    """Handler for 1:1 identity lineage. No special handling needed."""

    pass


class AggregationLineageHandler(LineageHandler):
    """Handler for N:1 aggregation lineage.

    Pre-aggregates provenance columns field-by-field so all rows in the same
    group have identical metaxy values. Does NOT reduce rows.
    """

    @property
    def requires_aggregation(self) -> bool:
        return True

    def get_required_columns(self) -> set[str]:
        """Return aggregation 'on' columns."""
        relationship = self.dep.lineage.relationship
        if isinstance(relationship, AggregationRelationship) and relationship.on:
            return set(relationship.on)
        return set()

    def transform_upstream(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> FrameT:
        """Aggregate upstream metaxy columns field-by-field using window functions."""
        upstream_spec = self.plan.parent_features_by_key[self.dep.feature]
        transformer = self.dep_transformer

        # Get field names from upstream feature spec
        upstream_field_names = [f.key.to_struct_key() for f in upstream_spec.fields]

        return self.engine.aggregate_metadata_columns(
            df,  # ty: ignore[invalid-argument-type]
            group_columns=self.output_id_columns,
            order_by_columns=transformer.selected_id_columns,
            upstream_field_names=upstream_field_names,
            renamed_data_version_col=transformer.renamed_data_version_col,
            renamed_data_version_by_field_col=transformer.renamed_data_version_by_field_col,
            renamed_prov_col=transformer.renamed_provenance_col,
            renamed_prov_by_field_col=transformer.renamed_provenance_by_field_col,
            hash_algorithm=hash_algorithm,
        )


class ExpansionLineageHandler(LineageHandler):
    """Handler for 1:N expansion lineage.

    During comparison, collapses current metadata to parent level since all
    children from the same parent have the same provenance.
    """

    def get_required_columns(self) -> set[str]:
        """Return expansion 'on' columns."""
        relationship = self.dep.lineage.relationship
        if isinstance(relationship, ExpansionRelationship) and relationship.on:
            return set(relationship.on)
        return set()

    def transform_current_for_comparison(
        self,
        current: FrameT,
        join_columns: list[str],
    ) -> FrameT:
        """Collapse expanded rows to parent level for comparison.

        For expansion lineage, all children inherit the same provenance from their parent,
        so we just need to pick any one child per parent. We use the engine's
        keep_latest_by_group method with a timestamp column (or create a row number)
        to deterministically pick one row per parent.
        """
        from metaxy.models.constants import METAXY_CREATED_AT

        # Use the engine's keep_latest_by_group to pick one row per parent group
        # This handles all column types correctly (including JSONB for PostgreSQL)
        return self.engine.keep_latest_by_group(
            df=current,  # type: ignore[invalid-argument-type]
            group_columns=join_columns,
            timestamp_column=METAXY_CREATED_AT,
        )


def create_lineage_handler(
    feature_dep: FeatureDep,
    plan: FeaturePlan,
    engine: VersioningEngine,
    dep_transformer: FeatureDepTransformer,
) -> LineageHandler:
    """Create appropriate lineage handler for a dependency based on its lineage type."""
    relationship_type = feature_dep.lineage.relationship.type

    if relationship_type == LineageRelationshipType.IDENTITY:
        return IdentityLineageHandler(feature_dep, plan, engine, dep_transformer)
    elif relationship_type == LineageRelationshipType.AGGREGATION:
        return AggregationLineageHandler(feature_dep, plan, engine, dep_transformer)
    elif relationship_type == LineageRelationshipType.EXPANSION:
        return ExpansionLineageHandler(feature_dep, plan, engine, dep_transformer)
    else:
        raise ValueError(f"Unknown lineage relationship type: {relationship_type}")


def get_lineage_required_columns(feature_dep: FeatureDep) -> set[str]:
    """Get 'on' columns required by lineage relationship. Empty set for identity lineage."""
    relationship = feature_dep.lineage.relationship
    if isinstance(relationship, (AggregationRelationship, ExpansionRelationship)):
        if relationship.on:
            return set(relationship.on)
    return set()
