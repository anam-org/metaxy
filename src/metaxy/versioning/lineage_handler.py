"""Lineage handlers for different lineage relationship types.

Each dependency can have its own lineage relationship (identity, aggregation, expansion).
This module provides handlers that know how to:
- Determine output ID columns for each lineage type
- Transform upstream data for lineage-specific handling (aggregation only)
- Transform current metadata for comparison (expansion only)

IMPORTANT: Metaxy does NOT transform user data. Users perform actual data
transformations (aggregation, expansion) in their feature computation code.
Metaxy only tracks versioning/provenance metadata.

Lineage types and when they matter:
- Identity (1:1): No special handling. Upstream provenance flows directly.
- Aggregation (N:1): Upstream metaxy columns are aggregated field-by-field using
  window functions. All rows in the same aggregation group get identical values.
  The user performs actual data aggregation; Metaxy just pre-computes the aggregated
  provenance so the user can pick any_value() for metaxy columns.
- Expansion (1:N): Handled during comparison. Current metadata is collapsed to
  parent level since all children inherit the same parent provenance.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.models.lineage import LineageRelationshipType
from metaxy.utils.hashing import get_hash_truncation_length

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureDep
    from metaxy.models.plan import FeaturePlan
    from metaxy.versioning.engine import VersioningEngine
    from metaxy.versioning.feature_dep_transformer import FeatureDepTransformer
    from metaxy.versioning.types import HashAlgorithm


class LineageHandler(ABC):
    """Base class for lineage relationship handlers.

    Each handler knows how to handle a specific lineage relationship type.
    Handlers transform upstream metaxy columns as needed for the lineage type
    and provide utilities for the versioning engine to compute provenance correctly.
    """

    def __init__(
        self,
        feature_dep: FeatureDep,
        plan: FeaturePlan,
        engine: VersioningEngine,
        dep_transformer: FeatureDepTransformer,
    ):
        """Initialize handler for a specific dependency.

        Args:
            feature_dep: The dependency this handler handles
            plan: The feature plan for the downstream feature
            engine: The versioning engine instance
            dep_transformer: The FeatureDepTransformer that handles column naming
        """
        self.dep = feature_dep
        self.plan = plan
        self.engine = engine
        self.dep_transformer = dep_transformer

    def transform_upstream(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> FrameT:
        """Transform upstream data according to lineage relationship.

        Applied per-dependency before joining with other dependencies.
        By default, returns the data unchanged.

        Args:
            df: Upstream DataFrame (already filtered, renamed, selected)
            hash_algorithm: Hash algorithm for any provenance computation

        Returns:
            Transformed DataFrame
        """
        return df

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

    @property
    def requires_aggregation(self) -> bool:
        """Whether this lineage type requires aggregating upstream values."""
        return False


class IdentityLineageHandler(LineageHandler):
    """Handler for 1:1 identity lineage.

    Each upstream row maps to exactly one downstream row.
    No special handling needed.
    """

    pass


class AggregationLineageHandler(LineageHandler):
    """Handler for N:1 aggregation lineage.

    Multiple upstream rows map to one downstream row.
    When computing downstream provenance, upstream values within each
    downstream ID group must be aggregated (concat+hash).

    The user performs actual data aggregation in their feature computation code.
    Metaxy pre-aggregates the provenance/data_version columns field-by-field using
    window functions so all rows in the same group have identical metaxy values.
    The user can then use any_value() or first() when doing their aggregation.
    """

    @property
    def requires_aggregation(self) -> bool:
        """Aggregation lineage requires aggregating upstream values."""
        return True

    def transform_upstream(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> FrameT:
        """Aggregate upstream metaxy columns field-by-field using window functions.

        For each field in metaxy_data_version_by_field and metaxy_provenance_by_field:
        1. Extract the field value from the struct to a temp column
        2. Use engine.concat_strings_over_groups to aggregate within groups
        3. Hash the concatenated values
        4. All rows in the group get the same hashed value

        Then rebuild the struct columns and compute sample-level versions.

        IMPORTANT: This does NOT reduce rows. All original rows are preserved.
        The user performs actual row aggregation in their code.
        """
        agg_columns = self.output_id_columns
        upstream_spec = self.plan.parent_features_by_key[self.dep.feature]
        transformer = self.dep_transformer

        # Get renamed column names
        renamed_data_version_col = transformer.renamed_data_version_col
        renamed_data_version_by_field_col = (
            transformer.renamed_data_version_by_field_col
        )
        renamed_prov_col = transformer.renamed_provenance_col
        renamed_prov_by_field_col = transformer.renamed_provenance_by_field_col
        selected_id_columns = transformer.selected_id_columns

        # Get field names from upstream feature spec
        upstream_field_names = [f.key.to_struct_key() for f in upstream_spec.fields]

        # Step 1: Extract each field value from struct to a temp column
        extracted_cols: dict[str, str] = {}  # field_name -> extracted_col_name
        for field_name in upstream_field_names:
            extracted_col = f"__extract_{field_name}"
            extracted_cols[field_name] = extracted_col

            extract_expr = (
                nw.col(renamed_data_version_by_field_col)
                .struct.field(field_name)
                .cast(nw.String)
            )
            df = df.with_columns(extract_expr.alias(extracted_col))  # ty: ignore[invalid-argument-type]

        # Step 2: Use engine method to aggregate within groups (window function)
        aggregated_cols: dict[str, str] = {}  # field_name -> aggregated_col_name
        for field_name, extracted_col in extracted_cols.items():
            aggregated_col = f"__agg_{field_name}"
            aggregated_cols[field_name] = aggregated_col

            df = self.engine.concat_strings_over_groups(
                df,  # ty: ignore[invalid-argument-type]
                source_column=extracted_col,
                target_column=aggregated_col,
                group_by_columns=agg_columns,
                order_by_columns=selected_id_columns,
                separator="|",
            )

        # Step 3: Hash each aggregated field
        hash_length = get_hash_truncation_length()
        hashed_field_cols: dict[str, str] = {}  # field_name -> hashed_col_name

        for field_name, aggregated_col in aggregated_cols.items():
            hash_col = f"__hash_{field_name}"
            hashed_field_cols[field_name] = hash_col

            df = self.engine.hash_string_column(
                df,  # ty: ignore[invalid-argument-type]
                aggregated_col,
                hash_col,
                hash_algorithm,
            )
            df = df.with_columns(nw.col(hash_col).str.slice(0, hash_length))  # ty: ignore[invalid-argument-type]

        # Drop the original struct columns and temp columns, then rebuild structs
        df = df.drop(  # ty: ignore[invalid-argument-type]
            renamed_data_version_by_field_col,
            renamed_prov_by_field_col,
            renamed_data_version_col,
            renamed_prov_col,
            *extracted_cols.values(),
            *aggregated_cols.values(),
        )

        # Build new struct columns from hashed fields
        df = self.engine.build_struct_column(  # ty: ignore[invalid-assignment]
            df, renamed_data_version_by_field_col, hashed_field_cols
        )
        df = self.engine.build_struct_column(  # ty: ignore[invalid-assignment]
            df,  # ty: ignore[invalid-argument-type]
            renamed_prov_by_field_col,
            hashed_field_cols,
        )

        # Compute sample-level data_version and provenance by hashing all fields together
        # Concatenate all field hashes with separator, then hash
        field_exprs = [
            nw.col(renamed_data_version_by_field_col).struct.field(field_name)
            for field_name in sorted(upstream_field_names)
        ]
        sample_concat = nw.concat_str(field_exprs, separator="|")
        df = df.with_columns(sample_concat.alias("__sample_concat"))  # ty: ignore[invalid-argument-type]

        df = self.engine.hash_string_column(  # ty: ignore[invalid-assignment]
            df,
            "__sample_concat",
            renamed_data_version_col,
            hash_algorithm,  # ty: ignore[invalid-argument-type]
        )
        df = df.with_columns(  # ty: ignore[invalid-argument-type]
            nw.col(renamed_data_version_col).str.slice(0, hash_length),
            nw.col(renamed_data_version_col).alias(renamed_prov_col),
        )

        # Drop temp columns
        df = df.drop("__sample_concat", *hashed_field_cols.values())  # ty: ignore[invalid-argument-type]

        return df  # ty: ignore[invalid-return-type]


class ExpansionLineageHandler(LineageHandler):
    """Handler for 1:N expansion lineage.

    One upstream row expands to many downstream rows.
    All expanded rows inherit their parent's provenance.
    The expansion itself happens in user code; Metaxy just tracks lineage.

    During comparison, current metadata is collapsed to parent level
    since all children from the same parent have the same provenance.
    """

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


def create_lineage_handler(
    feature_dep: FeatureDep,
    plan: FeaturePlan,
    engine: VersioningEngine,
    dep_transformer: FeatureDepTransformer,
) -> LineageHandler:
    """Factory function to create appropriate lineage handler for a dependency.

    Args:
        feature_dep: The dependency to create a handler for
        plan: The feature plan for the downstream feature
        engine: The versioning engine instance
        dep_transformer: The FeatureDepTransformer for column naming

    Returns:
        Appropriate LineageHandler instance based on lineage type
    """
    relationship_type = feature_dep.lineage.relationship.type

    if relationship_type == LineageRelationshipType.IDENTITY:
        return IdentityLineageHandler(feature_dep, plan, engine, dep_transformer)
    elif relationship_type == LineageRelationshipType.AGGREGATION:
        return AggregationLineageHandler(feature_dep, plan, engine, dep_transformer)
    elif relationship_type == LineageRelationshipType.EXPANSION:
        return ExpansionLineageHandler(feature_dep, plan, engine, dep_transformer)
    else:
        raise ValueError(f"Unknown lineage relationship type: {relationship_type}")
