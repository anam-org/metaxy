"""Handler for normalizing provenance based on lineage relationships.

This module provides abstractions for handling different lineage relationship types
(identity, aggregation, expansion) when comparing expected vs current provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.models.constants import METAXY_PROVENANCE, METAXY_PROVENANCE_BY_FIELD
from metaxy.utils.hashing import get_hash_truncation_length

if TYPE_CHECKING:
    from metaxy.models.plan import FeaturePlan
    from metaxy.versioning.engine import VersioningEngine
    from metaxy.versioning.types import HashAlgorithm


class LineageHandler(ABC):
    """Base class for handling lineage-based provenance normalization."""

    def __init__(self, feature_plan: FeaturePlan, engine: VersioningEngine):
        """Initialize handler with feature plan and engine.

        Args:
            feature_plan: The feature plan containing lineage information
            engine: The provenance engine instance
        """
        self.plan = feature_plan
        self.feature_spec = feature_plan.feature
        self.engine = engine

    @abstractmethod
    def normalize_for_comparison(
        self,
        expected: FrameT,
        current: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> tuple[FrameT, FrameT, list[str]]:
        """Normalize expected and current DataFrames for provenance comparison.

        Args:
            expected: Expected metadata computed from upstream
            current: Current metadata from store
            hash_algorithm: Hash algorithm to use
            hash_length: Hash truncation length

        Returns:
            Tuple of (normalized_expected, normalized_current, join_columns)
        """
        pass

    @property
    def input_id_columns(self) -> list[str]:
        """Columns that define a logical input unit.

        These are the columns used to count distinct input units for progress calculation.
        Delegates to FeaturePlan.input_id_columns.

        Returns:
            List of column names that define a logical input unit.
        """
        return self.plan.input_id_columns


class IdentityLineageHandler(LineageHandler):
    """Handler for 1:1 identity lineage relationships.

    No normalization needed - each upstream row maps to exactly one downstream row.
    """

    def normalize_for_comparison(
        self,
        expected: FrameT,
        current: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> tuple[FrameT, FrameT, list[str]]:
        """No normalization needed for identity relationships."""
        id_columns = list(self.feature_spec.id_columns)
        return expected, current, id_columns


class AggregationLineageHandler(LineageHandler):
    """Handler for N:1 aggregation lineage relationships.

    Multiple upstream rows aggregate to one downstream row. We need to:
    1. Group expected metadata by aggregation columns (sorted within group)
    2. Concatenate provenance values deterministically
    3. Hash the concatenated result using engine's hash method
    """

    def normalize_for_comparison(
        self,
        expected: FrameT,
        current: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> tuple[FrameT, FrameT, list[str]]:
        """Aggregate expected provenance by grouping."""
        agg_columns = self.plan.input_id_columns

        # Aggregate expected provenance
        expected_agg = self._aggregate_provenance(expected, agg_columns, hash_algorithm)  # ty: ignore[invalid-argument-type]

        return expected_agg, current, agg_columns

    def _aggregate_provenance(
        self,
        expected: FrameT,
        agg_columns: list[str],
        hash_algorithm: HashAlgorithm,
    ) -> FrameT:
        """Aggregate provenance for N:1 relationships.

        Strategy:
        1. Sort by id_columns within each group for deterministic ordering
        2. Group by aggregation columns and concatenate provenance with engine's method
        3. Hash the concatenated result using engine's hash_string_column

        Args:
            expected: Expected metadata with upstream provenance
            agg_columns: Columns to group by
            hash_algorithm: Hash algorithm to use
            hash_length: Length to truncate hash to

        Returns:
            Aggregated DataFrame with one row per group
        """
        # Sort by all id_columns for deterministic ordering within groups
        id_columns = list(self.feature_spec.id_columns)
        expected_sorted = expected.sort(id_columns)  # ty: ignore[invalid-argument-type]

        # Use engine's aggregate_with_string_concat method
        # This concatenates provenance strings and stores in a temporary column
        grouped = self.engine.aggregate_with_string_concat(
            df=expected_sorted,
            group_by_columns=agg_columns,
            concat_column=METAXY_PROVENANCE,
            concat_separator="|",
            exclude_columns=[METAXY_PROVENANCE_BY_FIELD],
        )

        # Hash the concatenated provenance using engine's method
        # Note: the concat column still has name METAXY_PROVENANCE after aggregation
        hashed = self.engine.hash_string_column(
            grouped, METAXY_PROVENANCE, "__hashed_prov", hash_algorithm
        )

        # Replace METAXY_PROVENANCE with truncated hash
        hashed = hashed.drop(METAXY_PROVENANCE).rename(
            {"__hashed_prov": METAXY_PROVENANCE}
        )
        hashed = hashed.with_columns(
            nw.col(METAXY_PROVENANCE).str.slice(0, get_hash_truncation_length())
        )

        # Create placeholder provenance_by_field struct using engine's method
        field_names = [f.key.to_struct_key() for f in self.plan.feature.fields]
        field_map = {name: "__aggregated_placeholder" for name in field_names}

        # Add placeholder column
        hashed = hashed.with_columns(
            nw.lit("aggregated").alias("__aggregated_placeholder")
        )

        # Build struct using engine's method
        result = self.engine.build_struct_column(
            hashed, METAXY_PROVENANCE_BY_FIELD, field_map
        )

        # Drop placeholder
        result = result.drop("__aggregated_placeholder")

        return result  # ty: ignore[invalid-return-type]


class ExpansionLineageHandler(LineageHandler):
    """Handler for 1:N expansion lineage relationships.

    One upstream row expands to many downstream rows. All downstream rows
    with the same parent ID should have the same provenance. We group
    current by parent columns and take any representative row.
    """

    def normalize_for_comparison(
        self,
        expected: FrameT,
        current: FrameT,
        hash_algorithm: HashAlgorithm,
    ) -> tuple[FrameT, FrameT, list[str]]:
        """Group current by parent ID columns."""
        parent_columns = self.plan.input_id_columns

        # Group current by parent columns and take any representative row
        current_grouped = (
            current.with_columns(nw.lit(True).alias("_dummy"))  # ty: ignore[invalid-argument-type]
            .filter(
                nw.col("_dummy")
                .is_first_distinct()
                .over(*parent_columns, order_by="_dummy")
            )
            .drop("_dummy")
        )

        return expected, current_grouped, parent_columns
