"""Abstract base class for field provenance calculators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import narwhals as nw

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


PROVENANCE_BY_FIELD_COL = METAXY_PROVENANCE_BY_FIELD
DATA_VERSION_BY_FIELD_COL = METAXY_DATA_VERSION_BY_FIELD


class ProvenanceByFieldCalculator(ABC):
    """Calculates metaxy_provenance_by_field hash from joined upstream data.

    The calculator takes joined upstream data (output from UpstreamJoiner)
    and computes the metaxy_provenance_by_field hash for each sample.

    This is Step 2 in the data provenance process:
    1. Join upstream features → unified upstream view
    2. Calculate metaxy_provenance_by_field from upstream → target provenance ← THIS STEP
    3. Diff with current metadata → identify changes

    All calculators work with Narwhals LazyFrames for backend compatibility.

    Examples:
        - PolarsProvenanceByFieldCalculator: Uses polars-hash for in-memory hashing
        - NarwhalsProvenanceByFieldCalculator: Flexibly allows to use almost any dataframe-like backend
        - IbisProvenanceByFieldCalculator: Generates backend-specific SQL for native hashing
        - DuckDBProvenanceByFieldCalculator: Handles DuckDB extensions before delegating to Ibis
    """

    @property
    @abstractmethod
    def supported_algorithms(self) -> list[HashAlgorithm]:
        """List of hash algorithms this calculator supports.

        Returns:
            List of supported HashAlgorithm values

        Example:
            ```py
            calc = PolarsProvenanceByFieldCalculator()
            HashAlgorithm.XXHASH64 in calc.supported_algorithms
            # True
            ```
        """
        pass

    @property
    @abstractmethod
    def default_algorithm(self) -> HashAlgorithm:
        """Default hash algorithm for this calculator.

        Should be the most performant algorithm that's widely compatible.
        Typically xxHash64 for cross-database compatibility.

        Returns:
            Default HashAlgorithm
        """
        pass

    @abstractmethod
    def calculate_provenance_by_field(
        self,
        joined_upstream: nw.LazyFrame[Any],
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
        upstream_column_mapping: dict[str, str],
        hash_algorithm: HashAlgorithm | None = None,
    ) -> nw.LazyFrame[Any]:
        """Calculate metaxy_provenance_by_field column from joined upstream data.

        Computes a Merkle tree hash for each sample by:
        1. For each field in the feature:
           a. Concatenate: field_key | code_version | upstream hashes
           b. Hash the concatenated string
        2. Create struct with all field hashes
        3. Add as metaxy_provenance_by_field column

        Args:
            joined_upstream: Narwhals LazyFrame with all upstream metaxy_provenance_by_field columns joined
                (output from UpstreamJoiner.join_upstream)
            feature_spec: Specification of the feature being computed
            feature_plan: Resolved feature plan with dependencies
            upstream_column_mapping: Maps upstream feature key -> column name
                where its metaxy_provenance_by_field struct is located in joined_upstream
                Example: {"video": "__upstream_video__metaxy_provenance_by_field"}
            hash_algorithm: Hash algorithm to use. If None, uses self.default_algorithm.
                Must be in self.supported_algorithms.

        Returns:
            Narwhals LazyFrame with metaxy_provenance_by_field column added
            Shape: [sample_uid, __upstream_*__metaxy_provenance_by_field columns, metaxy_provenance_by_field (new)]

        Raises:
            ValueError: If hash_algorithm not in supported_algorithms
        """
        pass
