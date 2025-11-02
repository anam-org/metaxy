"""Abstract base class for data version calculators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import narwhals as nw

from metaxy.data_versioning.hash_algorithms import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.models.feature_spec import BaseFeatureSpec, IDColumns
    from metaxy.models.plan import FeaturePlan


class DataVersionCalculator(ABC):
    """Calculates data_version hash from joined upstream data.

    The calculator takes joined upstream data (output from UpstreamJoiner)
    and computes the data_version hash for each sample.

    This is Step 2 in the data versioning process:
    1. Join upstream features → unified upstream view
    2. Calculate data_version from upstream → target versions ← THIS STEP
    3. Diff with current metadata → identify changes

    All calculators work with Narwhals LazyFrames for backend compatibility.

    Examples:
        - PolarsDataVersionCalculator: Uses polars-hash for in-memory hashing
        - NarwhalsDataVersionCalculator: Uses native SQL hash functions in the database
    """

    @property
    @abstractmethod
    def supported_algorithms(self) -> list[HashAlgorithm]:
        """List of hash algorithms this calculator supports.

        Returns:
            List of supported HashAlgorithm values

        Example:
            ```py
            calc = PolarsDataVersionCalculator()
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
    def calculate_data_versions(
        self,
        joined_upstream: nw.LazyFrame[Any],
        feature_spec: "BaseFeatureSpec[IDColumns]",
        feature_plan: "FeaturePlan",
        upstream_column_mapping: dict[str, str],
        hash_algorithm: HashAlgorithm | None = None,
    ) -> nw.LazyFrame[Any]:
        """Calculate data_version column from joined upstream data.

        Computes a Merkle tree hash for each sample by:
        1. For each field in the feature:
           a. Concatenate: field_key | code_version | upstream hashes
           b. Hash the concatenated string
        2. Create struct with all field hashes
        3. Add as data_version column

        Args:
            joined_upstream: Narwhals LazyFrame with all upstream data_version columns joined
                (output from UpstreamJoiner.join_upstream)
            feature_spec: Specification of the feature being computed
            feature_plan: Resolved feature plan with dependencies
            upstream_column_mapping: Maps upstream feature key -> column name
                where its data_version struct is located in joined_upstream
                Example: {"video": "__upstream_video__data_version"}
            hash_algorithm: Hash algorithm to use. If None, uses self.default_algorithm.
                Must be in self.supported_algorithms.

        Returns:
            Narwhals LazyFrame with data_version column added
            Shape: [sample_uid, __upstream_*__data_version columns, data_version (new)]

        Raises:
            ValueError: If hash_algorithm not in supported_algorithms
        """
        pass
