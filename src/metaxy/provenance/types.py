"""Hash algorithms supported for field provenance calculation."""

from enum import Enum
from typing import Any, NamedTuple

import narwhals as nw


class HashAlgorithm(Enum):
    """Supported hash algorithms for field provenance calculation.

    These algorithms are chosen for:
    - Speed (non-cryptographic hashes preferred)
    - Cross-database availability
    - Good collision resistance for field provenance calculation
    """

    XXHASH64 = "xxhash64"  # Fast, available in DuckDB, ClickHouse, Polars
    XXHASH32 = "xxhash32"  # Faster for small data, less collision resistant
    WYHASH = "wyhash"  # Very fast, Polars-specific
    SHA256 = "sha256"  # Cryptographic, slower, universally available
    MD5 = "md5"  # Legacy, widely available, not recommended for new code
    FARMHASH = "farmhash"  # Better than MD5, available in BigQuery


class Increment(NamedTuple):
    """Result of an incremental update containing eager dataframes.

    Contains three sets of samples:
    - added: New samples from upstream not present in current metadata
    - changed: Samples with different provenance
    - removed: Samples in current metadata but not in upstream state
    """

    added: nw.DataFrame[Any]
    changed: nw.DataFrame[Any]
    removed: nw.DataFrame[Any]

    def collect(self) -> "Increment":
        """No-op for eager Increment (already collected)."""
        return self


class LazyIncrement(NamedTuple):
    """Result of an incremental update containing lazy dataframes.

    Contains three sets of samples:
    - added: New samples from upstream not present in current metadata
    - changed: Samples with different provenance
    - removed: Samples in current metadata but not in upstream state
    """

    added: nw.LazyFrame[Any]
    changed: nw.LazyFrame[Any]
    removed: nw.LazyFrame[Any]

    def collect(self) -> Increment:
        """Collect all lazy frames to eager DataFrames."""
        return Increment(
            added=self.added.collect(),
            changed=self.changed.collect(),
            removed=self.removed.collect(),
        )
