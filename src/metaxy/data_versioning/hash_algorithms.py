"""Hash algorithms supported for field provenance calculation."""

from enum import Enum


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
