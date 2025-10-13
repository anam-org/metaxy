"""Hash algorithms supported for data versioning."""

from enum import Enum


class HashAlgorithm(Enum):
    """Supported hash algorithms for data versioning.

    These algorithms are chosen for:
    - Speed (non-cryptographic hashes preferred)
    - Cross-database availability
    - Good collision resistance for data versioning
    """

    XXHASH64 = "xxhash64"  # Fast, available in DuckDB, ClickHouse, Polars
    XXHASH32 = "xxhash32"  # Faster for small data, less collision resistant
    WYHASH = "wyhash"  # Very fast, Polars-specific
    SHA256 = "sha256"  # Cryptographic, slower, universally available
    MD5 = "md5"  # Legacy, widely available, not recommended for new code
