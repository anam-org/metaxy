"""Provenance tracking system for Metaxy.

This package provides a unified interface for tracking field and sample-level provenance
across different backend implementations (Polars, DuckDB, ClickHouse, etc).

The VersioningEngine is the core abstraction that:
1. Joins upstream feature metadata
2. Calculates field-level provenance hashes
3. Assembles sample-level provenance
4. Compares with existing metadata to find incremental updates

Backend-specific implementations:
- PolarsVersioningEngine: Uses polars_hash plugin, may materialize lazy frames
- IbisVersioningEngine: Base class for SQL backends, stays completely lazy
- DuckDBVersioningEngine: DuckDB-specific hash functions (xxHash via hashfuncs extension)
- ClickHouseVersioningEngine: ClickHouse-specific hash functions (native support)
"""

from metaxy.versioning.engine import (
    RenamedDataFrame,
    VersioningEngine,
)
from metaxy.versioning.feature_dep_transformer import IdColumnTracker
from metaxy.versioning.types import (
    HashAlgorithm,
    Increment,
    LazyIncrement,
    PolarsIncrement,
    PolarsLazyIncrement,
)

__all__ = [
    "VersioningEngine",
    "RenamedDataFrame",
    "IdColumnTracker",
    "HashAlgorithm",
    "Increment",
    "LazyIncrement",
    "PolarsIncrement",
    "PolarsLazyIncrement",
]
