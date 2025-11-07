"""Provenance tracking system for Metaxy.

This package provides a unified interface for tracking field and sample-level provenance
across different backend implementations (Polars, DuckDB, ClickHouse, etc).

The ProvenanceTracker is the core abstraction that:
1. Joins upstream feature metadata
2. Calculates field-level provenance hashes
3. Assembles sample-level provenance
4. Compares with existing metadata to find incremental updates

Backend-specific implementations:
- PolarsProvenanceTracker: Uses polars_hash plugin, may materialize lazy frames
- IbisProvenanceTracker: Base class for SQL backends, stays completely lazy
- DuckDBProvenanceTracker: DuckDB-specific hash functions (xxHash via hashfuncs extension)
- ClickHouseProvenanceTracker: ClickHouse-specific hash functions (native support)
"""

from metaxy.provenance.tracker import (
    ProvenanceTracker,
    RenamedDataFrame,
)
from metaxy.provenance.types import HashAlgorithm, Increment, LazyIncrement

__all__ = [
    "ProvenanceTracker",
    "RenamedDataFrame",
    "HashAlgorithm",
    "Increment",
    "LazyIncrement",
]
