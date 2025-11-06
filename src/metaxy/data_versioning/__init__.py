"""Data provenance module for sample-level field provenance calculation.

NOTE: This module provides the old three-component system (joiners/calculators/diff) for
backward compatibility with existing metadata stores. New code should use metaxy.provenance instead.
"""

from metaxy.data_versioning.calculators import (
    PolarsProvenanceByFieldCalculator,
    ProvenanceByFieldCalculator,
)
from metaxy.data_versioning.diff import (
    Increment,
    LazyIncrement,
    MetadataDiffResolver,
    NarwhalsDiffResolver,
)
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.data_versioning.joiners import NarwhalsJoiner, UpstreamJoiner

__all__ = [
    "HashAlgorithm",
    "UpstreamJoiner",
    "NarwhalsJoiner",
    "ProvenanceByFieldCalculator",
    "PolarsProvenanceByFieldCalculator",
    "Increment",
    "LazyIncrement",
    "MetadataDiffResolver",
    "NarwhalsDiffResolver",
]
