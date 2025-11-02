"""Metadata diff resolvers for identifying changed field provenance entries."""

from metaxy.data_versioning.diff.base import (
    DiffResult,
    LazyDiffResult,
    MetadataDiffResolver,
)
from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver

__all__ = [
    "DiffResult",
    "LazyDiffResult",
    "MetadataDiffResolver",
    "NarwhalsDiffResolver",
]
