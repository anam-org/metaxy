"""Internal protocols for metadata store components."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import narwhals as nw
import polars as pl
from typing_extensions import Self

from metaxy.models.types import FeatureKey


class MetadataStoreProtocol(Protocol):
    """Protocol defining the interface needed by SystemTableStorage.

    This protocol breaks the circular dependency between base.py and system_tables.py
    by defining only the methods that SystemTableStorage actually uses.
    """

    def _write_metadata_impl(
        self,
        feature_key: FeatureKey,
        df: pl.DataFrame,
    ) -> None:
        """Write metadata for a feature key."""
        ...

    def read_metadata_in_store(
        self,
        feature: FeatureKey,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata from this store only (no fallback)."""
        ...

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        ...
