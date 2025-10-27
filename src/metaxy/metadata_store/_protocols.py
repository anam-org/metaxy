"""Internal protocols for metadata store components."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import narwhals as nw
import polars as pl

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

    def _read_metadata_native(
        self,
        feature: FeatureKey,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata from this store only (no fallback)."""
        ...
