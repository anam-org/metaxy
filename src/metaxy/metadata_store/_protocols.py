"""Internal protocols for metadata store components."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Protocol

import narwhals as nw
from narwhals.typing import Frame
from typing_extensions import Self

from metaxy.metadata_store.types import AccessMode
from metaxy.models.types import FeatureKey


class MetadataStoreProtocol(Protocol):
    """Protocol defining the interface needed by SystemTableStorage.

    This protocol breaks the circular dependency between base.py and system_tables.py
    by defining only the methods that SystemTableStorage actually uses.
    """

    def write_metadata(
        self,
        feature: FeatureKey | type[Any],
        df: Any,
    ) -> None:
        """Write metadata for a feature."""
        ...

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
    ) -> None:
        """Write metadata for a feature key."""
        ...

    def read_metadata_in_store(
        self,
        feature: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata from this store only (no fallback)."""
        ...

    @contextmanager
    def open(self, mode: AccessMode = AccessMode.READ) -> Iterator[Self]:
        """Open store connection with specified access mode."""
        ...

    def __enter__(self) -> Self:
        """Enter the context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the context manager."""
        ...
