"""Abstract base class for storage handlers.

A `StorageHandler` knows how to perform I/O for a specific combination of
compute engine and storage format. The compute engine's connection is passed
to each method call.

Examples of concrete handlers:

- `IbisSQLHandler` — Ibis engine reading/writing SQL tables
- `DuckDBDuckLakeHandler` — DuckDB engine reading DuckLake tables
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import narwhals as nw
from narwhals.typing import Frame

from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    pass

ConnectionT = TypeVar("ConnectionT")


class StorageHandler(ABC, Generic[ConnectionT]):
    """Handles I/O for a specific storage format via a specific compute engine.

    Type parameter `ConnectionT` is the compute engine's connection type
    (e.g. `SQLBackend` for Ibis engines).
    """

    @abstractmethod
    def can_read(self, storage_config: StorageConfig, key: FeatureKey) -> bool:
        """Whether this handler can read `key` from `storage_config`."""
        ...

    @abstractmethod
    def can_write(self, storage_config: StorageConfig, key: FeatureKey) -> bool:
        """Whether this handler can write `key` to `storage_config`."""
        ...

    @abstractmethod
    def read(
        self,
        conn: ConnectionT,
        storage_config: StorageConfig,
        key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read `key` from `storage_config` using `conn`.

        Returns `None` when the table does not exist.
        """
        ...

    @abstractmethod
    def write(
        self,
        conn: ConnectionT,
        storage_config: StorageConfig,
        key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Write `df` for `key` to `storage_config` using `conn`."""
        ...

    @abstractmethod
    def has_feature(
        self,
        conn: ConnectionT,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> bool:
        """Whether `key` exists in `storage_config`."""
        ...

    @abstractmethod
    def drop(
        self,
        conn: ConnectionT,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> None:
        """Drop the table for `key` from `storage_config`."""
        ...

    @abstractmethod
    def delete(
        self,
        conn: ConnectionT,
        storage_config: StorageConfig,
        key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
    ) -> None:
        """Hard-delete rows matching `filters` for `key` from `storage_config`."""
        ...

    def get_store_metadata(
        self,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> dict[str, Any]:
        """Return backend-specific metadata for `key` in `storage_config`.

        Default returns an empty dict.
        """
        return {}

    def on_connection_opened(self, conn: ConnectionT) -> None:
        """Called after the engine opens its connection.

        Override to perform one-time setup that needs a live connection.
        """
