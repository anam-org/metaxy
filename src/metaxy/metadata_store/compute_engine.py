"""Abstract base class for compute engines.

A `ComputeEngine` delegates I/O to a single
[`StorageHandler`][metaxy.metadata_store.storage_handler.StorageHandler].

```python
MetadataStore(
    engine=DuckDBEngine(database=":memory:"),
    storage=[IbisStorageConfig(format="duckdb", location=":memory:")],
)
```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import Frame

from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.storage_handler import StorageHandler
from metaxy.metadata_store.types import AccessMode
from metaxy.models.types import FeatureKey
from metaxy.versioning import VersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStoreConfig
    from metaxy.models.plan import FeaturePlan


class ComputeEngine(ABC):
    """Abstract compute engine that delegates I/O to a single StorageHandler.

    Subclasses provide a ``connection``, a ``handler``, and versioning support.
    """

    # Subclasses must set this to the VersioningEngine class they support.
    versioning_engine_cls: type[VersioningEngine]

    @property
    @abstractmethod
    def handler(self) -> StorageHandler[Any]:
        """The storage handler for this engine."""
        ...

    @property
    @abstractmethod
    def connection(self) -> Any:
        """The compute engine's active connection."""
        ...

    # --- lifecycle -----------------------------------------------------------

    @abstractmethod
    def open(self, mode: AccessMode) -> None:
        """Open/connect to the backend."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close/disconnect from the backend."""
        ...

    # --- CRUD (delegated to handler) -----------------------------------------

    def read(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        return self.handler.read(self.connection, storage_config, key, filters=filters, columns=columns)

    def write(
        self,
        key: FeatureKey,
        df: Frame,
        storage_config: StorageConfig,
        **kwargs: Any,
    ) -> None:
        self.handler.write(self.connection, storage_config, key, df, **kwargs)

    def has_feature(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
    ) -> bool:
        return self.handler.has_feature(self.connection, storage_config, key)

    def drop(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
    ) -> None:
        self.handler.drop(self.connection, storage_config, key)

    def delete(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
    ) -> None:
        self.handler.delete(
            self.connection,
            storage_config,
            key,
            filters,
            with_feature_history=with_feature_history,
        )

    def get_store_metadata(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
    ) -> dict[str, Any]:
        return self.handler.get_store_metadata(storage_config, key)

    # --- hashing / versioning ------------------------------------------------

    @abstractmethod
    def get_default_hash_algorithm(self) -> HashAlgorithm:
        """Return the default hash algorithm for this backend."""
        ...

    def validate_hash_algorithm_support(self, algorithm: HashAlgorithm) -> None:
        """Raise if `algorithm` is not supported.

        The default implementation is a no-op (all algorithms assumed supported).
        Override in backends with limited hash support.
        """

    @abstractmethod
    @contextmanager
    def create_versioning_engine(self, plan: FeaturePlan) -> Iterator[VersioningEngine]:
        """Create and yield a backend-specific ``VersioningEngine``."""
        ...

    # --- display / config ----------------------------------------------------

    @abstractmethod
    def display(self) -> str:
        """Human-readable description of the backend (for logs/CLI)."""
        ...

    @classmethod
    @abstractmethod
    def config_model(cls) -> type[MetadataStoreConfig]:
        """Return the Pydantic config class for this adapter."""
        ...


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "PolarsComputeEngine":
        import warnings

        from metaxy.metadata_store.polars.engine import PolarsComputeEngine

        warnings.warn(
            "Importing PolarsComputeEngine from metaxy.metadata_store.compute_engine is deprecated. "
            "Import from metaxy.metadata_store.polars instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return PolarsComputeEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
