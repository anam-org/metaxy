"""Abstract base class for compute engines.

A `ComputeEngine` maintains a flat handler registry of
[`IOHandler`][metaxy.metadata_store.io_handler.IOHandler] instances.
User-provided handlers take priority; defaults are lazily appended
via `_create_default_handlers`.

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
from contextlib import ExitStack, contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import Frame
from typing_extensions import Self

from metaxy.metadata_store.io_handler import IOHandler
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.models.types import FeatureKey
from metaxy.versioning import VersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStoreConfig
    from metaxy.models.plan import FeaturePlan


class ComputeEngine(ABC):
    """Abstract compute engine with a flat handler registry.

    Subclasses provide a ``connection``, default handlers, and versioning support.
    """

    # Subclasses must set this to the VersioningEngine class they support.
    versioning_engine_cls: type[VersioningEngine]

    _handlers: list[IOHandler[Any, Any]]
    _defaults_loaded: bool

    def __init__(self, *, handlers: Sequence[IOHandler[Any, Any]] | None = None) -> None:
        self._handlers = list(handlers or [])
        self._defaults_loaded = False
        self._handler_cache: dict[int, IOHandler[Any, Any]] = {}
        self._mode: AccessMode = "r"
        self._exit_stack: ExitStack | None = None

    # --- context manager -----------------------------------------------------

    def __enter__(self) -> Self:
        stack = ExitStack().__enter__()
        self.open(self._mode)
        stack.callback(self.close)
        self._exit_stack = stack
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._exit_stack is not None:
            self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None

    # --- handler registry ----------------------------------------------------

    def _ensure_defaults(self) -> None:
        """Lazily append default handlers (once)."""
        if not self._defaults_loaded:
            self._handlers.extend(self._create_default_handlers())
            self._defaults_loaded = True

    def get_handler(self, storage_config: StorageConfig) -> IOHandler[Any, Any]:
        """Return the first handler that can handle this storage_config."""
        cached = self._handler_cache.get(id(storage_config))
        if cached is not None:
            return cached
        self._ensure_defaults()
        for h in self._handlers:
            if h.can_handle(storage_config):
                self._handler_cache[id(storage_config)] = h
                return h
        from metaxy.metadata_store.exceptions import NoHandlerError

        raise NoHandlerError(f"No handler registered for format '{storage_config.format}'")

    def can_handle(self, storage_config: StorageConfig) -> bool:
        """Whether any registered handler can handle this storage_config."""
        self._ensure_defaults()
        return any(h.can_handle(storage_config) for h in self._handlers)

    @abstractmethod
    def _create_default_handlers(self) -> list[IOHandler[Any, Any]]:
        """Return default handlers for this engine. Called once, lazily."""
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
        with self.get_handler(storage_config).open(self.connection, self._mode) as handler:
            return handler.read(self.connection, storage_config.resolve(key), filters=filters, columns=columns)

    def write(
        self,
        key: FeatureKey,
        df: Frame,
        storage_config: StorageConfig,
        **kwargs: Any,
    ) -> None:
        with self.get_handler(storage_config).open(self.connection, self._mode) as handler:
            handler.write(self.connection, storage_config.resolve(key), df, **kwargs)

    def has_feature(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
    ) -> bool:
        with self.get_handler(storage_config).open(self.connection, self._mode) as handler:
            return handler.has_feature(self.connection, storage_config.resolve(key))

    def drop(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
    ) -> None:
        with self.get_handler(storage_config).open(self.connection, self._mode) as handler:
            handler.drop(self.connection, storage_config.resolve(key))

    def delete(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
    ) -> None:
        with self.get_handler(storage_config).open(self.connection, self._mode) as handler:
            handler.delete(
                self.connection,
                storage_config.resolve(key),
                filters,
                with_feature_history=with_feature_history,
            )

    def get_store_metadata(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
    ) -> dict[str, Any]:
        with self.get_handler(storage_config).open(self.connection, self._mode) as handler:
            return handler.get_store_metadata(storage_config.resolve(key))

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
