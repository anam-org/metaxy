"""Abstract base class for compute engines.

A `ComputeEngine` maintains a flat handler registry of
[`IOHandler`][metaxy.metadata_store.io_handler.IOHandler] instances.
User-provided handlers take priority; defaults are lazily appended
via `_create_default_handlers`.

```python
MetadataStore(
    engine=DuckDBEngine(database="metadata.db"),
    fallback_storage=[LanceStorageConfig(format="lance", location="s3://bucket/prod")],
)
```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from contextlib import ExitStack, contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

import narwhals as nw
from narwhals.typing import Frame

from metaxy.metadata_store.io_handler import IOHandler
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.models.types import FeatureKey
from metaxy.versioning import VersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStoreConfig
    from metaxy.models.plan import FeaturePlan

HandlerT = TypeVar("HandlerT", bound=IOHandler[Any, Any])
ConnT = TypeVar("ConnT")


class ComputeEngine(ABC, Generic[ConnT]):
    """Abstract compute engine with a flat handler registry.

    Subclasses provide a ``connection``, default handlers, and versioning support.
    Only handlers that accept the engine's connection type `ConnT` can be registered.
    """

    # Subclasses must set this to the VersioningEngine class they support.
    versioning_engine_cls: type[VersioningEngine]

    _handlers: list[IOHandler[ConnT, Any]]
    _defaults_loaded: bool

    def __init__(self, *, handlers: Sequence[IOHandler[ConnT, Any]] | None = None) -> None:
        self._handlers = list(handlers or [])
        self._defaults_loaded = False
        self._handler_cache: dict[int, IOHandler[ConnT, Any]] = {}
        self._mode: AccessMode = "r"
        self._exit_stack: ExitStack | None = None
        self._opened_handlers: list[IOHandler[ConnT, Any]] = []

    # --- context manager -----------------------------------------------------

    def __enter__(self) -> Self:
        with ExitStack() as stack:
            self.open(self._mode)
            stack.callback(self.close)
            self._exit_stack = stack
            stack.callback(self._opened_handlers.clear)
            for handler in dict.fromkeys(self._handler_cache.values()):
                self._open_handler(handler)
            self._exit_stack = stack.pop_all()
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

    def _open_handler(self, handler: IOHandler[ConnT, Any]) -> None:
        """Run the engine-scoped hooks of `handler` while the engine is open."""
        assert self._exit_stack is not None
        handler.on_open(self.connection, self._mode)
        self._opened_handlers.append(handler)
        self._exit_stack.callback(handler.on_close, self.connection)

    # --- handler registry ----------------------------------------------------

    def _ensure_defaults(self) -> None:
        """Lazily append default handlers (once)."""
        if not self._defaults_loaded:
            self._handlers.extend(self._create_default_handlers())
            self._defaults_loaded = True

    def get_handler(self, storage_config: StorageConfig) -> IOHandler[ConnT, Any]:
        """Return the first handler that can handle this storage_config."""
        cached = self._handler_cache.get(id(storage_config))
        if cached is not None:
            return cached
        self._ensure_defaults()
        for h in self._handlers:
            if h.can_handle(storage_config):
                self._handler_cache[id(storage_config)] = h
                if self._exit_stack is not None and h not in self._opened_handlers:
                    self._open_handler(h)
                return h
        from metaxy.metadata_store.exceptions import NoHandlerError

        raise NoHandlerError(f"No handler registered for format '{storage_config.format}'")

    def can_handle(self, storage_config: StorageConfig) -> bool:
        """Whether any registered handler can handle this storage_config."""
        self._ensure_defaults()
        return any(h.can_handle(storage_config) for h in self._handlers)

    def find_handler(self, handler_cls: type[HandlerT]) -> HandlerT | None:
        """Return the first registered handler that is an instance of ``handler_cls``."""
        self._ensure_defaults()
        return next((h for h in self._handlers if isinstance(h, handler_cls)), None)

    @abstractmethod
    def _create_default_handlers(self) -> list[IOHandler[ConnT, Any]]:
        """Return default handlers for this engine. Called once, lazily."""
        ...

    def native_storage(self) -> StorageConfig | None:
        """Return the storage the engine owns natively, or `None` if it has none.

        A store without an explicit primary storage reads and writes here.
        """
        return None

    @property
    @abstractmethod
    def connection(self) -> ConnT:
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
        return self.get_handler(storage_config).read(
            self.connection, storage_config.resolve(key), filters=filters, columns=columns
        )

    def write(
        self,
        key: FeatureKey,
        df: Frame,
        storage_config: StorageConfig,
        **kwargs: Any,
    ) -> None:
        with self.get_handler(storage_config).write_scope(self.connection) as handler:
            handler.write(self.connection, storage_config.resolve(key), df, **kwargs)

    def has_feature(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
    ) -> bool:
        return self.get_handler(storage_config).has_feature(self.connection, storage_config.resolve(key))

    def drop(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
    ) -> None:
        self.get_handler(storage_config).drop(self.connection, storage_config.resolve(key))

    def delete(
        self,
        key: FeatureKey,
        storage_config: StorageConfig,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
    ) -> None:
        self.get_handler(storage_config).delete(
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
        return self.get_handler(storage_config).get_store_metadata(storage_config.resolve(key))

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
    def create_versioning_engine(self, plan: FeaturePlan) -> Generator[VersioningEngine]:
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
