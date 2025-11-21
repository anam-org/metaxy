from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dagster as dg

from metaxy.config import MetaxyConfig
from metaxy.metadata_store.base import MetadataStore


class MetaxyMetadataStoreResource(dg.ConfigurableResource):  # pyright: ignore[reportMissingTypeArgument]
    """Dagster resource that provides a Metaxy ``MetadataStore`` from configuration.

    This resource acts as a wrapper around MetadataStore and is injected directly
    into assets (not unwrapped). Assets receive the resource wrapper and can call
    methods on it or use it as a context manager.
    """

    store_name: str | None = None
    fallback_stores: list[str] | None = None
    config_file: str | None = None
    search_parents: bool = True
    auto_discovery_start: str | None = None

    def create_resource(self, context: dg.InitResourceContext) -> MetadataStore:  # noqa: ARG002
        """Dagster hook to provide the concrete MetadataStore to assets."""
        return self.get_store()

    def __enter__(self) -> MetaxyMetadataStoreResource:
        """Context manager entry - opens the underlying store and returns self.

        This allows the resource to be used as:
        ```python
        with store:
            store.resolve_update(...)
            store.write_metadata(...)
        ```
        """
        if not hasattr(self, "_active_store") or self._active_store is None:
            self._active_store = self.get_store()
        self._active_store.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: ANN401
        """Context manager exit - closes the underlying store."""
        if hasattr(self, "_active_store") and self._active_store is not None:
            self._active_store.__exit__(*args)
            self._active_store = None  # type: ignore[assignment]

    def resolve_update(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Proxy to underlying store's resolve_update method."""
        if not hasattr(self, "_active_store") or self._active_store is None:
            raise RuntimeError(
                "MetadataStore is not open. Use MetaxyMetadataStoreResource as a context manager: "
                "with store: ..."
            )
        return self._active_store.resolve_update(*args, **kwargs)  # type: ignore[no-any-return]

    def write_metadata(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Proxy to underlying store's write_metadata method."""
        if not hasattr(self, "_active_store") or self._active_store is None:
            raise RuntimeError(
                "MetadataStore is not open. Use MetaxyMetadataStoreResource as a context manager: "
                "with store: ..."
            )
        self._active_store.write_metadata(*args, **kwargs)

    def read_metadata(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Proxy to underlying store's read_metadata method."""
        if not hasattr(self, "_active_store") or self._active_store is None:
            raise RuntimeError(
                "MetadataStore is not open. Use MetaxyMetadataStoreResource as a context manager: "
                "with store: ..."
            )
        return self._active_store.read_metadata(*args, **kwargs)  # type: ignore[no-any-return]

    def get_store(self) -> MetadataStore:
        """Get the MetadataStore instance without requiring a Dagster context.

        This is useful for IOManagers that need to access the store without
        having access to an InitResourceContext.

        Returns:
            MetadataStore instance configured with fallback stores
        """
        config = self._load_config()
        store = config.get_store(self.store_name)

        if self.fallback_stores is not None:
            store.fallback_stores = _resolve_fallback_stores(
                config=config,
                fallback_names=self.fallback_stores,
                primary_store=store,
                primary_store_name=self.store_name or config.store,
            )

        return store

    @classmethod
    def from_config(
        cls,
        store_name: str | None = None,
        *,
        fallback_stores: list[str] | None = None,
        config_file: str | None = None,
        search_parents: bool = True,
        auto_discovery_start: str | None = None,
    ) -> MetaxyMetadataStoreResource:
        """Build the resource using Metaxy's configuration."""
        return cls(
            store_name=store_name,
            fallback_stores=list(fallback_stores) if fallback_stores else None,
            config_file=config_file,
            search_parents=search_parents,
            auto_discovery_start=auto_discovery_start,
        )

    def _load_config(self) -> MetaxyConfig:
        """Load the Metaxy configuration respecting explicit overrides."""
        config_file = Path(self.config_file) if self.config_file is not None else None
        auto_discovery_start = (
            Path(self.auto_discovery_start)
            if self.auto_discovery_start is not None
            else None
        )

        if config_file is not None:
            return MetaxyConfig.load(
                config_file=config_file,
                search_parents=self.search_parents,
                auto_discovery_start=auto_discovery_start,
            )

        if MetaxyConfig.is_set():
            return MetaxyConfig.get()

        return MetaxyConfig.load(
            search_parents=self.search_parents,
            auto_discovery_start=auto_discovery_start,
        )


def _resolve_fallback_stores(
    *,
    config: MetaxyConfig,
    fallback_names: Sequence[str],
    primary_store: MetadataStore,
    primary_store_name: str,
) -> list[MetadataStore]:
    """Instantiate fallback stores from names and validate compatibility."""
    seen: set[str] = set()
    fallbacks: list[MetadataStore] = []

    for name in fallback_names:
        if name in seen:
            raise ValueError(f"Duplicate fallback store '{name}' in configuration.")
        if name == primary_store_name:
            raise ValueError("A store cannot list itself as a fallback store.")

        fallback_store = config.get_store(name)
        _validate_fallback_store(primary_store, fallback_store, name)

        fallbacks.append(fallback_store)
        seen.add(name)

    return fallbacks


def _validate_fallback_store(
    primary_store: MetadataStore, fallback_store: MetadataStore, fallback_name: str
) -> None:
    """Ensure fallback stores share hashing settings with the primary store."""
    if primary_store.hash_algorithm != fallback_store.hash_algorithm:
        raise ValueError(
            f"Fallback store '{fallback_name}' uses a different hash algorithm "
            f"({fallback_store.hash_algorithm}) than primary store "
            f"({primary_store.hash_algorithm})."
        )

    if primary_store.hash_truncation_length != fallback_store.hash_truncation_length:
        raise ValueError(
            f"Fallback store '{fallback_name}' uses hash truncation length "
            f"{fallback_store.hash_truncation_length}, expected "
            f"{primary_store.hash_truncation_length}."
        )
