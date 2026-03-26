"""DuckDB metadata store — thin wrapper that composes DuckDBEngine + IbisStorageConfig."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from duckdb import DuckDBPyConnection  # noqa: TID252

from metaxy._decorators import public
from metaxy.ext.duckdb.config import (
    DuckDBMetadataStoreConfig,
    ExtensionSpec,
    _normalise_extensions,
)
from metaxy.ext.duckdb.engine import DuckDBEngine
from metaxy.ext.duckdb.handlers.ducklake.attachment import DuckLakeAttachmentManager
from metaxy.ext.duckdb.handlers.ducklake.config import DuckLakeConfig
from metaxy.ext.duckdb.handlers.ducklake.handler import DuckDBDuckLakeHandler
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.ibis_compute_engine import (
    DuckLakeStorageConfig,
    IbisStorageConfig,
    IbisStoreBackcompat,
)


@public
class DuckDBMetadataStore(IbisStoreBackcompat):
    """[DuckDB](https://duckdb.org/) metadata store using [Ibis](https://ibis-project.org/) backend.

    Example: Local File
        ```py
        store = DuckDBMetadataStore("metadata.db")
        ```

    Example: With extensions
        ```py
        store = DuckDBMetadataStore("md:my_database", extensions=["spatial"])
        ```
    """

    def __init__(
        self,
        database: str | Path = "",  # noqa: ARG002
        *,
        config: dict[str, str] | None = None,  # noqa: ARG002
        extensions: Sequence[str | ExtensionSpec] | None = None,  # noqa: ARG002
        fallback_stores: list[MetadataStore] | None = None,  # noqa: ARG002
        ducklake: DuckLakeConfig | None = None,  # noqa: ARG002
        table_prefix: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass  # __new__ already initialized via MetadataStore.__init__

    # -- backcompat properties (deprecated, will be removed in 0.2.0) --

    @property
    def database(self) -> str:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("database")
        assert isinstance(self._engine, DuckDBEngine)
        return self._engine.database

    @property
    def extensions(self) -> list[ExtensionSpec]:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("extensions")
        assert isinstance(self._engine, DuckDBEngine)
        return self._engine.extensions

    def _open(self, mode: Any) -> None:
        """Backcompat: delegates to engine.open(). Patchable for tests."""
        self._engine.open(mode)

    def _close(self) -> None:
        """Backcompat: delegates to engine.close()."""
        self._engine.close()

    def _load_extensions(self) -> None:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("_load_extensions")
        assert isinstance(self._engine, DuckDBEngine)
        self._engine._load_extensions()

    def _duckdb_raw_connection(self) -> DuckDBPyConnection:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("_duckdb_raw_connection")
        assert isinstance(self._engine, DuckDBEngine)
        return self._engine._duckdb_raw_connection()

    def preview_ducklake_sql(self) -> list[str]:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("preview_ducklake_sql")
        assert isinstance(self._engine, DuckDBEngine)
        handler = self._engine.handler if isinstance(self._engine.handler, DuckDBDuckLakeHandler) else None
        assert handler is not None, "DuckLake is not configured"
        return handler.preview_ducklake_sql()

    @property
    def ducklake_attachment(self) -> DuckLakeAttachmentManager:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("ducklake_attachment")
        assert isinstance(self._engine, DuckDBEngine)
        handler = self._engine.handler if isinstance(self._engine.handler, DuckDBDuckLakeHandler) else None
        assert handler is not None, "DuckLake is not configured"
        return handler.attachment_manager

    @property
    def ducklake_attachment_config(self) -> DuckLakeConfig:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("ducklake_attachment_config")
        assert isinstance(self._engine, DuckDBEngine)
        handler = self._engine.handler if isinstance(self._engine.handler, DuckDBDuckLakeHandler) else None
        assert handler is not None, "DuckLake is not configured"
        return handler.ducklake_config

    def __new__(
        cls,
        database: str | Path = "",
        *,
        config: dict[str, str] | None = None,
        extensions: Sequence[str | ExtensionSpec] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        ducklake: DuckLakeConfig | None = None,
        table_prefix: str | None = None,
        **kwargs: Any,
    ) -> MetadataStore:
        auto_create_tables = kwargs.pop("auto_create_tables", None)
        if auto_create_tables is None:
            from metaxy.config import MetaxyConfig

            auto_create_tables = MetaxyConfig.get().auto_create_tables

        all_extensions = list(_normalise_extensions(extensions or []))
        ducklake_handler: DuckDBDuckLakeHandler | None = None

        prefix = table_prefix or ""
        storage: list[IbisStorageConfig]
        if ducklake is not None:
            ducklake_handler = DuckDBDuckLakeHandler(
                ducklake,
                auto_create_tables=auto_create_tables,
                store_name=kwargs.get("name"),
            )
            for ext in ducklake_handler.required_extensions():
                if ext.name not in {e.name for e in all_extensions}:
                    all_extensions.append(ext)
            storage = [DuckLakeStorageConfig(format="ducklake", location=str(database), table_prefix=prefix)]
        else:
            storage = [IbisStorageConfig(format="duckdb", location=str(database), table_prefix=prefix)]

        engine = DuckDBEngine(
            database=database,
            config=config,
            extensions=all_extensions,
            auto_create_tables=auto_create_tables,
            handler=ducklake_handler,
        )

        instance = IbisStoreBackcompat.__new__(cls)
        MetadataStore.__init__(
            instance,
            engine=engine,
            storage=storage,
            fallback_stores=fallback_stores,
            auto_create_tables=auto_create_tables,
            **kwargs,
        )
        return instance

    @classmethod
    def from_config(cls, config: MetadataStoreConfig, **kwargs: Any) -> MetadataStore:
        from metaxy.config import MetaxyConfig
        from metaxy.metadata_store.fallback import FallbackStoreList

        assert isinstance(config, DuckDBMetadataStoreConfig)
        config_dict = config.model_dump(exclude_unset=True, exclude={"ducklake", "fallback_stores"})
        store = cast(MetadataStore, cls(ducklake=config.ducklake, **config_dict, **kwargs))
        fallback_store_names = config.model_dump(exclude_unset=True).get("fallback_stores", [])
        if fallback_store_names:
            store.fallback_stores = FallbackStoreList(
                fallback_store_names,
                config=MetaxyConfig.get(),
                parent_hash_algorithm=store.hash_algorithm,
            )
        return store

    @classmethod
    def config_model(cls) -> type[DuckDBMetadataStoreConfig]:
        return DuckDBMetadataStoreConfig
