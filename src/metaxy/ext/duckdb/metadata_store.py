"""DuckDB metadata store -- thin wrapper that composes DuckDBEngine + IbisStorageConfig."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from pydantic import Field

from metaxy._decorators import public
from metaxy.ext.duckdb.engine import ExtensionSpec
from metaxy.ext.duckdb.handlers.ducklake import (
    DuckDBDuckLakeHandler,
    DuckDBPyConnection,
    DuckLakeAttachmentManager,
    DuckLakeConfig,
)
from metaxy.ext.ibis.engine import (
    DuckLakeStorageConfig,
    IbisStorageConfig,
    IbisStoreBackcompat,
)
from metaxy.ext.ibis.metadata_store import IbisMetadataStoreConfig
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig


@public
class DuckDBMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for DuckDBMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.duckdb.DuckDBMetadataStore"

        [stores.dev.config]
        database = "metadata.db"
        hash_algorithm = "xxhash64"
        ```
    """

    database: str | Path = Field(
        description="Database path (:memory:, file path, or md:database).",
    )
    config: dict[str, str] | None = Field(
        default=None,
        description="DuckDB configuration settings (e.g., {'threads': '4'}).",
    )
    extensions: Sequence[str | ExtensionSpec] | None = Field(
        default=None,
        description="DuckDB extensions to install and load on open. If only a string is provided, the `core` repository is assumed.",
    )
    ducklake: DuckLakeConfig | None = Field(
        default=None,
        description="DuckLake attachment configuration. Learn more [here](/integrations/metadata-stores/storage/ducklake.md).",
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

    # -- backcompat properties (deprecated, will be removed in 0.2.0) --

    @property
    def database(self) -> str:
        from metaxy.ext.duckdb.engine import DuckDBEngine
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("database")
        assert isinstance(self._engine, DuckDBEngine)
        return self._engine.database

    @property
    def extensions(self) -> list[ExtensionSpec]:
        from metaxy.ext.duckdb.engine import DuckDBEngine
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("extensions")
        assert isinstance(self._engine, DuckDBEngine)
        ducklake_handler = self._backcompat_find_ducklake_handler()
        return [
            *self._engine.extensions,
            *(ducklake_handler.required_extensions() if ducklake_handler is not None else []),
        ]

    def _load_extensions(self) -> None:
        from metaxy.ext.duckdb.engine import DuckDBEngine
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("_load_extensions")
        assert isinstance(self._engine, DuckDBEngine)
        self._engine._load_extensions()

    def _duckdb_raw_connection(self) -> DuckDBPyConnection:
        from metaxy.ext.duckdb.engine import DuckDBEngine
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("_duckdb_raw_connection")
        assert isinstance(self._engine, DuckDBEngine)
        return self._engine._duckdb_raw_connection()

    def _backcompat_find_ducklake_handler(self) -> DuckDBDuckLakeHandler | None:
        return self._engine.find_handler(DuckDBDuckLakeHandler)

    def preview_ducklake_sql(self) -> list[str]:
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("preview_ducklake_sql")
        handler = self._backcompat_find_ducklake_handler()
        assert handler is not None, "DuckLake is not configured"
        return handler.preview_ducklake_sql()

    @property
    def ducklake_attachment(self) -> DuckLakeAttachmentManager:
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("ducklake_attachment")
        handler = self._backcompat_find_ducklake_handler()
        assert handler is not None, "DuckLake is not configured"
        return handler.attachment_manager

    @property
    def ducklake_attachment_config(self) -> DuckLakeConfig:
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("ducklake_attachment_config")
        handler = self._backcompat_find_ducklake_handler()
        assert handler is not None, "DuckLake is not configured"
        return handler.ducklake_config

    def __init__(
        self,
        database: str | Path = "",
        *,
        config: dict[str, str] | None = None,
        extensions: Sequence[str | ExtensionSpec] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        ducklake: DuckLakeConfig | None = None,
        table_prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        from metaxy.ext.duckdb.engine import DuckDBEngine

        auto_create_tables = kwargs.pop("auto_create_tables", None)
        if auto_create_tables is None:
            from metaxy.config import MetaxyConfig

            auto_create_tables = MetaxyConfig.get().auto_create_tables

        ducklake_handler: DuckDBDuckLakeHandler | None = None

        prefix = table_prefix or ""
        if ducklake is not None:
            ducklake_handler = DuckDBDuckLakeHandler(
                ducklake,
                auto_create_tables=auto_create_tables,
                store_name=kwargs.get("name"),
            )
            storage = DuckLakeStorageConfig(location=str(database), table_prefix=prefix)
        else:
            storage = IbisStorageConfig(location=str(database), table_prefix=prefix)

        engine = DuckDBEngine(
            database=database,
            config=config,
            extensions=extensions,
            auto_create_tables=auto_create_tables,
            handlers=[ducklake_handler] if ducklake_handler else None,
        )

        super().__init__(
            engine=engine,
            storage=storage,
            fallback_stores=fallback_stores,
            auto_create_tables=auto_create_tables,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config: MetadataStoreConfig, **kwargs: Any) -> Self:
        from metaxy.config import MetaxyConfig
        from metaxy.metadata_store.fallback import FallbackStoreList

        assert isinstance(config, DuckDBMetadataStoreConfig)
        config_dict = config.model_dump(exclude_unset=True, exclude={"ducklake", "fallback_stores"})
        store = cls(ducklake=config.ducklake, **config_dict, **kwargs)
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
