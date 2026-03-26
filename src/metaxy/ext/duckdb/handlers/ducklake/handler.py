"""Storage handler for DuckLake tables accessed through a DuckDB connection."""

from __future__ import annotations

from typing import Any

from metaxy.ext.duckdb.config import ExtensionSpec
from metaxy.ext.duckdb.handlers.ducklake.attachment import DuckLakeAttachmentManager
from metaxy.ext.duckdb.handlers.ducklake.config import DuckLakeConfig, MotherDuckCatalogConfig
from metaxy.metadata_store.ibis_compute_engine import DuckLakeStorageConfig, IbisSQLHandler
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.models.types import FeatureKey


class DuckDBDuckLakeHandler(IbisSQLHandler):
    """Storage handler for DuckLake tables accessed through a DuckDB connection.

    Attachment happens via the ``on_connection_opened`` hook — the engine
    calls it after connecting, before any I/O.
    """

    def __init__(
        self,
        ducklake_config: DuckLakeConfig,
        *,
        auto_create_tables: bool = False,
        store_name: str | None = None,
    ) -> None:
        super().__init__(auto_create_tables=auto_create_tables)
        self._config = ducklake_config
        self._attachment = DuckLakeAttachmentManager(ducklake_config, store_name=store_name)

    # -- capability: only handles DuckLakeStorageConfig -----------------------

    def can_read(self, storage_config: StorageConfig, key: FeatureKey) -> bool:  # noqa: ARG002
        return isinstance(storage_config, DuckLakeStorageConfig)

    def can_write(self, storage_config: StorageConfig, key: FeatureKey) -> bool:  # noqa: ARG002
        return isinstance(storage_config, DuckLakeStorageConfig)

    # -- lifecycle hook -------------------------------------------------------

    def on_connection_opened(self, conn: Any) -> None:
        """ATTACH the DuckLake catalog on the live DuckDB connection."""
        raw_conn = conn.con
        self._attachment._attached = False
        self._attachment.configure(raw_conn)

    # -- public DuckLake API -------------------------------------------------

    @property
    def ducklake_config(self) -> DuckLakeConfig:
        return self._config

    @property
    def attachment_manager(self) -> DuckLakeAttachmentManager:
        return self._attachment

    def preview_ducklake_sql(self) -> list[str]:
        return self._attachment.preview_sql()

    def required_extensions(self) -> list[ExtensionSpec]:
        """Extensions this handler needs loaded on the DuckDB connection."""
        exts = [ExtensionSpec(name="ducklake")]
        if isinstance(self._config.catalog, MotherDuckCatalogConfig):
            exts.append(ExtensionSpec(name="motherduck"))
        return exts
