"""LanceDB metadata store implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import Field

from metaxy._decorators import public
from metaxy.ext.lancedb.handler import LanceDBHandler
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.storage_config import LanceStorageConfig
from metaxy.metadata_store.utils import sanitize_uri
from metaxy.models.types import FeatureKey

logger = logging.getLogger(__name__)


@public
class LanceDBMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for LanceDBMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.lancedb.LanceDBMetadataStore"

        [stores.dev.config]
        uri = "/path/to/featuregraph"

        [stores.dev.config.connect_kwargs]
        api_key = "your-api-key"
        ```
    """

    uri: str | Path = Field(
        description="Directory path or URI for LanceDB tables.",
    )
    connect_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Extra keyword arguments passed to lancedb.connect().",
    )


@public
class LanceDBMetadataStore(MetadataStore):
    """
    [LanceDB](https://lancedb.github.io/lancedb/) metadata store for vector and structured data.

    LanceDB is a columnar database optimized for vector search and multimodal data.
    Each feature is stored in its own Lance table within the database directory.
    Uses Polars components for data processing (no native SQL execution).

    Storage layout:

    - Each feature gets its own table: `{namespace}__{feature_name}`

    - Tables are stored as Lance format in the directory specified by the URI

    - LanceDB handles schema evolution, transactions, and compaction automatically

    Example: Local Directory
        ```py
        from pathlib import Path
        from metaxy.ext.lancedb import LanceDBMetadataStore

        # Local filesystem
        store = LanceDBMetadataStore(Path("/path/to/featuregraph"))
        ```

    Example: Object Storage (S3, GCS, Azure)
        ```py
        # object store (requires credentials)
        store = LanceDBMetadataStore("s3:///path/to/featuregraph")
        ```

    Example: LanceDB Cloud
        ```py
        import os

        # Option 1: Environment variable
        os.environ["LANCEDB_API_KEY"] = "your-api-key"
        store = LanceDBMetadataStore("db://my-database")

        # Option 2: Explicit credentials
        store = LanceDBMetadataStore(
            "db://my-database", connect_kwargs={"api_key": "your-api-key", "region": "us-east-1"}
        )
        ```
    """

    _should_warn_auto_create_tables = False

    def __init__(
        self,
        uri: str | Path | None = None,
        *,
        fallback_stores: list[MetadataStore] | None = None,
        connect_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if uri is None:
            raise ValueError("uri is required")

        from metaxy.ext.polars.engine import PolarsComputeEngine

        super().__init__(
            engine=PolarsComputeEngine(handlers=[LanceDBHandler()]),
            storage=LanceStorageConfig(location=str(uri), connect_kwargs=connect_kwargs or {}),
            fallback_stores=fallback_stores,
            auto_create_tables=True,
            **kwargs,
        )

    @property
    def uri(self) -> str:
        return self._write_storage().location

    @property
    def conn(self) -> Any:
        """The LanceDB connection.

        Raises:
            StoreNotOpenError: If the store is not open
        """
        from metaxy.metadata_store.exceptions import StoreNotOpenError

        if not self._is_open:
            raise StoreNotOpenError("LanceDB connection is not open. Store must be used as a context manager.")
        storage = self._write_storage()
        assert isinstance(storage, LanceStorageConfig)
        handler = self._engine.get_handler(storage)
        assert isinstance(handler, LanceDBHandler)
        return handler.connection(storage.location, dict(storage.connect_kwargs))

    @property
    def _conn(self) -> Any:
        return self.conn if self._is_open else None

    def _table_name(self, feature_key: FeatureKey) -> str:
        return feature_key.table_name

    def display(self) -> str:
        """Human-readable representation with sanitized credentials."""
        return f"LanceDBMetadataStore(path={sanitize_uri(self.uri)})"

    @classmethod
    def config_model(cls) -> type[LanceDBMetadataStoreConfig]:
        return LanceDBMetadataStoreConfig
