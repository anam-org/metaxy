"""LanceDB metadata store configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from metaxy._decorators import public
from metaxy.metadata_store.base import MetadataStoreConfig


@public
class LanceDBMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for LanceDBMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.metadata_stores.lancedb.LanceDBMetadataStore"

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
