"""Delta Lake metadata store configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from metaxy._decorators import public
from metaxy.metadata_store.base import MetadataStoreConfig


@public
class DeltaMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for DeltaMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"

        [stores.dev.config]
        root_path = "s3://my-bucket/metaxy"
        layout = "nested"

        [stores.dev.config.storage_options]
        AWS_REGION = "us-west-2"
        ```
    """

    root_path: str | Path = Field(
        description="Base directory or URI where feature tables are stored.",
    )
    storage_options: dict[str, Any] | None = Field(
        default=None,
        description="Storage backend options passed to delta-rs.",
    )
    layout: Literal["flat", "nested"] = Field(
        default="nested",
        description="Directory layout for feature tables ('nested' or 'flat').",
    )
    delta_write_options: dict[str, Any] | None = Field(
        default=None,
        description="Options passed to [`deltalake.write_deltalake`][deltalake.write_deltalake].",
    )
