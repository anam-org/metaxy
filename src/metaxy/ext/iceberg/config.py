"""Iceberg metadata store configuration."""

from __future__ import annotations

from pathlib import Path
from typing import NewType

from pydantic import Field

from metaxy._decorators import public
from metaxy.metadata_store.base import MetadataStoreConfig

TableIdentifier = NewType("TableIdentifier", tuple[str, str])
"""A ``(namespace, table_name)`` pair used by PyIceberg to locate a table within a catalog."""


@public
class IcebergMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for IcebergMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.metadata_stores.iceberg.IcebergMetadataStore"

        [stores.dev.config]
        warehouse = "/path/to/warehouse"
        namespace = "metaxy"

        [stores.dev.config.catalog_properties]
        type = "sql"
        ```
    """

    warehouse: str | Path = Field(
        description="Warehouse directory or URI where Iceberg tables are stored.",
    )
    namespace: str = Field(
        default="metaxy",
        description="Iceberg namespace for feature tables.",
    )
    catalog_name: str = Field(
        default="metaxy",
        description="Name of the Iceberg catalog.",
    )
    catalog_properties: dict[str, str] | None = Field(
        default=None,
        description="Properties passed to pyiceberg.catalog.load_catalog.",
    )
    auto_create_namespace: bool = Field(
        default=True,
        description="Automatically create the namespace on first write if it does not exist.",
    )
