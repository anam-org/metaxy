"""Passive descriptor for a storage backend's location and format.

`StorageConfig` is a frozen Pydantic model that answers "where is the data?"
without any behaviour.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StorageConfig(BaseModel):
    """Base storage configuration.

    Subclass per backend to add typed fields instead of relying on
    an untyped options dict.

    Attributes:
        format: Short identifier for the storage format.
        location: Connection URI, directory path, or object store URI.
        schema: Optional database schema or dataset qualifier.
    """

    model_config = ConfigDict(frozen=True)

    format: str
    location: str
    schema_: str | None = Field(default=None, alias="schema")


class IcebergStorageConfig(StorageConfig):
    """Storage configuration for Apache Iceberg tables.

    Attributes:
        namespace: Iceberg namespace for feature tables.
        catalog_name: Name of the PyIceberg catalog instance.
        catalog_properties: Properties passed to `pyiceberg.catalog.load_catalog`.
        auto_create_namespace: Create the namespace on first write if missing.
    """

    namespace: str = "metaxy"
    catalog_name: str = "metaxy"
    catalog_properties: dict[str, str] | None = None
    auto_create_namespace: bool = True


class LanceStorageConfig(StorageConfig):
    """Storage configuration for Lance tables.

    Attributes:
        connect_kwargs: Extra keyword arguments passed to `lancedb.connect`.
    """

    connect_kwargs: dict[str, object] = Field(default_factory=dict)
