"""Deprecated: use metaxy.ext.polars.handlers.iceberg instead."""

from metaxy._warnings import _warn_deprecated_module
from metaxy.ext.polars.handlers.iceberg import (
    IcebergMetadataStore,
    IcebergMetadataStoreConfig,
    TableIdentifier,
)

_warn_deprecated_module("metaxy.ext.metadata_stores.iceberg", "metaxy.ext.polars.handlers.iceberg")

__all__ = ["IcebergMetadataStore", "IcebergMetadataStoreConfig", "TableIdentifier"]
