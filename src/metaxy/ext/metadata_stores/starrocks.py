"""Deprecated: use metaxy.ext.starrocks.metadata_store instead."""

from metaxy._warnings import _warn_deprecated_module
from metaxy.ext.starrocks import (
    StarRocksMetadataStore,
    StarRocksMetadataStoreConfig,
    StarRocksMySQLMetadataStore,
    StarRocksVersioningEngine,
)

_warn_deprecated_module("metaxy.ext.metadata_stores.starrocks", "metaxy.ext.starrocks")

__all__ = [
    "StarRocksMetadataStore",
    "StarRocksMetadataStoreConfig",
    "StarRocksMySQLMetadataStore",
    "StarRocksVersioningEngine",
]
