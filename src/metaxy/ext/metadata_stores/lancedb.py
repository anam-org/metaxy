"""Deprecated: use metaxy.ext.lancedb.metadata_store instead."""

from metaxy._warnings import _warn_deprecated_module
from metaxy.ext.lancedb import LanceDBMetadataStore, LanceDBMetadataStoreConfig

_warn_deprecated_module("metaxy.ext.metadata_stores.lancedb", "metaxy.ext.lancedb")

__all__ = ["LanceDBMetadataStore", "LanceDBMetadataStoreConfig"]
