"""Deprecated: use metaxy.ext.duckdb.metadata_store instead."""

from metaxy._warnings import _warn_deprecated_module
from metaxy.ext.duckdb import DuckDBMetadataStore, DuckDBMetadataStoreConfig, ExtensionSpec

_warn_deprecated_module("metaxy.ext.metadata_stores.duckdb", "metaxy.ext.duckdb")

__all__ = ["DuckDBMetadataStore", "DuckDBMetadataStoreConfig", "ExtensionSpec"]
