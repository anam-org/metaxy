"""Backwards-compatible re-exports from the ``metaxy.ext.duckdb`` subpackage."""

from metaxy.ext.duckdb.config import DuckDBMetadataStoreConfig, ExtensionSpec
from metaxy.ext.duckdb.engine import DuckDBEngine
from metaxy.ext.duckdb.handlers.ducklake.handler import DuckDBDuckLakeHandler
from metaxy.ext.duckdb.legacy_store import DuckDBMetadataStore

__all__ = [
    "DuckDBDuckLakeHandler",
    "DuckDBEngine",
    "DuckDBMetadataStore",
    "DuckDBMetadataStoreConfig",
    "ExtensionSpec",
]
