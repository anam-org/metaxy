"""Deprecated: use metaxy.ext.duckdb instead."""

from metaxy._warnings import _warn_deprecated_module
from metaxy.ext.duckdb import (
    DuckDBCatalogConfig,
    DuckDBPyConnection,
    DuckLakeAttachmentManager,
    DuckLakeConfig,
    GCSStorageConfig,
    LocalStorageConfig,
    MotherDuckCatalogConfig,
    PostgresCatalogConfig,
    R2StorageConfig,
    S3StorageConfig,
    SQLiteCatalogConfig,
    build_secret_sql,
    format_attach_options,
)

_warn_deprecated_module("metaxy.ext.metadata_stores.ducklake", "metaxy.ext.duckdb")

__all__ = [
    "DuckDBCatalogConfig",
    "DuckDBPyConnection",
    "DuckLakeAttachmentManager",
    "DuckLakeConfig",
    "GCSStorageConfig",
    "LocalStorageConfig",
    "MotherDuckCatalogConfig",
    "PostgresCatalogConfig",
    "R2StorageConfig",
    "S3StorageConfig",
    "SQLiteCatalogConfig",
    "build_secret_sql",
    "format_attach_options",
]
