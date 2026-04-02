"""DuckDB metadata store extension."""

from metaxy.ext.duckdb.handlers.ducklake import (
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
from metaxy.ext.duckdb.metadata_store import DuckDBMetadataStore, DuckDBMetadataStoreConfig, ExtensionSpec

__all__ = [
    "DuckDBCatalogConfig",
    "DuckDBMetadataStore",
    "DuckDBMetadataStoreConfig",
    "DuckDBPyConnection",
    "DuckLakeAttachmentManager",
    "DuckLakeConfig",
    "ExtensionSpec",
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
