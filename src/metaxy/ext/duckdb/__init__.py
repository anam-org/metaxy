"""DuckDB metadata store extension."""

from metaxy.ext.duckdb.engine import DuckDBEngine, ExtensionSpec
from metaxy.ext.duckdb.handlers.ducklake import (
    DuckDBCatalogConfig,
    DuckDBDuckLakeHandler,
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
from metaxy.ext.duckdb.handlers.lance import DuckDBLanceHandler
from metaxy.ext.duckdb.metadata_store import DuckDBMetadataStore, DuckDBMetadataStoreConfig

__all__ = [
    "DuckDBCatalogConfig",
    "DuckDBDuckLakeHandler",
    "DuckDBEngine",
    "DuckDBLanceHandler",
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
