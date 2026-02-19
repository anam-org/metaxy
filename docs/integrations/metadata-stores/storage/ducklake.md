---
title: "Metaxy + DuckLake"
description: "Store Metaxy metadata in DuckLake."
---

# DuckLake

!!! warning "Experimental"

    This functionality is experimental.

[DuckLake](https://ducklake.select) is a modern LakeHouse which uses a relational database as its metadata backend.

Currently, there is only one production-ready implementation of DuckLake - via DuckDB, and the built-in [`DuckDBMetadataStore`][metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore] can be configured to use DuckLake as its storage backend. Learn more about the DuckDB integration [here](/integrations/metadata-stores/databases/duckdb.md).

## Configuration

There are two main parts that configure DuckLake: a **metadata backend** (where the transaction log and the catalog is stored) and a **storage backend** (where the data files (1) live).
{ .annotate }

1. Parquet files

!!! example "Example Configuration"

    ```toml
    [stores.dev]
    type = "metaxy.ext.metadata_stores.duckdb.DuckDBMEtadataStore"

    [stores.dev.config.ducklake.metadata_backend]
    type = "postgres"
    host = "localhost"
    port = 5432
    database = "ducklake_meta"
    user = "ducklake"
    password = "changeme"
    # Extra parameters forwarded to DuckDB's CREATE SECRET (optional):
    secret_parameters = { sslmode = "require" }
    ```

See the [DuckLake example](/examples/ducklake.md) to learn more.

::: metaxy-config
    class: metaxy.ext.metadata_stores._ducklake_support.DuckLakeAttachmentConfig
    path_prefix: stores.dev.config.ducklake
    header_level: 3

### Metadata Backends

::: metaxy-config
    class: metaxy.ext.metadata_stores._ducklake_support.DuckDBMetadataBackendConfig
    path_prefix: stores.dev.config.ducklake.metadata_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores._ducklake_support.SQLiteMetadataBackendConfig
    path_prefix: stores.dev.config.ducklake.metadata_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores._ducklake_support.PostgresMetadataBackendConfig
    path_prefix: stores.dev.config.ducklake.metadata_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores._ducklake_support.MotherDuckMetadataBackendConfig
    path_prefix: stores.dev.config.ducklake.metadata_backend
    header_level: 4

### Storage Backends

::: metaxy-config
    class: metaxy.ext.metadata_stores._ducklake_support.LocalStorageBackendConfig
    path_prefix: stores.dev.config.ducklake.storage_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores._ducklake_support.S3StorageBackendConfig
    path_prefix: stores.dev.config.ducklake.storage_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores._ducklake_support.R2StorageBackendConfig
    path_prefix: stores.dev.config.ducklake.storage_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores._ducklake_support.GCSStorageBackendConfig
    path_prefix: stores.dev.config.ducklake.storage_backend
    header_level: 4
