---
title: "Metaxy + DuckLake"
description: "Store Metaxy metadata in DuckLake."
---

# DuckLake

!!! warning "Experimental"

    This functionality is experimental.

[DuckLake](https://ducklake.select) is a modern LakeHouse which uses a relational database as metadata catalog.

Currently, there is only one production-ready implementation of DuckLake - via DuckDB, and the built-in [`DuckDBMetadataStore`][metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore] can be configured to use DuckLake as its storage backend. Learn more about the DuckDB integration [here](/integrations/metadata-stores/databases/duckdb.md).

## Configuration

There are two main parts that configure DuckLake: a **catalog** (where the transaction log and other metadata is stored) and a **storage** (where the data files (1) live).
{ .annotate }

1. Parquet files

Each piece of configuration that manages secrets (e.g. PostgreSQL, S3, R2, GCS) requires a `secret_name` parameter. Metaxy uses this name to either create a new DuckDB secret (when inline credentials are provided) or reference a pre-existing one (when only the name is given).

!!! tip

    To use the credential chain (IAM roles, environment variables, etc.) instead of static S3 credentials, set `secret_parameters = { provider = "credential_chain" }`.
    Learn more in [DuckDB docs](https://duckdb.org/docs/stable/core_extensions/httpfs/s3api#credential_chain-provider).

!!! example "Example Configuration"

    ```toml
    [stores.dev]
    type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

    [stores.dev.config.ducklake.catalog]
    type = "postgres"
    secret_name = "my_pg_secret"
    host = "localhost"
    port = 5432
    database = "ducklake_meta"
    user = "ducklake"
    password = "changeme"

    [stores.dev.config.ducklake.storage]
    type = "s3"
    secret_name = "my_s3_secret"
    bucket = "my-ducklake-bucket"
    key_id = "AKIA..."
    secret = "..."
    region = "eu-central-1"
    ```

See the [DuckLake example](/examples/ducklake.md) to learn more.

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.DuckLakeConfig
    path_prefix: stores.dev.config.ducklake
    header_level: 3

### Catalog Backends

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.DuckDBCatalogConfig
    path_prefix: stores.dev.config.ducklake.catalog
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.SQLiteCatalogConfig
    path_prefix: stores.dev.config.ducklake.catalog
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.PostgresCatalogConfig
    path_prefix: stores.dev.config.ducklake.catalog
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.MotherDuckCatalogConfig
    path_prefix: stores.dev.config.ducklake.catalog
    header_level: 4

### Storage Backends

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.LocalStorageConfig
    path_prefix: stores.dev.config.ducklake.storage
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.S3StorageConfig
    path_prefix: stores.dev.config.ducklake.storage
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.R2StorageConfig
    path_prefix: stores.dev.config.ducklake.storage
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.GCSStorageConfig
    path_prefix: stores.dev.config.ducklake.storage
    header_level: 4
