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

Each storage backend that manages secrets (PostgreSQL, S3, R2, GCS) requires a `secret_name`. Metaxy uses this name to either create a new DuckDB secret (when inline credentials are provided) or reference a pre-existing one (when only the name is given).

!!! tip

    To use the credential chain (IAM roles, environment variables, etc.) instead of static credentials, set `secret_parameters = { provider = "credential_chain" }` on S3, R2, or GCS storage backends.

!!! example "Example Configuration"

    ```toml
    [stores.dev]
    type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

    [stores.dev.config.ducklake.metadata_backend]
    type = "postgres"
    secret_name = "my_pg_secret"
    host = "localhost"
    port = 5432
    database = "ducklake_meta"
    user = "ducklake"
    password = "changeme"

    [stores.dev.config.ducklake.storage_backend]
    type = "s3"
    secret_name = "my_s3_secret"
    bucket = "my-ducklake-bucket"
    key_id = "AKIA..."
    secret = "..."
    region = "eu-central-1"
    ```

See the [DuckLake example](/examples/ducklake.md) to learn more.

### MotherDuck Bring Your Own Bucket (BYOB)

MotherDuck can manage the DuckLake catalog while you provide your own S3-compatible storage. In this mode Metaxy creates the DuckLake database with a custom `DATA_PATH` and the storage secret is created `IN MOTHERDUCK` so that MotherDuck compute can access your bucket.

!!! example "MotherDuck BYOB Configuration"

    ```toml
    [stores.dev.config.ducklake.metadata_backend]
    type = "motherduck"
    database = "my_ducklake"
    region = "eu-central-1"

    [stores.dev.config.ducklake.storage_backend]
    type = "s3"
    secret_name = "my_s3_secret"
    key_id = "AKIA..."
    secret = "..."
    region = "eu-central-1"
    scope = "s3://mybucket/"
    bucket = "mybucket"
    ```

Without a `storage_backend`, MotherDuck uses its own fully managed storage and no additional configuration is needed.

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.DuckLakeAttachmentConfig
    path_prefix: stores.dev.config.ducklake
    header_level: 3

### Metadata Backends

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.DuckDBMetadataBackendConfig
    path_prefix: stores.dev.config.ducklake.metadata_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.SQLiteMetadataBackendConfig
    path_prefix: stores.dev.config.ducklake.metadata_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.PostgresMetadataBackendConfig
    path_prefix: stores.dev.config.ducklake.metadata_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.MotherDuckMetadataBackendConfig
    path_prefix: stores.dev.config.ducklake.metadata_backend
    header_level: 4

### Storage Backends

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.LocalStorageBackendConfig
    path_prefix: stores.dev.config.ducklake.storage_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.S3StorageBackendConfig
    path_prefix: stores.dev.config.ducklake.storage_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.R2StorageBackendConfig
    path_prefix: stores.dev.config.ducklake.storage_backend
    header_level: 4

::: metaxy-config
    class: metaxy.ext.metadata_stores.ducklake.GCSStorageBackendConfig
    path_prefix: stores.dev.config.ducklake.storage_backend
    header_level: 4
