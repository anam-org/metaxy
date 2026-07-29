---
title: "Metaxy + DuckLake"
description: "Learn how to store Metaxy metadata in DuckLake."
---

# Metaxy + DuckLake

!!! warning "Experimental"

    This functionality is experimental.

[DuckLake](https://ducklake.select) is a modern LakeHouse which uses a relational database as metadata catalog.

Currently, there is only one production-ready implementation of DuckLake - via DuckDB, and the built-in [`DuckDBMetadataStore`][metaxy.ext.duckdb.DuckDBMetadataStore] can be configured to use DuckLake as its storage backend. Learn more about the DuckDB integration [here](/integrations/metadata-stores/databases/duckdb.md).

!!! warning "Current MotherDuck limitation"

    MotherDuck-backed DuckLake stores currently hit a limitation tracked in [issue #1043](https://github.com/anam-org/metaxy/issues/1043): as community extensions for `hashfuncs` are not yet available in MotherDuck.

    Two workarounds are possible:

    - Set `versioning_engine = "polars"` to keep versioning on the client side and avoid the failing MotherDuck SQL path.
    - Change `hash_algorithm` to a server-supported hash such as `sha256` if you want to keep versioning native.

    Changing the versioning engine is the safer workaround if you want to preserve current `xxhash`-based version semantics. Changing the hash algorithm also works, but it changes the computed version / provenance values, so keep it consistent across existing stores and fallback chains.

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
    type = "metaxy.ext.duckdb.DuckDBMetadataStore"

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

## MotherDuck workarounds

If you are using a MotherDuck catalog and run into the limitation above, use one of these configurations for now.

### Keep `xxhash` semantics: use the Polars versioning engine

```toml
[stores.dev.config]
versioning_engine = "polars"
```

### Keep native execution: switch to a server-supported hash

```toml
[stores.dev.config]
hash_algorithm = "sha256"
```

Use the second option only if you are comfortable changing hash semantics for that store. In particular, avoid mixing different hash algorithms across fallback stores.

See the [DuckLake example](/examples/ducklake.md) to learn more.

::: metaxy-config
    class: metaxy.ext.duckdb.DuckLakeConfig
    path_prefix: stores.dev.config.ducklake
    header_level: 3

### Catalog Backends

::: metaxy-config
    class: metaxy.ext.duckdb.DuckDBCatalogConfig
    path_prefix: stores.dev.config.ducklake.catalog
    header_level: 4

::: metaxy-config
    class: metaxy.ext.duckdb.SQLiteCatalogConfig
    path_prefix: stores.dev.config.ducklake.catalog
    header_level: 4

::: metaxy-config
    class: metaxy.ext.duckdb.PostgresCatalogConfig
    path_prefix: stores.dev.config.ducklake.catalog
    header_level: 4

::: metaxy-config
    class: metaxy.ext.duckdb.MotherDuckCatalogConfig
    path_prefix: stores.dev.config.ducklake.catalog
    header_level: 4

### Storage Backends

::: metaxy-config
    class: metaxy.ext.duckdb.LocalStorageConfig
    path_prefix: stores.dev.config.ducklake.storage
    header_level: 4

::: metaxy-config
    class: metaxy.ext.duckdb.S3StorageConfig
    path_prefix: stores.dev.config.ducklake.storage
    header_level: 4

::: metaxy-config
    class: metaxy.ext.duckdb.R2StorageConfig
    path_prefix: stores.dev.config.ducklake.storage
    header_level: 4

::: metaxy-config
    class: metaxy.ext.duckdb.GCSStorageConfig
    path_prefix: stores.dev.config.ducklake.storage
    header_level: 4
