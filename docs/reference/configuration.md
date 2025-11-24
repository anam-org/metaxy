# Configuration

Metaxy can be configured using TOML configuration files, environment variables, or programmatically.

## Configuration Priority

When the same setting is defined in multiple places, Metaxy uses the following priority order (highest to lowest):

1. **Explicit arguments** - Values passed directly to `MetaxyConfig()`
2. **Environment variables** - Values from `METAXY_*` environment variables
3. **Configuration files** - Values from `metaxy.toml` or `pyproject.toml`

Configuration files are discovered automatically by searching in the current or parent directories.

## Configuration Options

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.config.MetaxyConfig
    header_level: 3
<!-- dprint-ignore-end -->

## Store Configuration

The `stores` field configures metadata store backends. Each store is defined by:

- **`type`**: Full import path to the store class (e.g., `metaxy.metadata_store.duckdb.DuckDBMetadataStore`)
- **`config`**: Dictionary of store-specific configuration options

### Example: Multiple Stores with Fallback Stores

=== "metaxy.toml"

    ```toml
    # Default store to use
    store = "dev"

    # Development store (in-memory) with fallback to production
    [stores.dev]
    type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
    [stores.dev.config]
    db_path = ":memory:"
    fallback_stores = ["prod"]

    # Production store
    [stores.prod]
    type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
    [stores.prod.config]
    db_path = "s3://my-bucket/metadata.duckdb"
    ```

=== "pyproject.toml"

    ```toml
    [tool.metaxy]
    store = "dev"

    [tool.metaxy.stores.dev]
    type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
    [tool.metaxy.stores.dev.config]
    db_path = ":memory:"
    fallback_stores = ["prod"]

    [tool.metaxy.stores.prod]
    type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
    [tool.metaxy.stores.prod.config]
    db_path = "s3://my-bucket/metadata.duckdb"
    ```

## Configuring Metadata Stores

Configuration options for metadata stores can be found at the respective store documentation page.

## Configuring Metaxy Plugins

Configuration options for Metaxy plugins can be found at the respective plugin documentation page.
