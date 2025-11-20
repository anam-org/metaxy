# Configuration

Metaxy can be configured using TOML configuration files, environment variables, or programmatically.

## Configuration Priority

When the same setting is defined in multiple places, Metaxy uses the following priority order (highest to lowest):

1. **Explicit arguments** - Values passed directly to `MetaxyConfig()`
2. **Environment variables** - Values from `METAXY_*` environment variables
3. **Configuration files** - Values from `metaxy.toml` or `pyproject.toml`

Configuration files are discovered automatically by searching in the current or parent directories.

## Fields

::: metaxy-config
    class: metaxy.config.MetaxyConfig
    header_level: 2
:::

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

### Available Store Types

| Store Type | Import Path | Description |
|------------|-------------|-------------|
| DuckDB | `metaxy.metadata_store.duckdb.DuckDBMetadataStore` | File-based or in-memory DuckDB backend |
| ClickHouse | `metaxy.metadata_store.clickhouse.ClickHouseMetadataStore` | ClickHouse database backend |
| In-Memory | `metaxy.metadata_store.memory.InMemoryMetadataStore` | In-memory backend for testing |

### Getting a Store Instance

```python
from metaxy.config import MetaxyConfig

config = MetaxyConfig.load()

# Get the default store
with config.get_store() as store:
    # Use store
    pass

# Get a specific store by name
with config.get_store("prod") as store:
    # Use store
    pass
```

## Plugin Configuration

Plugins are configured under the `ext` field. Each plugin has its own configuration options documented in the integration guides:

- [SQLModel Plugin Configuration](../learn/integrations/sqlmodel.md#configuration)
