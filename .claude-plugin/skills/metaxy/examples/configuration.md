# Configuration

See full documentation: https://docs.metaxy.io/latest/reference/configuration/

## TOML Configuration

```toml
# metaxy.toml

# Default store to use
store = "dev"

# Feature discovery paths
entrypoints = ["src/my_project/features"]

# Auto-create tables (global setting, not recommended for production)
# auto_create_tables = true

# Enable native Arrow Map columns (experimental, requires polars-map)
# enable_map_datatype = true

# Development store
[stores.dev]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
[stores.dev.config]
root_path = "${HOME}/.metaxy/metadata"

# Production store with S3
[stores.prod]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
[stores.prod.config]
root_path = "s3://my-bucket/metadata"
```

## pyproject.toml

```toml
[tool.metaxy]
store = "dev"
entrypoints = ["src/my_project/features"]

[tool.metaxy.stores.dev]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
[tool.metaxy.stores.dev.config]
root_path = "/tmp/metaxy/metadata"
```

## Environment Variable Templating

```toml
[stores.branch]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
[stores.branch.config]
root_path = "s3://my-bucket/${BRANCH_NAME}/metadata"
```

## Config Inheritance (for Monorepos)

```toml
# base.toml (shared store definitions)
[stores.dev]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"
[stores.dev.config]
database = "${HOME}/.metaxy/shared.duckdb"
```

```toml
# project_a/metaxy.toml
project = "project_a"
extend = "../base.toml"
```

```toml
# project_b/metaxy.toml
project = "project_b"
extend = "../base.toml"
```

## Initialize from Config

```python
import metaxy as mx

config = mx.init()  # Load from metaxy.toml or pyproject.toml
store = config.get_store("dev")
```

## Programmatic Configuration

```python
import metaxy as mx

with mx.MetaxyConfig(
    stores={"dev": mx.StoreConfig(
        type="metaxy.ext.metadata_stores.delta.DeltaMetadataStore",
        config={"root_path": "/tmp/metaxy"},
    )}
).use() as config:
    store = config.get_store("dev")
```
