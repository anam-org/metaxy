# Configuration

See full documentation: https://anam-org.github.io/metaxy/reference/configuration/

## TOML Configuration

```toml
# metaxy.toml

# Default store to use
store = "dev"

# Feature discovery paths
entrypoints = ["src/my_project/features"]

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
auto_create_tables = true
```

## Environment Variable Templating

```toml
[stores.branch]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
[stores.branch.config]
root_path = "s3://my-bucket/${BRANCH_NAME}/metadata"
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
    stores={"dev": mx.DeltaMetadataStore(root_path="/tmp/metaxy", auto_create_tables=True)}
).use() as config:
    store = config.get_store("dev")
```
