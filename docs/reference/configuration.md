---
title: "Configuration Reference"
description: "Configuration options for Metaxy."
---

# Configuration

## Configuring Metaxy

Metaxy can be configured using TOML configuration files, environment variables, or programmatically.

Either TOML-based or environment-based configuration is **required** to use the [Metaxy CLI](/reference/cli.md).

!!! example

    ```toml title="metaxy.toml"
    project = "my_package"

    [stores.dev]
    type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
    [stores.dev.config]
    root_path = "${HOME}/.metaxy/metadata"
    ```

### Configuration Priority

When the same setting is defined in multiple places, Metaxy uses the following priority order (highest to lowest):

1. **Explicit arguments** - Values passed directly to `MetaxyConfig()`
2. **Environment variables** - Values from `METAXY_*` environment variables
3. **Configuration files** - Values from `metaxy.toml` or `pyproject.toml`

### Config Discovery

Configuration files are discovered automatically by searching in the current or parent directories. `metaxy.toml` takes precedence over `pyproject.toml`.

### Templating Environment Variables

Metaxy supports templating environment variables in configuration files using the `${VARIABLE_NAME}` syntax.

!!! example

    ```toml {title="metaxy.toml"}
    [stores.branch.config]
    root_path = "s3://my-bucket/${BRANCH_NAME}"
    ```

## Configuration Options

::: metaxy-config
    class: metaxy.config.MetaxyConfig
    header_level: 3
    exclude_fields: migrations_dir,theme

---

The [`stores`](#metaxy.config.MetaxyConfig.stores) field configures metadata store backends. This is a mapping of store names to their configurations. The [default store](#metaxy.config.MetaxyConfig.store) is named `"dev"`.

::: metaxy-config
    class: metaxy.StoreConfig
    header_level: 3
    path_prefix: stores.dev
