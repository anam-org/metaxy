---
title: "Configuration Reference"
description: "Configuration options for Metaxy."
---

# Configuration

Metaxy can be configured using TOML configuration files, environment variables, or programmatically.

Either TOML-based or environment-based configuration is **required** to use the [Metaxy CLI](/reference/cli.md).

## Configuration Priority

When the same setting is defined in multiple places, Metaxy uses the following priority order (highest to lowest):

1. **Explicit arguments** - Values passed directly to `MetaxyConfig()`
2. **Environment variables** - Values from `METAXY_*` environment variables
3. **Configuration files** - Values from `metaxy.toml` or `pyproject.toml`

## Config Discovery

Configuration files are discovered automatically by searching in the current or parent directories. `metaxy.toml` takes precedence over `pyproject.toml`.

## Templating Environment Variables

Metaxy supports templating environment variables in configuration files using the `${VARIABLE_NAME}` syntax.

!!! example

    ```toml {title="metaxy.toml"}
    [stores.branch.config]
    root_path = "s3://my-bucket/${BRANCH_NAME}"
    ```

## Configuration Options

### Store Configuration

The `stores` field configures metadata store backends. Each store is defined by:

- **`type`**: Full import path to the store class (e.g., `metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore`)
- **`config`**: Dictionary of store-specific configuration options
- `**fallback_stores**` - optional list of store names to pull missing metadata from. Learn more [here](/guide/learn/metadata-stores.md/#fallback-stores).

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.config.MetaxyConfig
    header_level: 3
<!-- dprint-ignore-end -->
