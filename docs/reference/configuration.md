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
    type = "metaxy.ext.polars.handlers.delta.DeltaMetadataStore"
    [stores.dev.config]
    root_path = "${HOME}/.metaxy/metadata"
    ```

### Configuration Priority

When the same setting is defined in multiple places, Metaxy uses the following priority order (highest to lowest):

1. **Explicit arguments** - Values passed directly to `MetaxyConfig()`
2. **Environment variables** - Values from `METAXY_*` environment variables
3. **Configuration files** - Values from `metaxy.toml` or `pyproject.toml`

### Store Environment Overrides

Metaxy requires `pydantic-settings>=2.13.1`. Put list and dictionary store options inside the containing `CONFIG` JSON object:

```shell
export METAXY_STORES__DEV__CONFIG='{"fallback_stores": [], "storage_options": {"AWS_REGION": "eu-central-1"}}'
```

This clears the configured fallback list. Dictionary values merge with the TOML configuration, so other storage options remain available.

Replace collection overrides such as `METAXY_STORES__DEV__CONFIG__FALLBACK_STORES='[]'` with this form. Remove the old variables when migrating.

Scalar options can still use individual variables:

```shell
export METAXY_STORES__DEV__CONFIG__ROOT_PATH=123
```

The root path remains the string `"123"`.

### Config Discovery

Configuration files are discovered automatically by searching in the current or parent directories. `metaxy.toml` takes precedence over `pyproject.toml`.

### Templating Environment Variables

Metaxy supports templating environment variables in configuration files using the `${VARIABLE_NAME}` syntax.

!!! example

    ```toml {title="metaxy.toml"}
    [stores.branch.config]
    root_path = "s3://my-bucket/${BRANCH_NAME}"
    ```

### Configuration Inheritance

A configuration file can inherit settings from another file using the [`extend`](#metaxy.config.MetaxyConfig.extend) field. This is especially useful in monorepos: a shared `metaxy.toml` can define stores, plugins, and other common settings.

```toml title="metaxy.toml"
project = "my_project"
extend = "../base.toml"
```

The `extend` path can be either relative to the file that contains it or an absolute path.

When a child config extends a parent, scalar fields are replaced, dictionary fields are shallow-merged, and array fields are extended.

## Configuration Options

::: metaxy-config
    class: metaxy.config.MetaxyConfig
    header_level: 3
    exclude_fields: theme

---

The [`stores`](#metaxy.config.MetaxyConfig.stores) field configures metadata store backends. This is a mapping of store names to their configurations. The [default store](#metaxy.config.MetaxyConfig.store) is named `"dev"`.

::: metaxy-config
    class: metaxy.StoreConfig
    header_level: 3
    path_prefix: stores.dev
