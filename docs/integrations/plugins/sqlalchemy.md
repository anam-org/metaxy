---
title: "SQLAlchemy Plugin"
description: "SQLAlchemy integration for database migrations."
---

# SQLAlchemy

!!! warning "Experimental"

    This functionality is experimental.

Metaxy provides helpers for integrating with [SQLAlchemy](https://www.sqlalchemy.org/).
These helpers allow to construct `sqlalchemy.MetaData` objects for user-defined feature tables and for Metaxy system tables.

This integration is convenient for setting up [Alembic](https://alembic.sqlalchemy.org/en/latest/) migrations.

!!! tip "SQLModel Features"

    Check out our [SQLModel integration](sqlmodel.md) for metaclass features that combine Metaxy features with SQLModel ORM models. This is the recommended way to use Metaxy with SQLAlchemy.

## Alembic Integration

[Alembic](https://alembic.sqlalchemy.org/en/latest/) is a database migration toolkit for SQLAlchemy.

The two helper functions: [`filter_feature_sqla_metadata`][metaxy.ext.sqlalchemy.filter_feature_sqla_metadata] and [`get_system_slqa_metadata`][metaxy.ext.sqlalchemy.get_system_slqa_metadata] can be used to retrieve an SQLAlchemy URL and `MetaData` object for a given [`IbisMetadataStore`][metaxy.metadata_store.ibis.IbisMetadataStore].

`filter_feature_sqla_metadata` returns table metadata for the user-defined tables, while `get_system_slqa_metadata` returns metadata for Metaxy's system tables.

!!! important "Call `init` first"

    You **must** call [`init`][metaxy.init] before using `filter_feature_sqla_metadata` to ensure all features are loaded into the feature graph.

Here is an example Alembic `env.py` that uses the Metaxy SQLAlchemy integration:

```python title="env.py"
from alembic import context
from metaxy import init
from metaxy.ext.sqlalchemy import filter_feature_sqla_metadata

# Alembic Config object
config = context.config

metaxy_cfg = init()
store = metaxy_cfg.get_store("my_store")

# import your SQLAlchemy metadata from somewhere
my_metadata = ...

url, target_metadata = filter_feature_sqla_metadata(my_metadata, store)

# Configure Alembic
config.set_main_option("sqlalchemy.url", url)


# continue with the standard Alembic workflow
```

!!! tip "Naming Conventions"

    For consistent, deterministic and readable constraint names in your migrations, configure [naming conventions](https://alembic.sqlalchemy.org/en/latest/naming.html).

## Multi-Store Setup

You can configure separate metadata stores for different environments:

```toml title="metaxy.toml"
[stores.dev]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"
config = { database = "dev_metadata.duckdb" }

[stores.prod]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"
config = { database = "prod_metadata.duckdb" }
```

Then create multiple Alembic directories and register them with Alembic:

```ini title="alembic.ini"
[dev]
script_location = alembic/dev

[prod]
script_location = alembic/prod
```

!!! tip "Separate Alembic Version Tables"

    When using multiple Alembic environments (e.g., system tables vs feature tables), configure separate version tables to avoid conflicts. Set up separate script locations in `alembic.ini`:

    ```ini title="alembic.ini"
    [dev:metaxy_system]
    script_location = alembic/dev/system

    [dev:metaxy_features]
    script_location = alembic/dev/features
    ```

    Then pass `version_table` to `context.configure()` in each env.py:

    ```python title="alembic/dev/system/env.py"
    context.configure(
        url=url,
        target_metadata=target_metadata,
        version_table="alembic_version_metaxy_system",
    )
    ```

    ```python title="alembic/dev/features/env.py"
    context.configure(
        url=url,
        target_metadata=target_metadata,
        version_table="alembic_version_metaxy_features",
    )
    ```

    Each environment now tracks migrations independently:

    - `alembic_version_metaxy_system` for system tables
    - `alembic_version_metaxy_features` for feature tables

    Create and run migrations separately:

    ```bash
    alembic -n dev:metaxy_system revision --autogenerate -m "initialize"
    alembic -n dev:metaxy_features revision --autogenerate -m "initialize"
    alembic -n dev:metaxy_system upgrade head
    alembic -n dev:metaxy_features upgrade head
    ```

The two environments now can be managed independently:

=== "dev"

    ```python title="alembic/dev/env.py" hl_lines="3"
    from metaxy import init
    config = init()
    store = config.get_store("dev")
    url, target_metadata = filter_feature_sqla_metadata(my_metadata, store)
    ```

    The `-n` argument can be used to specify the target Alembic directory:

    ```bash
    alembic -n dev upgrade head
    ```

=== "prod"

    ```python title="alembic/prod/env.py"  hl_lines="3"
    from metaxy import init
    config = init()
    store = config.get_store("prod")
    url, target_metadata = filter_feature_sqla_metadata(my_metadata, store)
    ```

    The `-n` argument can be used to specify the target Alembic directory:

    ```bash
    alembic -n prod upgrade head
    ```

### Alembic + SQLModel

To throw `SQLModel` into the mix, make sure to use the [SQLModel integration](sqlmodel.md) and pass `sqlmodel.SQLModel.metadata` into `filter_feature_sqla_metadata`.

## API Reference

<!-- dprint-ignore-start -->
::: metaxy.ext.sqlalchemy
    options:
      members: false
      show_root_heading: true
      heading_level: 3

::: metaxy.ext.sqlalchemy.filter_feature_sqla_metadata
    options:
      members: false
      heading_level: 3

::: metaxy.ext.sqlalchemy.get_system_slqa_metadata
    options:
      members: false
      heading_level: 3
<!-- dprint-ignore-end -->

## Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.sqlalchemy.SQLAlchemyConfig
    path_prefix: ext.sqlalchemy
    header_level: 2
<!-- dprint-ignore-end -->
