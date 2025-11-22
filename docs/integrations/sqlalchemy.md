# Metaxy + SQLAlchemy

Metaxy provides helpers for integrating with [SQLAlchemy](https://www.sqlalchemy.org/).
These helpers allow to construct `sqlalchemy.MetaData` objects for user-defined feature tables and for Metaxy system tables.

This integration is convenient for setting up [Alembic](https://alembic.sqlalchemy.org/en/latest/) migrations.

!!! tip

    Check out our [SQLModel integration](../integrations/sqlmodel.md) to learn how to set up `sqlalchemy` metadata for user-defined feature tables.

## Usage

The integration has to be enabled in `metaxy.toml`:

```toml
[sqlalchemy]
enabled = true
```

See [API reference](../reference/api/ext/sqlalchemy.md) for the exact set of provided functions.

## Configuration Options

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.sqlalchemy.SQLAlchemyConfig
    path_prefix: ext.sqlalchemy
    header_level: 3
<!-- dprint-ignore-end -->

## Alembic Integration

[Alembic](https://alembic.sqlalchemy.org/en/latest/) is a database migration toolkit for SQLAlchemy.

The two helper functions: [`get_feature_sqla_metadata_for_store`][metaxy.ext.sqlalchemy.get_feature_sqla_metadata_for_store] and [`get_system_sqla_metadata_for_store`][metaxy.ext.sqlalchemy.get_system_sqla_metadata_for_store] can be used to retrieve an SQLAlchemy URL and `MetaData` object for a given [`metaxy.MetadataStore`][metaxy.MetadataStore] (given it's backed by an SQL database).

`get_feature_sqla_metadata_for_store` returns table metadata for the user-defined tables, while `get_system_sqla_metadata_for_store` returns metadata for Metaxy's system tables.

Here is an example Alembic `env.py` that uses the Metaxy SQLAlchemy integration:

```python title="env.py"
from alembic import context
from metaxy import init_metaxy
from metaxy.ext.sqlalchemy import get_feature_sqla_metadata_for_store

# Alembic Config object
config = context.config

# Load features and get URL + metadata from store
init_metaxy()
url, target_metadata = get_feature_sqla_metadata_for_store("my_store")

# Configure Alembic
config.set_main_option("sqlalchemy.url", url)


# continue with the standard Alembic workflow
```

## Multi-Store Setup

You can configure separate metadata stores for different environments:

```toml title="metaxy.toml"
[stores.dev]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
config = { database = "dev_metadata.duckdb" }

[stores.prod]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
config = { database = "prod_metadata.duckdb" }
```

Then create multiple Alembic directories and register them with Alembic:

```ini title="alembic.ini"
[dev]
script_location = alembic/dev

[prod]
script_location = alembic/prod
```

The two environments now can be managed independently:

=== "dev"

    ```python title="alembic/dev/env.py" hl_lines="2"
    url, target_metadata = get_feature_sqla_metadata_for_store(
        store_name="dev"
    )
    ```

    The `-n` argument can be used to specify the target Alembic directory:

    ```bash
    alembic -n dev upgrade head
    ```

=== "prod"

    ```python title="alembic/prod/env.py"  hl_lines="2"
    url, target_metadata = get_feature_sqla_metadata_for_store(
        store_name="prod"
    )
    ```

    The `-n` argument can be used to specify the target Alembic directory:

    ```bash
    alembic -n prod upgrade head
    ```

# Reference

- [API docs](../reference/api/ext/sqlalchemy.md)
