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

## Alembic

[Alembic](https://alembic.sqlalchemy.org/en/latest/) is a database migration toolkit for SQLAlchemy.
Here is an example Alembic `env.py` that makes use of the Metaxy integration with SQLAlchemy:

```python title="env.py"
from alembic import context

from metaxy import init_metaxy
from metaxy.ext.sqlalchemy import (
    get_features_sqlalchemy_metadata,
    get_store_sqlalchemy_url,
)

# Alembic Config object
config = context.config

# Load features and get metadata
init_metaxy()
target_metadata = get_features_sqlalchemy_metadata()

# Get database URL from metadata store
config.set_main_option("sqlalchemy.url", get_store_sqlalchemy_url("my-store"))
```

## Multi-Store Setup

Configure separate Alembic stores for different environments:

```toml title="metaxy.toml"
[stores.dev]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
config = { database = "dev_metadata.duckdb" }

[stores.prod]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
config = { database = "prod_metadata.duckdb" }
```

Then create multiple Alembic directories:

```ini title="alembic.ini"
[dev]
alembic_dir = "alembic/dev"

[prod]
alembic_dir = "alembic/prod"
```

Each should have a `env.py` file.

Use the `-n` argument to specify the environment:

```bash {linenums="0"}
alembic -n dev upgrade head
alembic -n prod upgrade head
```

# Reference

- [API docs](../reference/api/ext/sqlalchemy.md)
