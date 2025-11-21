# Alembic Integration

Metaxy provides helpers for integrating with [Alembic](https://alembic.sqlalchemy.org/) database migrations.

## Installation

Install Metaxy and Alembic:

```bash
pip install metaxy alembic
```

## Helper Functions

### `get_metaxy_system_metadata`

Get metadata for Metaxy system tables:

```python
from metaxy.ext.sqlalchemy import get_metaxy_system_metadata

# In your .metaxy/alembic-system/env.py
target_metadata = get_metaxy_system_metadata()
```

### `get_features_sqlalchemy_metadata`

Get metadata for user-defined SQLModel feature tables, with optional project filtering:

```python
from metaxy.ext.sqlalchemy import get_features_sqlalchemy_metadata
from metaxy import init_metaxy

# In your alembic/env.py
init_metaxy()  # Load all features

# Get metadata for current project only (default)
target_metadata = get_features_sqlalchemy_metadata()

# Get metadata for all projects
target_metadata = get_features_sqlalchemy_metadata(filter_by_project=False)
```

> **Note**: This function is specific to SQLModel-based features. For non-SQLModel features,
> you'll need to manage your own metadata.

### Database Connection

Get the SQLAlchemy connection URL from a configured [`MetadataStore`][metaxy.MetadataStore]:

```python
from metaxy.ext.sqlalchemy import get_store_sqlalchemy_url

# Get URL from default store
url = get_store_sqlalchemy_url()

# Get URL from named store
url = get_store_sqlalchemy_url("prod")

# Use in alembic/env.py
config.set_main_option("sqlalchemy.url", url)
```

## Example: Alembic's `env.py`

Complete example for user feature migrations:

```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

from metaxy import init_metaxy
from metaxy.ext.sqlalchemy import get_features_sqlalchemy_metadata
from metaxy.ext.sqlalchemy import get_store_sqlalchemy_url

# Alembic Config object
config = context.config

# Interpret the config file for logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Load features and get metadata
init_metaxy()
target_metadata = get_features_sqlalchemy_metadata()

# Get database URL from metadata store
config.set_main_option("sqlalchemy.url", get_store_sqlalchemy_url("my-store"))
```

## Multi-Store Setup

Configure separate Alembic stores for different environments:

```toml
[stores.dev]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
config = { database = "dev_metadata.duckdb" }

[stores.prod]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
config = { database = "prod_metadata.duckdb" }
```

Then create multiple Alembic migration sets:

```ini
[dev]
alembic_dir = "alembic/dev"

[prod]
alembic_dir = "alembic/prod"
```

Each should have a `env.py` file.

Use the `-n` argument to specify the environment:

```
alembic -n dev upgrade head
alembic -n prod upgrade head
``
```
