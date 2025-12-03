# Metaxy + SQLModel

The [SQLModel](https://sqlmodel.tiangolo.com/) integration enables Metaxy features to also act as [SQLAlchemy](https://www.sqlalchemy.org/) ORM models. It exposes user-defined feature tables to SQLAlchemy.

It is the primary way to use Metaxy with database-backed [metadata stores](../../../learn/metadata-stores.md).

!!! tip "Database Migrations"

    For database migration management with Alembic, see the [SQLAlchemy integration guide](../sqlalchemy/index.md#alembic-integration).

## Installation

The SQLModel integration requires the sqlmodel package:

```bash
pip install 'metaxy[sqlmodel]'
```

and has to be enabled explicitly:

=== "metaxy.toml"

    ```toml
    [ext.sqlmodel]
    enable = true
    ```

=== "pyproject.toml"

    ```toml
    [tool.metaxy.ext.sqlmodel]
    enable = true
    ```

=== "Environment Variable"

    ```bash
    export METAXY_EXT__SQLMODEL_ENABLE=true
    ```

## Usage

The SQLModel integration provides [`BaseSQLModelFeature`][metaxy.ext.sqlmodel.BaseSQLModelFeature] which combines the functionality of a Metaxy feature and an SQLModel table.

```python
import metaxy as mx
import metaxy.ext.sqlmodel as mxsql
from sqlmodel import Field


class VideoFeature(
    mxsql.BaseSQLModelFeature,
    table=True,
    spec=mx.FeatureSpec(
        key=FeatureKey(["video"]),
        id_columns=["video_id"],
        fields=[
            "frames",
            "duration",
        ],
    ),
):
    # User-defined metadata columns
    video_id: str
    path: str
    duration: float
```

!!! warning "Do Not Use Server-Generated IDs"

    ID columns **should not be server-generated** because they are typically used to determine **data** locations such as object storage keys, so they have to be defined before **metadata** is inserted into the database

!!! note "Automatic Table Naming"

    When `__tablename__` is not specified, it is automatically generated from the feature key. For `FeatureKey(["video", "processing"])`, it becomes `"video__processing"`. This behavior can be disabled in the plugin configuration.

## Database Migrations

When using SQLModel features with Alembic or other migration tools, use [`filter_feature_sqlmodel_metadata()`][metaxy.ext.sqlmodel.filter_feature_sqlmodel_metadata] to transform table names and filter metadata.

!!! info "Table Name Transformation"

    Pass `SQLModel.metadata` to `filter_feature_sqlmodel_metadata()` and it will transform table names by adding the store's `table_prefix`. The returned metadata will have prefixed table names that match the actual database tables.

```python
from sqlmodel import SQLModel
from metaxy.ext.sqlmodel import filter_feature_sqlmodel_metadata
from metaxy.config import MetaxyConfig
from metaxy import init_metaxy

init_metaxy()
config = MetaxyConfig.get()
store = config.get_store()

# Transform SQLModel metadata with table_prefix
url, target_metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

# Use with Alembic env.py
from alembic import context

context.configure(url=url, target_metadata=target_metadata)
```

The `filter_feature_sqlmodel_metadata()` function:

- Transforms table names by adding the store's `table_prefix`
- Filters tables by project (configurable)
- Returns the SQLAlchemy URL for the store
- Optionally injects primary key and index constraints

See the [SQLAlchemy integration guide](../sqlalchemy/index.md#alembic-integration) for complete Alembic setup examples.

!!! tip "Separate Alembic Version Tables"

    When managing both system tables and feature tables with Alembic, use separate version tables to avoid conflicts. See the [Multi-Store Setup](../sqlalchemy/index.md#multi-store-setup) section for configuration details.

## Reference

- [Configuration](configuration.md)
- [API](api.md)
