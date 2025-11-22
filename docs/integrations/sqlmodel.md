# Metaxy + SQLModel

The [SQLModel](https://sqlmodel.tiangolo.com/) integration enables Metaxy features to also act as [SQLAlchemy](https://www.sqlalchemy.org/) ORM models. It exposes user-defined feature tables to SQLAlchemy.

It is the primary way to use Metaxy with database-backed [metadata stores](../learn/metadata-stores.md).

!!! tip

    This integration should be used together with the [SQLAlchemy integration](sqlalchemy.md). See [Alembic instructions](sqlalchemy.md#alembic-integration) for more details on how to set up database migrations.

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

## Configuration Options

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.sqlmodel.SQLModelPluginConfig
    path_prefix: ext.sqlmodel
    header_level: 3
<!-- dprint-ignore-end -->

# Reference

- [API docs](../reference/api/ext/sqlmodel.md)
