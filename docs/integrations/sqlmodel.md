# SQLModel Integration

The [SQLModel](https://sqlmodel.tiangolo.com/) integration enables Metaxy features to function as both metadata-tracked features and SQLAlchemy ORM models.

This integration combines Metaxy's versioning and dependency tracking with SQLModel's database mapping and query capabilities.

It is the primary way to use Metaxy with database-backed [metadata stores](../learn/metadata-stores.md). The benefits of using SQLModel are mostly in the ability to use migration systems such as [Alembic](https://alembic.sqlalchemy.org/) that can ensure schema consistency with Metaxy features, and provide the tools for schema evolution as the features change over time.

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

Let's define a feature:

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

!!! tip "Database Migrations Generation"

    You can use Alembic with the [sqlalchemy integration](sqlalchemy.md) to automatically detect feature tables and generate migration scripts.

!!! warning "Do Not Use Server-Generated IDs"

    ID columns **should not be server-generated** because they are typically used to determine **data** locations such as object storage keys, so they have to be defined before **metadata** is inserted into the database

!!! note "Automatic Table Naming"

    When `__tablename__` is not specified, it is automatically generated from the feature key. For `FeatureKey(["video"])`, the table name becomes `"video"`. For `FeatureKey(["video", "processing"])`, it becomes `"video__processing"`. This behavior can be disabled in the plugin configuration.

## Configuration Options

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.sqlmodel.SQLModelPluginConfig
    path_prefix: ext.sqlmodel
    header_level: 3
<!-- dprint-ignore-end -->
