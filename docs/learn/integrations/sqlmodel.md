# SQLModel Integration

The [SQLModel](https://sqlmodel.tiangolo.com/) integration enables Metaxy features to function as both metadata-tracked features and SQLAlchemy ORM models.

This integration combines Metaxy's versioning and dependency tracking with SQLModel's database mapping and query capabilities.

It is the primary way to use Metaxy with database-backed [metadata stores](../../learn/metadata-stores.md). The benefits of using SQLModel are mostly in the ability to use migration systems such as [Alembic](https://alembic.sqlalchemy.org/) that can ensure schema consistency with Metaxy features, and provide the tools for schema evolution as the features change over time.

## Installation

The SQLModel integration requires the sqlmodel package:

```bash
pip install metaxy[sqlmodel]
```

## Basic Usage

The integration has to be enabled in the configuration file:

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
    export METAXY_EXT_SQLMODEL_ENABLE=true
    ```

This will expose Metaxy's system tables to SQLAlchemy.

First, as always with Metaxy features, we would have to define our ID columns:

```python
from metaxy import BaseFeatureSpec
from metaxy.ext.sqlmodel import BaseSQLModelFeature


class SampleFeatureSpec(BaseFeatureSpec):
    id_columns: tuple[str] = "sample_id"


class SampleFeature(BaseSQLModelFeature, table=False, spec=None):
    sample_id: str
```

Note that ID columns **cannot be server-generated**, so the cannot include a Primary Key.

Now we can define feature class that inherits from `SampleFeature` and specify both Metaxy's `spec` parameter and SQLModel's `table=True` parameter:

```python
from metaxy import FeatureKey, FieldSpec, FieldKey
from sqlmodel import Field


class VideoFeature(
    SampleFeature,
    table=True,
    spec=SampleFeatureSpec(
        key=FeatureKey(["video"]),
        # Root feature with no dependencies
        fields=[
            FieldSpec(key=FieldKey(["frames"]), code_version="1"),
            FieldSpec(key=FieldKey(["duration"]), code_version="1"),
        ],
    ),
):
    # User-defined metadata columns
    path: str
    duration: float
```

This class serves dual purposes:

- **Metaxy feature**: Tracks feature version, field versions, and dependencies
- **SQLModel table**: Maps to database schema with ORM functionality

!!! note "Automatic Table Naming"
When `__tablename__` is not specified, it is automatically generated from the feature key. For `FeatureKey(["video"])`, the table name becomes `"video"`. For `FeatureKey(["video", "processing"])`, it becomes `"video__processing"`. This behavior can be disabled in Metaxy's configuration.

### System-Managed Columns

Metaxy's metadata store automatically manages versioning columns:

- `data_version`: Struct column mapping field keys to hashes
- `feature_version`: Hash of feature specification
- `snapshot_version`: Hash of entire graph state

These columns need not be defined in your SQLModel class. The metadata store injects them during write and read operations.

### ID Columns

!!! warning "ID columns must exist before database insertion"
ID columns are used for joins between features, so their values must exist before insertion into the database. This means you cannot use server-generated values (autoincrement, sequences, server_default) for ID columns.

    Metaxy validates against autoincrement primary keys but cannot detect all server-generated patterns. Ensure your ID columns use client-provided values.

Example:

```python
# ✅ Good: Client-generated ID columns
class UserActivity(
    SQLModelFeature,
    table=True,
    spec=FeatureSpec(
        key=FeatureKey(["user", "activity"]),
        id_columns=["user_id", "session_id"],  # Client provides these
        ...
    ),
):
    user_id: str = Field(primary_key=True)  # Client-generated
    session_id: str = Field(primary_key=True)  # Client-generated
    created_at: str = Field(sa_column_kwargs={"server_default": "NOW()"})  # OK - not an ID column

# ❌ Bad: Autoincrement ID column
class BadFeature(
    SQLModelFeature,
    table=True,
    spec=FeatureSpec(
        key=FeatureKey(["bad"]),
        id_columns=["id"],  # This is listed as an ID column
        ...
    ),
):
    id: int = Field(primary_key=True, sa_column_kwargs={"autoincrement": True})  # Will raise error
```

### Loading Features and Populating Metadata

When using [metaxy.init_metaxy][metaxy.init_metaxy] to discover and import feature modules, all `SQLModelFeature` classes are automatically registered in SQLModel's metadata:

```python
from metaxy import init_metaxy, init_metaxy
from sqlmodel import SQLModel

# Load all features from configured entrypoints
graph = init_metaxy()

# All SQLModelFeature tables are now registered in SQLModel.metadata
# This metadata can be used with Alembic for migrations
print(f"Tables registered: {list(SQLModel.metadata.tables.keys())}")
```

This is particularly useful when:

- Generating Alembic migrations that need to discover all tables
- Setting up database connections that require the complete schema
- Using SQLModel's `create_all()` for development/testing (Metaxy's `auto_create_tables` setting should be preferred over `create_all()`)

!!! tip "Migration Generation"
After calling `init_metaxy`, you can use [Alembic](#database-migrations-with-alembic) to automatically detect all your SQLModelFeature tables and generate migration scripts.

## Configuration

Configure automatic table naming behavior:

=== "metaxy.toml"

    ```toml
    [ext.sqlmodel]
    enable = true
    infer_db_table_names = true  # Default
    ```

=== "pyproject.toml"

    ```toml
    [tool.metaxy.ext.sqlmodel]
    enable = true
    infer_db_table_names = true  # Default
    ```

=== "Environment Variable"

    ```bash
    export METAXY_EXT_SQLMODEL_INFER_DB_TABLE_NAMES=true
    ```

## Database Migrations with Alembic

Metaxy provides SQLModel definitions for its system tables that integrate with [Alembic](https://alembic.sqlalchemy.org/) for database migrations. This allows you to version control schema changes alongside your application code. Note that you might want to keep separate migrations per each DB-backed `MetadataStore` used with Metaxy.

### Separate Migration Management

Metaxy system tables and user application tables should be managed in separate Alembic migration directories. This separation provides critical safety guarantees:

**System Table Isolation**: Metaxy system tables (`metaxy-system__feature_versions`, `metaxy-system__migration_events`) have schemas managed by the framework. User migrations cannot accidentally modify these internal structures.

**Independent Evolution**: Metaxy can evolve its system table schemas independently through framework updates without conflicts with user migrations.

**Failure Isolation**: User migration failures remain isolated from metaxy's internal state tracking. A failed user migration leaves system tables intact for debugging and recovery.

**Clear Audit Trail**: Separate migration histories make it trivial to distinguish framework schema changes from application schema changes. This clarity is essential during rollbacks and incident investigation.

### Setup

Enable SQLModel system tables in your metaxy configuration and set up two Alembic directories:

```bash
# Standard structure
project/
├── alembic/              # User application migrations
│   ├── versions/
│   └── env.py
├── .metaxy/
│   └── alembic-system/   # Metaxy system table migrations
│       ├── versions/
│       └── env.py
└── metaxy.toml
```

Initialize both Alembic directories:

```bash
# Initialize user migrations
alembic init alembic

# Initialize metaxy system migrations
alembic init .metaxy/alembic-system
```

### Metaxy System Tables Configuration

Configure `.metaxy/alembic-system/env.py` to manage only metaxy system tables:

```python
# typical Alembic boilerplate
from metaxy.ext.alembic import get_metaxy_metadata

metaxy_system_metadata = get_metaxy_metadata()

# metaxy_system_metadata has system tables

# continue with alembic boilerplate
```

Configure `.metaxy/alembic-system/alembic.ini` with your database URL:

```ini
[alembic]
script_location = .metaxy/alembic-system
```

### User Application Tables Configuration

Configure `alembic/env.py` to manage user tables, excluding metaxy system tables:

```python
# standard Alembic boilerplate
from sqlmodel import SQLModel
from metaxy import init_metaxy

init_metaxy()

# SQLModel.metadata now has user-defined Metaxy tables


# continue with alembic boilerplate
```

### Migration Workflow

Generate and apply migrations separately for each concern:

```bash
# 1. Create metaxy system tables (run once during initial setup)
alembic -c .metaxy/alembic-system/alembic.ini revision --autogenerate -m "create metaxy system tables"
alembic -c .metaxy/alembic-system/alembic.ini upgrade head

# 2. Create and apply user table migrations
alembic revision --autogenerate -m "add video feature table"
alembic upgrade head

# 3. When modifying user tables, only user migrations change
alembic revision --autogenerate -m "add processing timestamp"
alembic upgrade head
```

When deploying to production, always apply system table migrations before user migrations:

```bash
# Production deployment order
alembic -c .metaxy/alembic-system/alembic.ini upgrade head  # System tables first
alembic upgrade head                                 # Then user tables
```

### Disabling SQLModel System Tables

If required, disable SQLModel system tables in `metaxy.toml`:

```toml
[ext.sqlmodel]
enabled = true
system_tables = false
```
