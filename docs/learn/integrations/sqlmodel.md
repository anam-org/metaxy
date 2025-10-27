# SQLModel Integration

The SQLModel integration enables Metaxy features to function as both metadata-tracked features and SQLAlchemy ORM models. This integration combines Metaxy's versioning and dependency tracking with SQLModel's database mapping and query capabilities.

## Problem Statement

Multimodal ML pipelines often require persistent storage of metadata alongside versioned feature definitions. While Metaxy tracks feature versions and data lineage, SQLModel provides ORM functionality for database interactions. The integration eliminates the need to maintain separate classes for feature definitions and database models, ensuring consistency between logical feature structure and physical storage schema.

Additionally, users can benefit from migration systems such as [Alembic](https://alembic.sqlalchemy.org/).

## Installation

The SQLModel integration requires the sqlmodel package:

```bash
pip install metaxy[sqlmodel]
```

## Basic Usage

Define a feature class that inherits from `SQLModelFeature` and specify both Metaxy's `spec` parameter and SQLModel's `table=True` parameter:

```python
from metaxy.ext.sqlmodel import SQLModelFeature
from metaxy import FeatureSpec, FeatureKey, FieldSpec, FieldKey
from sqlmodel import Field

class VideoFeature(
    SQLModelFeature,
    table=True,
    spec=FeatureSpec(
        key=FeatureKey(["video"]),
        deps=None,  # Root feature with no dependencies
        fields=[
            FieldSpec(key=FieldKey(["frames"]), code_version=1),
            FieldSpec(key=FieldKey(["duration"]), code_version=1),
        ],
    ),
):
    # Primary key
    uid: str = Field(primary_key=True)

    # Metadata columns
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

## Configuration

Configure automatic table naming behavior:

=== "metaxy.toml"

    ```toml
    [ext.sqlmodel]
    infer_db_table_names = true  # Default
    ```

=== "pyproject.toml"

    ```toml
    [ext.sqlmodel]
    infer_db_table_names = true  # Default
    ```

=== "Environment Variable"

    ```bash
    export METAXY_EXT_SQLMODEL_INFER_DB_TABLE_NAMES=true
    ```

## Database Migrations with Alembic

Metaxy provides SQLModel definitions for its system tables that integrate with [Alembic](https://alembic.sqlalchemy.org/) for database migrations. This allows you to version control schema changes alongside your application code.

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
├── alembic_metaxy/       # Metaxy system table migrations
│   ├── versions/
│   └── env.py
└── metaxy.toml
```

Initialize both Alembic directories:

```bash
# Initialize user migrations
alembic init alembic

# Initialize metaxy system migrations
alembic init alembic_metaxy
```

### Metaxy System Tables Configuration

Configure `alembic_metaxy/env.py` to manage only metaxy system tables:

```python
"""Alembic environment for metaxy system tables only."""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import metaxy helpers
from metaxy.ext.alembic import get_metaxy_metadata

# Alembic Config object
config = context.config

# Configure logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Get metaxy system tables metadata ONLY
target_metadata = get_metaxy_metadata()


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Configure `alembic_metaxy/alembic.ini` with your database URL:

```ini
[alembic]
script_location = alembic_metaxy
sqlalchemy.url = postgresql://user:pass@localhost/dbname
```

### User Application Tables Configuration

Configure `alembic/env.py` to manage user tables, excluding metaxy system tables:

```python
"""Alembic environment for user application tables."""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import your application models
from myapp.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use your application's metadata
# This should NOT include metaxy system tables
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Exclude metaxy system tables
        include_schemas=False,
        include_object=lambda obj, name, type_, reflected, compare_to: (
            not name.startswith("metaxy-system__")
        ),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Exclude metaxy system tables
            include_object=lambda obj, name, type_, reflected, compare_to: (
                not name.startswith("metaxy-system__")
            ),
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Migration Workflow

Generate and apply migrations separately for each concern:

```bash
# 1. Create metaxy system tables (run once during initial setup)
alembic -c alembic_metaxy/alembic.ini revision --autogenerate -m "create metaxy system tables"
alembic -c alembic_metaxy/alembic.ini upgrade head

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
alembic -c alembic_metaxy/alembic.ini upgrade head  # System tables first
alembic upgrade head                                 # Then user tables
```

### Enabling SQLModel System Tables

Enable SQLModel system tables in `metaxy.toml`:

```toml
[ext.sqlmodel]
enabled = true          # Enable SQLModel plugin
system_tables = true    # Use SQLModel definitions for system tables
```

Once enabled, metaxy system tables are available to Alembic through the `metaxy.ext.alembic` helpers. The tables appear in your database with the `metaxy-system__` prefix and can be managed through the separate Alembic configuration described above.
