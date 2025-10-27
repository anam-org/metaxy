"""Alembic integration helpers for metaxy.

This module provides utilities for including metaxy system tables
in Alembic migrations when using SQLModel integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy import MetaData


def get_metaxy_metadata() -> MetaData:
    """Get SQLAlchemy metadata containing metaxy system tables.

    This function returns the metadata object that should be included
    in your Alembic configuration to manage metaxy system tables.

    Returns:
        SQLAlchemy MetaData containing metaxy system table definitions

    Raises:
        ImportError: If SQLModel is not installed

    Example:
        >>> # In your alembic/env.py:
        >>> from metaxy.ext.alembic import get_metaxy_metadata
        >>> from sqlalchemy import MetaData
        >>>
        >>> # Combine your app metadata with metaxy metadata
        >>> target_metadata = MetaData()
        >>> target_metadata.reflect(get_metaxy_metadata())
        >>> target_metadata.reflect(my_app.metadata)
    """
    from metaxy.ext.sqlmodel_system_tables import get_system_metadata

    return get_system_metadata()


def include_metaxy_tables(target_metadata: MetaData) -> MetaData:
    """Include metaxy system tables in existing metadata.

    This is a convenience function that adds metaxy system tables
    to your existing SQLAlchemy metadata object.

    Args:
        target_metadata: Your application's metadata object

    Returns:
        The same metadata object with metaxy tables added

    Example:
        >>> # In your alembic/env.py:
        >>> from metaxy.ext.alembic import include_metaxy_tables
        >>> from myapp.models import metadata
        >>>
        >>> # Add metaxy tables to your metadata
        >>> target_metadata = include_metaxy_tables(metadata)
    """
    metaxy_metadata = get_metaxy_metadata()

    # Copy tables from metaxy metadata to target metadata
    for table_name, table in metaxy_metadata.tables.items():
        if table_name not in target_metadata.tables:
            # Create a new table with the same definition in target metadata
            table.to_metadata(target_metadata)

    return target_metadata


def check_sqlmodel_enabled() -> bool:
    """Check if SQLModel integration is enabled in the configuration.

    Returns:
        True if SQLModel integration is enabled, False otherwise

    Example:
        >>> from metaxy.ext.alembic import check_sqlmodel_enabled
        >>> if check_sqlmodel_enabled():
        ...     # SQLModel is enabled, include metaxy tables
        ...     include_metaxy_tables(metadata)
    """
    try:
        from metaxy.config import MetaxyConfig

        config = MetaxyConfig.get()
        # Check if SQLModel plugin is enabled in the plugins list
        return "sqlmodel" in config.plugins
    except Exception:
        return False


def generate_alembic_env_template() -> str:
    """Generate a template for Alembic env.py with metaxy integration.

    Returns:
        String containing example Alembic env.py configuration

    Example:
        >>> from metaxy.ext.alembic import generate_alembic_env_template
        >>> template = generate_alembic_env_template()
        >>> print(template)  # Shows example configuration
    """
    return '''"""Alembic env.py template with metaxy integration.

This template shows how to include metaxy system tables in your
Alembic migrations when using SQLModel integration.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Import your models here
# from myapp import models

# Metaxy integration
from metaxy.ext.alembic import get_metaxy_metadata, include_metaxy_tables
from metaxy.config import MetaxyConfig

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Load metaxy configuration
metaxy_config = MetaxyConfig.load()

# Add your model's MetaData object here for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata

# Option 1: Include metaxy tables in your existing metadata
# from myapp.models import metadata
# target_metadata = include_metaxy_tables(metadata)

# Option 2: Combine multiple metadata objects
from sqlalchemy import MetaData
target_metadata = MetaData()

# Add metaxy system tables (if using SQLModel integration)
if "sqlmodel" in metaxy_config.plugins:
    metaxy_metadata = get_metaxy_metadata()
    for table_name, table in metaxy_metadata.tables.items():
        table.to_metadata(target_metadata)

# Add your application tables
# for table_name, table in myapp_metadata.tables.items():
#     table.to_metadata(target_metadata)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
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
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''


def create_initial_migration_script() -> str:
    """Generate an Alembic migration script for creating metaxy system tables.

    Returns:
        String containing example migration script

    Example:
        >>> from metaxy.ext.alembic import create_initial_migration_script
        >>> script = create_initial_migration_script()
        >>> # Save to alembic/versions/001_create_metaxy_tables.py
    """
    return '''"""Create metaxy system tables.

Revision ID: create_metaxy_system_tables
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'create_metaxy_system_tables'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create metaxy system tables."""

    # Create feature_versions table
    op.create_table(
        'metaxy-system__feature_versions',
        sa.Column('feature_key', sa.String(), nullable=False),
        sa.Column('snapshot_version', sa.String(), nullable=False),
        sa.Column('feature_version', sa.String(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(), nullable=False),
        sa.Column('feature_spec', sa.String(), nullable=False),
        sa.Column('feature_class_path', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('feature_key', 'snapshot_version')
    )
    op.create_index('idx_feature_versions_recorded', 'metaxy-system__feature_versions', ['recorded_at'])
    op.create_index('idx_feature_versions_lookup', 'metaxy-system__feature_versions', ['feature_key', 'feature_version'])

    # Create migration_events table
    op.create_table(
        'metaxy-system__migration_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('migration_id', sa.String(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('feature_key', sa.String(), nullable=False),
        sa.Column('rows_affected', sa.Integer(), nullable=False),
        sa.Column('error_message', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_migration_events_lookup', 'metaxy-system__migration_events', ['migration_id', 'event_type'])
    op.create_index('idx_migration_events_feature', 'metaxy-system__migration_events', ['migration_id', 'feature_key'])
    op.create_index(op.f('ix_metaxy-system__migration_events_migration_id'), 'metaxy-system__migration_events', ['migration_id'])
    op.create_index(op.f('ix_metaxy-system__migration_events_timestamp'), 'metaxy-system__migration_events', ['timestamp'])


def downgrade() -> None:
    """Drop metaxy system tables."""
    op.drop_index(op.f('ix_metaxy-system__migration_events_timestamp'), table_name='metaxy-system__migration_events')
    op.drop_index(op.f('ix_metaxy-system__migration_events_migration_id'), table_name='metaxy-system__migration_events')
    op.drop_index('idx_migration_events_feature', table_name='metaxy-system__migration_events')
    op.drop_index('idx_migration_events_lookup', table_name='metaxy-system__migration_events')
    op.drop_table('metaxy-system__migration_events')

    op.drop_index('idx_feature_versions_lookup', table_name='metaxy-system__feature_versions')
    op.drop_index('idx_feature_versions_recorded', table_name='metaxy-system__feature_versions')
    op.drop_table('metaxy-system__feature_versions')
'''


def get_table_creation_sql(dialect: str = "postgresql") -> dict[str, str]:
    """Get SQL statements for creating metaxy system tables.

    Args:
        dialect: SQL dialect ('postgresql', 'mysql', 'sqlite', etc.)

    Returns:
        Dictionary mapping table name to CREATE TABLE SQL statement

    Example:
        >>> from metaxy.ext.alembic import get_table_creation_sql
        >>> sql_statements = get_table_creation_sql("postgresql")
        >>> print(sql_statements["feature_versions"])
    """
    from sqlalchemy import create_engine
    from sqlalchemy.schema import CreateTable

    # Get metaxy metadata
    metadata = get_metaxy_metadata()

    # Create a mock engine for the specified dialect
    if dialect == "postgresql":
        engine_url = "postgresql:///"
    elif dialect == "mysql":
        engine_url = "mysql:///"
    elif dialect == "sqlite":
        engine_url = "sqlite:///"
    else:
        engine_url = f"{dialect}:///"

    engine = create_engine(engine_url)

    # Generate CREATE TABLE statements
    sql_statements = {}
    for table_name, table in metadata.tables.items():
        create_statement = CreateTable(table).compile(engine)
        sql_statements[table_name] = str(create_statement)

    return sql_statements
