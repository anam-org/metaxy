"""SQLAlchemy integration plugin for metaxy.

This module provides SQLAlchemy Table definitions and helpers for metaxy system tables
and user-defined feature tables. These can be used with migration tools like Alembic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, Index, MetaData, String, Table

from metaxy.config import MetaxyConfig
from metaxy.ext.sqlalchemy.config import SQLAlchemyConfig
from metaxy.metadata_store.system import EVENTS_KEY, FEATURE_VERSIONS_KEY
from metaxy.models.constants import (
    METAXY_FEATURE_SPEC_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_FULL_DEFINITION_VERSION,
    METAXY_SNAPSHOT_VERSION,
)
from metaxy.models.feature_spec import FeatureSpec

if TYPE_CHECKING:
    from metaxy.metadata_store.ibis import IbisMetadataStore


# System Tables


def create_system_tables(
    metadata: MetaData,
    table_prefix: str = "",
) -> tuple[Table, Table]:
    """Create system table definitions in the given metadata.

    System tables always include primary key constraints since they are controlled by metaxy.

    Args:
        metadata: SQLAlchemy MetaData object to add tables to
        table_prefix: Optional prefix to prepend to table names (e.g., "dev_")

    Returns:
        Tuple of (feature_versions_table, events_table)
    """
    feature_versions_name = (
        f"{table_prefix}{FEATURE_VERSIONS_KEY.table_name}"
        if table_prefix
        else FEATURE_VERSIONS_KEY.table_name
    )

    feature_versions_table = Table(
        feature_versions_name,
        metadata,
        # Composite primary key
        Column("project", String, primary_key=True, index=True),
        Column("feature_key", String, primary_key=True, index=True),
        Column(
            METAXY_FEATURE_SPEC_VERSION,
            String,
            primary_key=True,
        ),
        # Versioning columns
        Column(METAXY_FEATURE_VERSION, String, index=True),
        Column(METAXY_FULL_DEFINITION_VERSION, String, index=True),
        Column(METAXY_SNAPSHOT_VERSION, String, index=True),
        # Metadata columns
        Column("recorded_at", DateTime, index=True),
        Column("feature_schema", String),  # JSON string
        Column("tags", String, default="{}"),  # JSON string
        # Additional indexes
        Index(
            f"idx_{feature_versions_name}_lookup",
            "project",
            "feature_key",
            METAXY_FEATURE_VERSION,
        ),
    )

    events_name = (
        f"{table_prefix}{EVENTS_KEY.table_name}"
        if table_prefix
        else EVENTS_KEY.table_name
    )

    events_table = Table(
        events_name,
        metadata,
        # Composite primary key matching Polars append-only storage
        Column("project", String, primary_key=True, index=True),
        Column("execution_id", String, primary_key=True, index=True),
        Column("timestamp", DateTime, primary_key=True),
        # Event fields
        Column("event_type", String, index=True),
        Column("feature_key", String, nullable=True, index=True),
        Column("payload", String, default=""),  # JSON string
        # Additional indexes
        Index(
            f"idx_{events_name}_lookup",
            "project",
            "execution_id",
            "event_type",
        ),
    )

    return feature_versions_table, events_table


def _get_store_sqlalchemy_url(store: IbisMetadataStore) -> str:
    """Get SQLAlchemy URL from an IbisMetadataStore instance.

    Args:
        store: IbisMetadataStore instance

    Returns:
        SQLAlchemy connection URL string

    Raises:
        ValueError: If sqlalchemy_url is empty
    """
    if not store.sqlalchemy_url:
        raise ValueError("IbisMetadataStore has an empty `sqlalchemy_url`.")

    return store.sqlalchemy_url


def _get_system_metadata(
    table_prefix: str = "",
) -> MetaData:
    """Create SQLAlchemy metadata containing system tables.

    System tables always include primary key constraints.

    Args:
        table_prefix: Optional prefix to prepend to table names

    Returns:
        MetaData containing system table definitions
    """
    metadata = MetaData()
    create_system_tables(metadata, table_prefix=table_prefix)
    return metadata


def get_system_slqa_metadata(
    store: IbisMetadataStore,
) -> tuple[str, MetaData]:
    """Get SQLAlchemy URL and Metaxy system tables metadata for a metadata store.

    This function retrieves both the connection URL and system table metadata
    for a store, with the store's `table_prefix` automatically applied to table names.

    Args:
        store: IbisMetadataStore instance

    Returns:
        Tuple of (sqlalchemy_url, system_metadata)

    Raises:
        ValueError: If store's sqlalchemy_url is empty

    Example:

        ```py
        from metaxy.ext.sqlalchemy import get_system_slqa_metadata
        from metaxy.config import MetaxyConfig

        # Get the store instance
        config = MetaxyConfig.get()
        store = config.get_store()

        # Get URL and metadata
        url, metadata = get_system_slqa_metadata(store)

        # Use with Alembic env.py
        from alembic import context
        url, target_metadata = get_system_slqa_metadata(store)
        context.configure(url=url, target_metadata=target_metadata)
        ```
    """
    url = _get_store_sqlalchemy_url(store)
    metadata = _get_system_metadata(table_prefix=store._table_prefix)
    return url, metadata


def _get_features_metadata(
    source_metadata: MetaData,
    store: IbisMetadataStore,
    project: str | None = None,
    filter_by_project: bool = True,
    inject_primary_key: bool | None = None,
    inject_index: bool | None = None,
) -> MetaData:
    """Filter user-defined feature tables from source metadata by project.

    This function must be called after init_metaxy() to ensure features are loaded.

    Args:
        source_metadata: Source SQLAlchemy MetaData to filter (e.g., SQLModel.metadata)
        store: IbisMetadataStore instance (used to get table_prefix)
        project: Project name to filter by. If None, uses MetaxyConfig.get().project
        filter_by_project: If True, only include features for the specified project.
        inject_primary_key: If True, inject composite primary key constraints.
                           If False, do not inject. If None, uses config default.
        inject_index: If True, inject composite index.
                     If False, do not inject. If None, uses config default.

    Returns:
        Filtered SQLAlchemy MetaData containing only project-scoped feature tables
    """
    from metaxy.models.feature import FeatureGraph

    config = MetaxyConfig.get()

    if project is None:
        project = config.project

    # Check plugin config for defaults
    sqlalchemy_config = config.get_plugin("sqlalchemy", SQLAlchemyConfig)
    if inject_primary_key is None:
        inject_primary_key = sqlalchemy_config.inject_primary_key
    if inject_index is None:
        inject_index = sqlalchemy_config.inject_index

    # Get table_prefix from store
    table_prefix = store._table_prefix

    # Get the active feature graph
    graph = FeatureGraph.get_active()

    # Compute expected table names for features in the project
    expected_table_names = set()
    feature_specs_by_table_name = {}

    for feature_key, feature_cls in graph.features_by_key.items():
        # Filter by project if requested
        if filter_by_project and hasattr(feature_cls, "project"):
            feature_project = getattr(feature_cls, "project")
            if feature_project != project:
                continue

        # Compute table name using same logic as IbisMetadataStore.get_table_name
        base_name = feature_key.table_name
        table_name = f"{table_prefix}{base_name}" if table_prefix else base_name

        expected_table_names.add(table_name)
        feature_specs_by_table_name[table_name] = feature_cls.spec()

    # Filter source metadata to only include expected tables
    filtered_metadata = MetaData()

    for table_name, table in source_metadata.tables.items():
        if table_name in expected_table_names:
            # Copy table to filtered metadata
            new_table = table.to_metadata(filtered_metadata)

            # Inject constraints if requested
            spec = feature_specs_by_table_name[table_name]
            _inject_constraints(
                table=new_table,
                spec=spec,
                inject_primary_key=inject_primary_key,
                inject_index=inject_index,
            )

    return filtered_metadata


def _inject_constraints(
    table: Table,
    spec: FeatureSpec,
    inject_primary_key: bool,
    inject_index: bool,
) -> None:
    """Inject primary key and/or index constraints on a table.

    Args:
        table: SQLAlchemy Table to modify
        spec: Feature specification with id_columns
        inject_primary_key: If True, inject composite primary key
        inject_index: If True, inject composite index
    """
    from sqlalchemy import PrimaryKeyConstraint

    from metaxy.models.constants import METAXY_CREATED_AT, METAXY_DATA_VERSION

    # Composite key/index columns: id_columns + metaxy_created_at + metaxy_data_version
    key_columns = list(spec.id_columns) + [METAXY_CREATED_AT, METAXY_DATA_VERSION]

    if inject_primary_key:
        # Add primary key constraint
        pk_constraint = PrimaryKeyConstraint(*key_columns, name="pk_metaxy_composite")
        table.append_constraint(pk_constraint)

    if inject_index:
        # Add composite index
        idx = Index(
            f"idx_{table.name}_metaxy_composite",
            *key_columns,
        )
        table.append_constraint(idx)


def filter_feature_sqla_metadata(
    store: IbisMetadataStore,
    source_metadata: MetaData | None = None,
    project: str | None = None,
    filter_by_project: bool = True,
    inject_primary_key: bool | None = None,
    inject_index: bool | None = None,
) -> tuple[str, MetaData]:
    """Get SQLAlchemy URL and feature table metadata for a metadata store.

    This function filters the source metadata to include only feature tables
    belonging to the specified project, and returns the connection URL for the store.

    This function must be called after init_metaxy() to ensure features are loaded.

    Args:
        store: IbisMetadataStore instance
        source_metadata: Source SQLAlchemy MetaData to filter. If None, uses SQLModel.metadata
                        (requires SQLModel to be installed).
        project: Project name to filter by. If None, uses MetaxyConfig.get().project
        filter_by_project: If True, only include features for the specified project.
                          If False, include all features.
        inject_primary_key: If True, inject composite primary key constraints.
                           If False, do not inject. If None, uses config default.
        inject_index: If True, inject composite index.
                     If False, do not inject. If None, uses config default.

    Returns:
        Tuple of (sqlalchemy_url, filtered_metadata)

    Raises:
        ValueError: If store's sqlalchemy_url is empty
        ImportError: If source_metadata is None and SQLModel is not installed

    Example:

        ```py
        from metaxy.ext.sqlalchemy import filter_feature_sqla_metadata
        from metaxy import init_metaxy
        from metaxy.config import MetaxyConfig

        # Load features first
        init_metaxy()

        # Get store instance
        config = MetaxyConfig.get()
        store = config.get_store()

        # With SQLModel (default)
        url, metadata = filter_feature_sqla_metadata(store)

        # With custom metadata
        from sqlalchemy import MetaData
        my_metadata = MetaData()
        # ... define tables in my_metadata ...
        url, metadata = filter_feature_sqla_metadata(store, source_metadata=my_metadata)

        # Use with Alembic env.py
        from alembic import context
        url, target_metadata = filter_feature_sqla_metadata(store)
        context.configure(url=url, target_metadata=target_metadata)
        ```
    """
    # Default to SQLModel.metadata if not provided
    if source_metadata is None:
        try:
            from sqlmodel import SQLModel

            source_metadata = SQLModel.metadata
        except ImportError as e:
            raise ImportError(
                "source_metadata is required when SQLModel is not installed. "
                "Either install SQLModel or provide a MetaData object explicitly."
            ) from e

    url = _get_store_sqlalchemy_url(store)
    metadata = _get_features_metadata(
        source_metadata=source_metadata,
        store=store,
        project=project,
        filter_by_project=filter_by_project,
        inject_primary_key=inject_primary_key,
        inject_index=inject_index,
    )
    return url, metadata
