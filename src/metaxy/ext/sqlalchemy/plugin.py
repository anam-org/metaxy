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
        Column("feature_spec", String),  # Full serialized FeatureSpec as JSON string
        Column("feature_schema", String),  # Full Pydantic model schema as JSON string
        Column("feature_class_path", String),  # Python import path to Feature class
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


def _get_store_sqlalchemy_url(
    store: IbisMetadataStore,
    protocol: str | None = None,
    port: int | None = None,
) -> str:
    """Get SQLAlchemy URL from an IbisMetadataStore instance.

    Args:
        store: IbisMetadataStore instance
        protocol: Optional protocol (drivername) to replace the existing one.
            Useful when Ibis uses a different protocol than SQLAlchemy requires.
        port: Optional port to replace the existing one. Useful when the
            SQLAlchemy driver uses a different port than Ibis.

    Returns:
        SQLAlchemy connection URL string

    Raises:
        ValueError: If sqlalchemy_url is empty
    """
    if not store.sqlalchemy_url:
        raise ValueError("IbisMetadataStore has an empty `sqlalchemy_url`.")

    base_url = store.sqlalchemy_url

    if protocol is None and port is None:
        return base_url

    # Use SQLAlchemy's URL utilities for proper parsing
    from sqlalchemy.engine.url import make_url

    url = make_url(base_url)

    # Apply protocol and port overrides
    if protocol is not None and port is not None:
        url = url.set(drivername=protocol, port=port)
    elif protocol is not None:
        url = url.set(drivername=protocol)
    elif port is not None:
        url = url.set(port=port)

    return url.render_as_string(hide_password=False)


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
    protocol: str | None = None,
    port: int | None = None,
) -> tuple[str, MetaData]:
    """Get SQLAlchemy URL and Metaxy system tables metadata for a metadata store.

    This function retrieves both the connection URL and system table metadata
    for a store, with the store's `table_prefix` automatically applied to table names.

    Args:
        store: IbisMetadataStore instance
        protocol: Optional protocol (drivername) to replace the existing one in the URL.
            Useful when Ibis uses a different protocol than SQLAlchemy requires.
        port: Optional port to replace the existing one in the URL.
            Useful when the SQLAlchemy driver uses a different port than Ibis.

    Returns:
        Tuple of (sqlalchemy_url, system_metadata)

    Raises:
        ValueError: If store's sqlalchemy_url is empty

    Note:
        Metadata stores do their best at providing the correct `sqlalchemy_url`, so you typically don't need to modify the output of this function.
    """
    url = _get_store_sqlalchemy_url(store, protocol=protocol, port=port)
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

        table_name = store.get_table_name(feature_key)

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

    from metaxy.models.constants import METAXY_CREATED_AT, METAXY_FEATURE_VERSION

    # Composite key/index columns: metaxy_feature_version + id_columns + metaxy_created_at
    key_columns = [METAXY_FEATURE_VERSION] + list(spec.id_columns) + [METAXY_CREATED_AT]

    if inject_primary_key:
        # Add primary key constraint
        pk_constraint = PrimaryKeyConstraint(*key_columns, name="metaxy_pk")
        table.append_constraint(pk_constraint)

    if inject_index:
        # Add composite index
        idx = Index(
            "metaxy_idx",
            *key_columns,
        )
        table.append_constraint(idx)


def filter_feature_sqla_metadata(
    store: IbisMetadataStore,
    source_metadata: MetaData,
    project: str | None = None,
    filter_by_project: bool = True,
    inject_primary_key: bool | None = None,
    inject_index: bool | None = None,
    protocol: str | None = None,
    port: int | None = None,
) -> tuple[str, MetaData]:
    """Get SQLAlchemy URL and feature table metadata for a metadata store.

    This function filters the source metadata to include only feature tables
    belonging to the specified project, and returns the connection URL for the store.

    This function must be called after init_metaxy() to ensure features are loaded.

    Args:
        store: IbisMetadataStore instance
        source_metadata: Source SQLAlchemy MetaData to filter.
        project: Project name to filter by. If None, uses MetaxyConfig.get().project
        filter_by_project: If True, only include features for the specified project.
                          If False, include all features.
        inject_primary_key: If True, inject composite primary key constraints.
                           If False, do not inject. If None, uses config default.
        inject_index: If True, inject composite index.
                     If False, do not inject. If None, uses config default.
        protocol: Optional protocol to replace the existing one in the URL.
            Useful when Ibis uses a different protocol than SQLAlchemy requires.
        port: Optional port to replace the existing one in the URL.
            Useful when the SQLAlchemy driver uses a different port than Ibis.

    Returns:
        Tuple of (sqlalchemy_url, filtered_metadata)

    Raises:
        ValueError: If store's sqlalchemy_url is empty
        ImportError: If source_metadata is None and SQLModel is not installed

    Note:
        Metadata stores do their best at providing the correct `sqlalchemy_url`, so you typically don't need to modify the output of this function.

    Example: Basic Usage

        ```py
        from metaxy.ext.sqlalchemy import filter_feature_sqla_metadata
        from metaxy import init_metaxy
        from metaxy.config import MetaxyConfig

        # Load features first
        init_metaxy()

        # Get store instance
        config = MetaxyConfig.get()
        store = config.get_store("my_store")

        # With custom metadata
        from sqlalchemy import MetaData
        my_metadata = MetaData()
        # ... define tables in my_metadata ...

        # apply the filter function
        url, metadata = filter_feature_sqla_metadata(store, source_metadata=my_metadata)
        ```

    Example: With SQLModel

        ```py
        # With SQLModel
        from sqlmodel import SQLModel
        url, metadata = filter_feature_sqla_metadata(store, SQLModel.metadata)
        ```
    """
    url = _get_store_sqlalchemy_url(store, protocol=protocol, port=port)
    metadata = _get_features_metadata(
        source_metadata=source_metadata,
        store=store,
        project=project,
        filter_by_project=filter_by_project,
        inject_primary_key=inject_primary_key,
        inject_index=inject_index,
    )
    return url, metadata
