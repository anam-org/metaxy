"""SQLModel integration for Metaxy.

This module provides a combined metaclass that allows Metaxy Feature classes
to also be SQLModel table classes, enabling seamless integration with SQLAlchemy/SQLModel ORMs.
"""

from typing import TYPE_CHECKING, Any

from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelMetaclass

from metaxy.config import MetaxyConfig
from metaxy.ext.sqlmodel.config import SQLModelPluginConfig
from metaxy.models.constants import (
    ALL_SYSTEM_COLUMNS,
    METAXY_CREATED_AT,
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_FEATURE_SPEC_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
    SYSTEM_COLUMN_PREFIX,
)
from metaxy.models.feature import BaseFeature, MetaxyMeta
from metaxy.models.feature_spec import FeatureSpecWithIDColumns

if TYPE_CHECKING:
    from sqlalchemy import MetaData

    from metaxy.metadata_store.ibis import IbisMetadataStore

RESERVED_SQLMODEL_FIELD_NAMES = frozenset(
    set(ALL_SYSTEM_COLUMNS)
    | {
        name.removeprefix(SYSTEM_COLUMN_PREFIX)
        for name in ALL_SYSTEM_COLUMNS
        if name.startswith(SYSTEM_COLUMN_PREFIX)
    }
)


class SQLModelFeatureMeta(MetaxyMeta, SQLModelMetaclass):  # pyright: ignore[reportUnsafeMultipleInheritance]
    def __new__(
        cls,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *,
        spec: FeatureSpecWithIDColumns | None = None,
        inject_primary_key: bool | None = None,
        inject_index: bool | None = None,
        **kwargs: Any,
    ) -> type[Any]:
        """Create a new SQLModel + Metaxy Feature class.

        Args:
            cls_name: Name of the class being created
            bases: Base classes
            namespace: Class namespace (attributes and methods)
            spec: Metaxy FeatureSpec (required for concrete features)
            inject_primary_key: If True, automatically create composite primary key
                including id_columns + (metaxy_created_at, metaxy_data_version).
            inject_index: If True, automatically create composite index
                including id_columns + (metaxy_created_at, metaxy_data_version).
            **kwargs: Additional keyword arguments (e.g., table=True for SQLModel)

        Returns:
            New class that is both a SQLModel table and a Metaxy feature
        """
        # Override frozen config for SQLModel - instances need to be mutable for ORM
        if "model_config" not in namespace:
            from pydantic import ConfigDict

            namespace["model_config"] = ConfigDict(frozen=False)

        # Check plugin config for defaults
        config = MetaxyConfig.get()
        sqlmodel_config = config.get_plugin("sqlmodel", SQLModelPluginConfig)
        if inject_primary_key is None:
            inject_primary_key = sqlmodel_config.inject_primary_key
        if inject_index is None:
            inject_index = sqlmodel_config.inject_index

        # If this is a concrete table (table=True) with a spec
        if kwargs.get("table") and spec is not None:
            # Prevent user-defined fields from shadowing system-managed columns
            conflicts = {
                attr_name
                for attr_name in namespace
                if attr_name in RESERVED_SQLMODEL_FIELD_NAMES
            }

            # Also guard against explicit sa_column_kwargs targeting system columns
            for attr_name, attr_value in namespace.items():
                sa_column_kwargs = getattr(attr_value, "sa_column_kwargs", None)
                if isinstance(sa_column_kwargs, dict):
                    column_name = sa_column_kwargs.get("name")
                    if column_name in ALL_SYSTEM_COLUMNS:
                        conflicts.add(attr_name)

            if conflicts:
                reserved = ", ".join(sorted(ALL_SYSTEM_COLUMNS))
                conflict_list = ", ".join(sorted(conflicts))
                raise ValueError(
                    "Cannot define SQLModel field(s) "
                    f"{conflict_list} because they map to reserved Metaxy system columns. "
                    f"Reserved columns: {reserved}"
                )

            # Automatically set __tablename__ from the feature key if not provided
            if "__tablename__" not in namespace:
                namespace["__tablename__"] = spec.key.table_name

            # Inject constraints if requested
            if inject_primary_key or inject_index:
                cls._inject_constraints(
                    namespace, spec, cls_name, inject_primary_key, inject_index
                )

        # Call super().__new__ which follows MRO: MetaxyMeta -> SQLModelMetaclass -> ...
        # MetaxyMeta will consume the spec parameter and pass remaining kwargs to SQLModelMetaclass
        new_class = super().__new__(
            cls, cls_name, bases, namespace, spec=spec, **kwargs
        )

        return new_class

    @staticmethod
    def _inject_constraints(
        namespace: dict[str, Any],
        spec: FeatureSpecWithIDColumns,
        cls_name: str,
        inject_primary_key: bool,
        inject_index: bool,
    ) -> None:
        """Inject composite primary key and/or index constraints into the table.

        Creates constraints including:
        - All user-provided id_columns
        - metaxy_created_at
        - metaxy_data_version

        Args:
            namespace: Class namespace to modify
            spec: Feature specification with id_columns
            cls_name: Name of the class being created
            inject_primary_key: If True, inject composite primary key
            inject_index: If True, inject composite index
        """
        from sqlalchemy import Index, PrimaryKeyConstraint

        # Composite key/index columns: id_columns + metaxy_created_at + metaxy_data_version
        key_columns = list(spec.id_columns) + [METAXY_CREATED_AT, METAXY_DATA_VERSION]

        constraints = []
        if inject_primary_key:
            pk_constraint = PrimaryKeyConstraint(
                *key_columns, name="pk_metaxy_composite"
            )
            constraints.append(pk_constraint)

        if inject_index:
            # Note: table name will be available after SQLModel creates the table
            # We use a placeholder name here, it will be finalized by SQLModel
            idx = Index("idx_metaxy_composite", *key_columns)
            constraints.append(idx)

        if not constraints:
            return

        # Add to __table_args__
        if "__table_args__" in namespace:
            # User already defined __table_args__, merge with it
            existing_args = namespace["__table_args__"]
            if isinstance(existing_args, dict):
                # Dict format - convert to tuple + dict
                namespace["__table_args__"] = tuple(constraints) + (existing_args,)
            elif isinstance(existing_args, tuple):
                # Tuple format - append constraints
                namespace["__table_args__"] = existing_args + tuple(constraints)
            else:
                raise ValueError(
                    f"Invalid __table_args__ type in {cls_name}: {type(existing_args)}"
                )
        else:
            namespace["__table_args__"] = tuple(constraints)


class BaseSQLModelFeature(  # pyright: ignore[reportIncompatibleMethodOverride, reportUnsafeMultipleInheritance]
    SQLModel, BaseFeature, metaclass=SQLModelFeatureMeta, spec=None
):  # type: ignore[misc]
    """Base class for `Metaxy` features that are also `SQLModel` tables.

    !!! example

        ```py
        from metaxy.integrations.sqlmodel import BaseSQLModelFeature
        from metaxy import FeatureSpec, FeatureKey, FieldSpec, FieldKey
        from sqlmodel import Field

        class VideoFeature(
            BaseSQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["video"]),
                id_columns=["uid"],
                fields=[
                    FieldSpec(
                        key=FieldKey(["video_file"]),
                        code_version="1",
                    ),
                ],
            ),
        ):

            uid: str = Field(primary_key=True)
            path: str
            duration: float

            # Now you can use both Metaxy and SQLModel features:
            # - VideoFeature.feature_version() -> Metaxy versioning
            # - session.exec(select(VideoFeature)) -> SQLModel queries
        ```
    """

    # Override the frozen config from Feature's FrozenBaseModel
    # SQLModel instances need to be mutable for ORM operations
    model_config = {"frozen": False}  # pyright: ignore[reportAssignmentType]

    # Using sa_column_kwargs to map to the actual column names used by Metaxy
    metaxy_provenance: str | None = Field(
        default=None,
        sa_column_kwargs={
            "name": METAXY_PROVENANCE,
        },
    )

    metaxy_provenance_by_field: str | None = Field(
        default=None,
        sa_type=JSON,
        sa_column_kwargs={
            "name": METAXY_PROVENANCE_BY_FIELD,
        },
    )

    metaxy_feature_version: str | None = Field(
        default=None,
        sa_column_kwargs={
            "name": METAXY_FEATURE_VERSION,
        },
    )

    metaxy_feature_spec_version: str | None = Field(
        default=None,
        sa_column_kwargs={
            "name": METAXY_FEATURE_SPEC_VERSION,
        },
    )

    metaxy_snapshot_version: str | None = Field(
        default=None,
        sa_column_kwargs={
            "name": METAXY_SNAPSHOT_VERSION,
        },
    )

    metaxy_data_version: str | None = Field(
        default=None,
        sa_column_kwargs={
            "name": METAXY_DATA_VERSION,
        },
    )

    metaxy_data_version_by_field: str | None = Field(
        default=None,
        sa_type=JSON,
        sa_column_kwargs={
            "name": METAXY_DATA_VERSION_BY_FIELD,
        },
    )

    metaxy_created_at: str | None = Field(
        default=None,
        sa_column_kwargs={
            "name": METAXY_CREATED_AT,
        },
    )


# Convenience wrappers for filtering SQLModel metadata


def filter_feature_sqlmodel_metadata(
    store: "IbisMetadataStore",
    source_metadata: "MetaData",
    project: str | None = None,
    filter_by_project: bool = True,
    inject_primary_key: bool | None = None,
    inject_index: bool | None = None,
) -> tuple[str, "MetaData"]:
    """Get SQLAlchemy URL and filtered SQLModel feature metadata for a metadata store.

    This function transforms SQLModel table names to include the store's table_prefix,
    ensuring that table names in the metadata match what's expected in the database.

    You can pass `SQLModel.metadata` directly - this function will transform table names
    by adding the store's `table_prefix`. The returned metadata will have prefixed table
    names that match the actual database tables.

    This function must be called after init_metaxy() to ensure features are loaded.

    Args:
        store: IbisMetadataStore instance (provides table_prefix and sqlalchemy_url)
        source_metadata: Source SQLAlchemy MetaData to filter (typically SQLModel.metadata).
                        Tables are looked up in this metadata by their unprefixed names.
        project: Project name to filter by. If None, uses MetaxyConfig.get().project
        filter_by_project: If True, only include features for the specified project.
        inject_primary_key: If True, inject composite primary key constraints.
                           If False, do not inject. If None, uses config default.
        inject_index: If True, inject composite index.
                     If False, do not inject. If None, uses config default.

    Returns:
        Tuple of (sqlalchemy_url, filtered_metadata)

    Raises:
        ValueError: If store's sqlalchemy_url is empty

    Example:

        ```py
        from sqlmodel import SQLModel
        from metaxy.ext.sqlmodel import filter_feature_sqlmodel_metadata
        from metaxy import init_metaxy
        from metaxy.config import MetaxyConfig

        # Load features first
        init_metaxy()

        # Get store instance
        config = MetaxyConfig.get()
        store = config.get_store("my_store")

        # Filter SQLModel metadata with prefix transformation
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Use with Alembic env.py
        from alembic import context
        url, target_metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)
        context.configure(url=url, target_metadata=target_metadata)
        ```
    """
    from sqlalchemy import MetaData

    config = MetaxyConfig.get()

    if project is None:
        project = config.project

    # Check plugin config for defaults
    sqlmodel_config = config.get_plugin("sqlmodel", SQLModelPluginConfig)
    if inject_primary_key is None:
        inject_primary_key = sqlmodel_config.inject_primary_key
    if inject_index is None:
        inject_index = sqlmodel_config.inject_index

    # Get SQLAlchemy URL from store
    if not store.sqlalchemy_url:
        raise ValueError("IbisMetadataStore has an empty `sqlalchemy_url`.")
    url = store.sqlalchemy_url

    # Create new metadata with transformed table names
    filtered_metadata = MetaData()

    # Build a mapping of table names to feature classes
    table_to_feature: dict[str, type[BaseSQLModelFeature]] = {}
    for feature_cls in BaseSQLModelFeature.__subclasses__():
        # Skip if this is not a concrete table (no __table__)
        if not hasattr(feature_cls, "__table__"):
            continue

        # Filter by project if requested
        if filter_by_project:
            feature_project = getattr(feature_cls, "project", None)
            if feature_project != project:
                continue

        # Get the unprefixed table name from SQLModel
        unprefixed_name: str = getattr(feature_cls, "__tablename__")
        table_to_feature[unprefixed_name] = feature_cls

    # Iterate over tables in source metadata
    for table_name, original_table in source_metadata.tables.items():
        # Check if this table corresponds to a SQLModel feature
        if table_name not in table_to_feature:
            continue

        feature_cls = table_to_feature[table_name]

        # Compute prefixed name using store's table_prefix
        prefixed_name = store.get_table_name(feature_cls.spec().key)

        # Copy table to new metadata with prefixed name
        new_table = original_table.to_metadata(filtered_metadata, name=prefixed_name)

        # Inject constraints if requested
        if inject_primary_key or inject_index:
            from metaxy.ext.sqlalchemy.plugin import _inject_constraints

            spec = feature_cls.spec()
            _inject_constraints(
                table=new_table,
                spec=spec,
                inject_primary_key=inject_primary_key,
                inject_index=inject_index,
            )

    return url, filtered_metadata
