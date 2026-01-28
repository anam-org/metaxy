"""SQLModel integration for Metaxy.

This module provides a combined metaclass that allows Metaxy Feature classes
to also be SQLModel table classes, enabling seamless integration with SQLAlchemy/SQLModel ORMs.
"""

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import AwareDatetime, BaseModel
from sqlalchemy.types import JSON, DateTime
from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelMetaclass

from metaxy import FeatureSpec
from metaxy._decorators import public
from metaxy.config import MetaxyConfig
from metaxy.ext.sqlmodel.config import SQLModelPluginConfig
from metaxy.models.constants import (
    ALL_SYSTEM_COLUMNS,
    METAXY_CREATED_AT,
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_DELETED_AT,
    METAXY_FEATURE_VERSION,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
    METAXY_UPDATED_AT,
    SYSTEM_COLUMN_PREFIX,
)
from metaxy.models.feature import BaseFeature, FeatureGraph, MetaxyMeta
from metaxy.models.feature_spec import FeatureSpecWithIDColumns
from metaxy.models.types import ValidatedFeatureKey

if TYPE_CHECKING:
    from sqlalchemy import MetaData

    from metaxy.metadata_store.ibis import IbisMetadataStore

RESERVED_SQLMODEL_FIELD_NAMES = frozenset(
    set(ALL_SYSTEM_COLUMNS)
    | {name.removeprefix(SYSTEM_COLUMN_PREFIX) for name in ALL_SYSTEM_COLUMNS if name.startswith(SYSTEM_COLUMN_PREFIX)}
)


class MetaxyTableInfo(BaseModel):
    feature_key: ValidatedFeatureKey


class SQLModelFeatureMeta(MetaxyMeta, SQLModelMetaclass):
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
                including (metaxy_feature_version, *id_columns, metaxy_updated_at).
            inject_index: If True, automatically create composite index
                including (metaxy_feature_version, *id_columns, metaxy_updated_at).
            **kwargs: Additional keyword arguments (e.g., table=True for SQLModel)

        Returns:
            New class that is both a SQLModel table and a Metaxy feature
        """
        # Override frozen config for SQLModel - instances need to be mutable for ORM
        if "model_config" not in namespace:
            from pydantic import ConfigDict

            namespace["model_config"] = ConfigDict(frozen=False)

        # Check plugin config for defaults
        sqlmodel_config = MetaxyConfig.get_plugin("sqlmodel", SQLModelPluginConfig)
        if inject_primary_key is None:
            inject_primary_key = sqlmodel_config.inject_primary_key
        if inject_index is None:
            inject_index = sqlmodel_config.inject_index

        # If this is a concrete table (table=True) with a spec
        if kwargs.get("table") and spec is not None:
            # Forbid custom __tablename__ since it won't work with metadata store's get_table_name()
            if "__tablename__" in namespace:
                raise ValueError(
                    f"Cannot define custom __tablename__ in {cls_name}. "
                    "The table name is automatically derived from the feature key. "
                    "If you need a different table name, adjust the feature key instead."
                )

            # Prevent user-defined fields from shadowing system-managed columns
            conflicts = {attr_name for attr_name in namespace if attr_name in RESERVED_SQLMODEL_FIELD_NAMES}

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

            # Automatically set __tablename__ from the feature key
            namespace["__tablename__"] = spec.key.table_name

            # Inject table args (info metadata + optional constraints)
            cls._inject_table_args(namespace, spec, cls_name, inject_primary_key, inject_index)

        # Call super().__new__ which follows MRO: MetaxyMeta -> SQLModelMetaclass -> ...
        # MetaxyMeta will consume the spec parameter and pass remaining kwargs to SQLModelMetaclass
        new_class = super().__new__(cls, cls_name, bases, namespace, spec=spec, **kwargs)

        return new_class

    @staticmethod
    def _inject_table_args(
        namespace: dict[str, Any],
        spec: FeatureSpec,
        cls_name: str,
        inject_primary_key: bool,
        inject_index: bool,
    ) -> None:
        """Inject Metaxy table args (info metadata + optional constraints) via __table_args__.

        This method handles:

        1. Always injects info metadata with feature key for efficient lookup

        2. Optionally injects composite primary key and/or index constraints

        Args:
            namespace: Class namespace to modify
            spec: Feature specification with key and id_columns
            cls_name: Name of the class being created
            inject_primary_key: If True, inject composite primary key
            inject_index: If True, inject composite index
        """

        from sqlalchemy import Index, PrimaryKeyConstraint

        # Prepare info dict with Metaxy metadata (always added)
        metaxy_info = {"metaxy-system": MetaxyTableInfo(feature_key=spec.key).model_dump()}

        # Base table kwargs that are always applied
        base_table_kwargs = {"extend_existing": True}

        # Prepare constraints if requested
        constraints = []
        if inject_primary_key or inject_index:
            # Composite key/index columns: metaxy_feature_version + id_columns + metaxy_updated_at
            key_columns = [METAXY_FEATURE_VERSION, *spec.id_columns, METAXY_UPDATED_AT]

            if inject_primary_key:
                constraints.append(PrimaryKeyConstraint(*key_columns, name="metaxy_pk"))

            if inject_index:
                constraints.append(Index("metaxy_idx", *key_columns))

        # Merge with existing __table_args__
        if "__table_args__" in namespace:
            existing_args = namespace["__table_args__"]

            if isinstance(existing_args, dict):
                # Dict format: merge info and base kwargs, convert to tuple if we have constraints
                existing_info = existing_args.get("info", {})
                existing_info.update(metaxy_info)
                existing_args["info"] = existing_info
                # Merge base table kwargs (don't override user settings)
                for key, value in base_table_kwargs.items():
                    existing_args.setdefault(key, value)

                if constraints:
                    # Convert to tuple format with constraints
                    namespace["__table_args__"] = tuple(constraints) + (existing_args,)
                # else: keep as dict

            elif isinstance(existing_args, tuple):
                # Tuple format: append constraints and merge info in table kwargs dict
                # Extract existing constraints and table kwargs
                if existing_args and isinstance(existing_args[-1], dict):
                    # Has table kwargs dict at the end
                    existing_constraints = existing_args[:-1]
                    table_kwargs = dict(existing_args[-1])
                else:
                    # No table kwargs dict
                    existing_constraints = existing_args
                    table_kwargs = {}

                # Merge info
                existing_info = table_kwargs.get("info", {})
                existing_info.update(metaxy_info)
                table_kwargs["info"] = existing_info
                # Merge base table kwargs (don't override user settings)
                for key, value in base_table_kwargs.items():
                    table_kwargs.setdefault(key, value)

                # Combine: existing constraints + new constraints + table kwargs
                namespace["__table_args__"] = existing_constraints + tuple(constraints) + (table_kwargs,)
            else:
                raise ValueError(f"Invalid __table_args__ type in {cls_name}: {type(existing_args)}")
        else:
            # No existing __table_args__
            table_kwargs = {**base_table_kwargs, "info": metaxy_info}
            if constraints:
                # Create tuple format with constraints + table kwargs
                namespace["__table_args__"] = tuple(constraints) + (table_kwargs,)
            else:
                # Just table kwargs, use dict format
                namespace["__table_args__"] = table_kwargs


@public
class BaseSQLModelFeature(SQLModel, BaseFeature, metaclass=SQLModelFeatureMeta, spec=None):
    """Base class for `Metaxy` features that are also `SQLModel` tables.

    !!! example

        <!-- skip next -->
        ```py
        from metaxy.integrations.sqlmodel import BaseSQLModelFeature
        from sqlmodel import Field


        class VideoFeature(
            BaseSQLModelFeature,
            table=True,
            spec=mx.FeatureSpec(
                key=mx.FeatureKey(["video"]),
                id_columns=["uid"],
                fields=[
                    mx.FieldSpec(
                        key=mx.FieldKey(["video_file"]),
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
    model_config = {"frozen": False}

    # Re-declare ClassVar attributes from BaseFeature for type checker visibility.
    # These are set by MetaxyMeta at class creation time but type checkers can't see them
    # through the complex metaclass inheritance chain.
    _spec: ClassVar[FeatureSpec]
    graph: ClassVar[FeatureGraph]
    project: ClassVar[str]

    # Using sa_column_kwargs to map to the actual column names used by Metaxy
    # Descriptions match those in BaseFeature for consistency in Dagster UI
    metaxy_provenance: str | None = Field(
        default=None,
        description="Hash of metaxy_provenance_by_field",
        sa_column_kwargs={
            "name": METAXY_PROVENANCE,
        },
        nullable=False,
    )

    metaxy_provenance_by_field: dict[str, str] = Field(
        default=None,
        description="Field-level provenance hashes (maps field names to hashes)",
        sa_type=JSON,
        sa_column_kwargs={
            "name": METAXY_PROVENANCE_BY_FIELD,
        },
        nullable=False,
    )

    metaxy_feature_version: str | None = Field(
        default=None,
        description="Hash of the feature definition (dependencies + fields + code_versions)",
        sa_column_kwargs={
            "name": METAXY_FEATURE_VERSION,
        },
        nullable=False,
    )

    metaxy_snapshot_version: str | None = Field(
        default=None,
        description="Hash of the entire feature graph snapshot",
        sa_column_kwargs={
            "name": METAXY_SNAPSHOT_VERSION,
        },
        nullable=False,
    )

    metaxy_data_version: str | None = Field(
        default=None,
        description="Hash of metaxy_data_version_by_field",
        sa_column_kwargs={
            "name": METAXY_DATA_VERSION,
        },
        nullable=False,
    )

    metaxy_data_version_by_field: dict[str, str] | None = Field(
        default=None,
        description="Field-level data version hashes (maps field names to version hashes)",
        sa_type=JSON,
        sa_column_kwargs={
            "name": METAXY_DATA_VERSION_BY_FIELD,
        },
        nullable=False,
    )

    metaxy_created_at: AwareDatetime | None = Field(
        default=None,
        description="Timestamp when the metadata row was created (UTC)",
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={
            "name": METAXY_CREATED_AT,
        },
        nullable=False,
    )

    metaxy_updated_at: AwareDatetime | None = Field(
        default=None,
        description="Timestamp when the metadata row was last updated (UTC)",
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={
            "name": METAXY_UPDATED_AT,
        },
        nullable=False,
    )

    metaxy_materialization_id: str | None = Field(
        default=None,
        description="External orchestration run ID (e.g., Dagster Run ID)",
        sa_column_kwargs={
            "name": METAXY_MATERIALIZATION_ID,
        },
        nullable=True,
    )

    metaxy_deleted_at: AwareDatetime | None = Field(
        default=None,
        description="Soft delete timestamp (UTC); null means active row",
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={
            "name": METAXY_DELETED_AT,
        },
        nullable=True,
    )


# Convenience wrappers for filtering SQLModel metadata


@public
def filter_feature_sqlmodel_metadata(
    store: "IbisMetadataStore",
    source_metadata: "MetaData",
    project: str | None = None,
    filter_by_project: bool = True,
    inject_primary_key: bool | None = None,
    inject_index: bool | None = None,
    protocol: str | None = None,
    port: int | None = None,
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
        protocol: Optional protocol to replace the existing one in the URL.
            Useful when Ibis uses a different protocol than SQLAlchemy requires.
        port: Optional port to replace the existing one in the URL.
            Useful when the SQLAlchemy driver uses a different port than Ibis.

    Returns:
        Tuple of (sqlalchemy_url, filtered_metadata)

    Raises:
        ValueError: If store's sqlalchemy_url is empty

    Note:
        For ClickHouse, the `sqlalchemy_url` property already returns the native
        protocol with port 9000, so you typically don't need to override these.

    Example: Basic Usage

        <!-- skip next -->
        ```py
        from sqlmodel import SQLModel
        from metaxy.ext.sqlmodel import filter_feature_sqlmodel_metadata
        from alembic import context

        # Load features first
        mx.init_metaxy()

        # Get store instance
        config = mx.MetaxyConfig.get()
        store = config.get_store("my_store")

        # Filter SQLModel metadata with prefix transformation
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Use with Alembic env.py
        url, target_metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)
        context.configure(url=url, target_metadata=target_metadata)
        ```
    """
    from sqlalchemy import MetaData

    from metaxy.ext.sqlalchemy.plugin import _get_store_sqlalchemy_url

    config = MetaxyConfig.get(load=True)

    if project is None:
        project = config.project

    # Check plugin config for defaults
    sqlmodel_config = config.get_plugin("sqlmodel", SQLModelPluginConfig)
    if inject_primary_key is None:
        inject_primary_key = sqlmodel_config.inject_primary_key
    if inject_index is None:
        inject_index = sqlmodel_config.inject_index

    url = _get_store_sqlalchemy_url(store, protocol=protocol, port=port)

    # Create new metadata with transformed table names
    filtered_metadata = MetaData()

    # Get the FeatureGraph to look up feature classes by key
    from metaxy.models.feature import FeatureGraph

    feature_graph = FeatureGraph.get_active()

    # Iterate over tables in source metadata
    for table_name, original_table in source_metadata.tables.items():
        # Check if this table has Metaxy feature metadata
        if metaxy_system_info := original_table.info.get("metaxy-system"):
            metaxy_info = MetaxyTableInfo.model_validate(metaxy_system_info)
            feature_key = metaxy_info.feature_key
        else:
            continue
        # Look up the feature definition from the FeatureGraph
        definition = feature_graph.feature_definitions_by_key.get(feature_key)
        if definition is None:
            # Skip tables for features that aren't registered
            continue

        # Filter by project if requested
        if filter_by_project:
            if definition.project != project:
                continue

        # Compute prefixed name using store's table_prefix
        prefixed_name = store.get_table_name(feature_key)

        # Copy table to new metadata with prefixed name
        new_table = original_table.to_metadata(filtered_metadata, name=prefixed_name)

        # Inject constraints if requested
        if inject_primary_key or inject_index:
            from metaxy.ext.sqlalchemy.plugin import _inject_constraints

            _inject_constraints(
                table=new_table,
                spec=definition.spec,
                inject_primary_key=inject_primary_key,
                inject_index=inject_index,
            )

    return url, filtered_metadata
