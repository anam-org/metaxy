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
    pass

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
            **kwargs: Additional keyword arguments (e.g., table=True for SQLModel)

        Returns:
            New class that is both a SQLModel table and a Metaxy feature
        """
        # Override frozen config for SQLModel - instances need to be mutable for ORM
        if "model_config" not in namespace:
            from pydantic import ConfigDict

            namespace["model_config"] = ConfigDict(frozen=False)

        # Check plugin config for inject_primary_key default
        if inject_primary_key is None:
            config = MetaxyConfig.get()
            sqlmodel_config = config.get_plugin("sqlmodel", SQLModelPluginConfig)
            inject_primary_key = sqlmodel_config.inject_primary_key

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

            # Inject composite primary key if requested
            if inject_primary_key:
                cls._inject_composite_primary_key(namespace, spec, cls_name)

        # Call super().__new__ which follows MRO: MetaxyMeta -> SQLModelMetaclass -> ...
        # MetaxyMeta will consume the spec parameter and pass remaining kwargs to SQLModelMetaclass
        new_class = super().__new__(
            cls, cls_name, bases, namespace, spec=spec, **kwargs
        )

        return new_class

    @staticmethod
    def _inject_composite_primary_key(
        namespace: dict[str, Any],
        spec: FeatureSpecWithIDColumns,
        cls_name: str,
    ) -> None:
        """Inject composite primary key constraint into the table.

        Creates a composite primary key including:
        - All user-provided id_columns
        - metaxy_created_at
        - metaxy_data_version

        Args:
            namespace: Class namespace to modify
            spec: Feature specification with id_columns
            cls_name: Name of the class being created
        """
        from sqlalchemy import PrimaryKeyConstraint

        pk_columns = list(spec.id_columns) + [METAXY_CREATED_AT, METAXY_DATA_VERSION]
        pk_constraint = PrimaryKeyConstraint(*pk_columns, name="pk_metaxy_composite")

        # Add to __table_args__
        if "__table_args__" in namespace:
            # User already defined __table_args__, merge with it
            existing_args = namespace["__table_args__"]
            if isinstance(existing_args, dict):
                # Dict format - convert to tuple + dict
                namespace["__table_args__"] = (pk_constraint, existing_args)
            elif isinstance(existing_args, tuple):
                # Tuple format - append constraint
                namespace["__table_args__"] = existing_args + (pk_constraint,)
            else:
                raise ValueError(
                    f"Invalid __table_args__ type in {cls_name}: {type(existing_args)}"
                )
        else:
            namespace["__table_args__"] = (pk_constraint,)


class BaseSQLModelFeature(  # pyright: ignore[reportIncompatibleMethodOverride, reportUnsafeMultipleInheritance]
    SQLModel, BaseFeature, metaclass=SQLModelFeatureMeta, spec=None
):  # type: ignore[misc]
    """Base class for features that are also SQLModel tables.

    !!! note

        Unlike regular Feature classes, `BaseSQLModelFeature` instances are mutable
        to support SQLModel ORM operations. Only the spec and graph class attributes
        are used from BaseFeature, not the instance behavior.

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
