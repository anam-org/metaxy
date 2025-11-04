"""SQLModel integration for Metaxy.

This module provides a combined metaclass that allows Metaxy Feature classes
to also be SQLModel table classes, enabling seamless integration with SQLAlchemy/SQLModel ORMs.
"""

from typing import TYPE_CHECKING, Any

from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelMetaclass

from metaxy.config import MetaxyConfig
from metaxy.models.feature import BaseFeature, MetaxyMeta
from metaxy.models.feature_spec import BaseFeatureSpecWithIDColumns

if TYPE_CHECKING:
    pass


class SQLModelFeatureMeta(MetaxyMeta, SQLModelMetaclass):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """Combined metaclass for SQLModel + Metaxy Feature.

    This metaclass inherits from both MetaxyMeta and SQLModelMetaclass,
    allowing classes to be both SQLModel tables and Metaxy features.

    The MRO (Method Resolution Order) ensures that:
    1. MetaxyMeta handles Metaxy feature registration
    2. SQLModelMetaclass handles SQLAlchemy table creation

    Note: MetaxyMeta comes first in the inheritance to ensure spec parameter is handled.

    Automatic table naming:
        __tablename__ is automatically generated from the feature key if not explicitly provided.
        This ensures consistency with Metaxy's metadata store table naming conventions.

    Example:
        ```py
        from metaxy.integrations.sqlmodel import SQLModelFeature
        from metaxy import BaseFeatureSpec, FeatureKey, FieldSpec, FieldKey
        from sqlmodel import Field

        class MyFeature(
            SQLModelFeature,
            table=True,
            spec=BaseFeatureSpec(
                key=FeatureKey(["my", "feature"]),

                fields=[
                    FieldSpec(
                        key=FieldKey(["data"]),
                        code_version="1",
                    ),
                ],
            ),
        ):
            __tablename__ = "my_feature"

            uid: str = Field(primary_key=True)
            data: str
        ```
    """

    def __new__(
        cls,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *,
        spec: BaseFeatureSpecWithIDColumns | None = None,
        **kwargs: Any,
    ) -> type[Any]:
        """Create a new SQLModel + Feature class.

        Args:
            cls_name: Name of the class being created
            bases: Base classes
            namespace: Class namespace (attributes and methods)
            spec: Metaxy BaseFeatureSpec (required for concrete features)
            **kwargs: Additional keyword arguments (e.g., table=True for SQLModel)

        Returns:
            New class that is both a SQLModel table and a Metaxy feature
        """
        # If this is a concrete table (table=True) with a spec
        config = MetaxyConfig.get()

        if kwargs.get("table") and spec is not None:
            # Automatically set __tablename__ from the feature key if not provided
            if (
                "__tablename__" not in namespace
                and config.ext.sqlmodel.infer_db_table_names
            ):
                namespace["__tablename__"] = spec.key.table_name

        # Call super().__new__ which follows MRO: MetaxyMeta -> SQLModelMetaclass -> ...
        # MetaxyMeta will consume the spec parameter and pass remaining kwargs to SQLModelMetaclass
        new_class = super().__new__(
            cls, cls_name, bases, namespace, spec=spec, **kwargs
        )

        # After class creation, validate id_columns are not server-defined
        if kwargs.get("table") and spec is not None:
            cls._validate_id_columns_not_server_defined(new_class, spec)

        return new_class

    @staticmethod
    def _validate_id_columns_not_server_defined(
        new_class: type[Any],
        spec: BaseFeatureSpecWithIDColumns,
    ) -> None:
        """Validate that primary key id_columns are not autoincrement.

        In analytical workloads, id_columns must be predictable ahead of time
        to enable join predictions. Autoincrement primary keys break this predictability.

        Args:
            new_class: The newly created SQLModel class
            spec: The BaseFeatureSpec containing id_columns definition

        Raises:
            ValueError: If any id_column is an autoincrement primary key
        """
        # Get the actual id_columns (use default if not specified)
        id_columns = spec.id_columns if spec.id_columns else ["sample_uid"]

        # Check each id_column field
        for col_name in id_columns:
            # Get the field from the model_fields (SQLModel stores field info there)
            # Use model_fields for Pydantic v2 compatibility
            if not hasattr(new_class, "model_fields"):
                continue  # SQLModel may not have initialized yet

            fields = getattr(new_class, "model_fields", {})
            if col_name not in fields:
                continue  # Will be caught elsewhere if missing

            # The field_info is the FieldInfo object directly
            field_info = fields[col_name]

            # Check if this is a primary key
            is_primary_key = getattr(field_info, "primary_key", False)

            # Get sa_column_kwargs
            sa_column_kwargs = getattr(field_info, "sa_column_kwargs", {})

            # sa_column_kwargs might be PydanticUndefined, None, or a dict
            if not isinstance(sa_column_kwargs, dict):
                sa_column_kwargs = {}

            # Check for autoincrement on primary keys
            if is_primary_key and sa_column_kwargs.get("autoincrement", False):
                raise ValueError(
                    f"ID column '{col_name}' in {new_class.__name__} cannot be an autoincrement primary key. "
                    f"In analytical workloads, ID values must be predictable ahead of time "
                    f"to enable join predictions. Use client-generated IDs instead."
                )


class BaseSQLModelFeature(  # pyright: ignore[reportIncompatibleMethodOverride]
    SQLModel, BaseFeature, metaclass=SQLModelFeatureMeta, spec=None
):  # type: ignore[misc]
    """Base class for features that are also SQLModel tables.

    Use this as a base class when you want to create Metaxy features
    that are also SQLAlchemy/SQLModel ORM models.

    This class combines:
    - Metaxy's Feature functionality (versioning, dependency tracking)
    - SQLModel's ORM functionality (database mapping, queries)

    Note: Unlike regular Feature classes, SQLModelFeature instances are mutable
    to support SQLModel ORM operations. Only the spec and graph class attributes
    are used from Feature, not the instance behavior.

    System-managed fields are defined as optional here and will be populated
    by the metadata store when reading/writing data.

    Example:
        ```py
        from metaxy.integrations.sqlmodel import SQLModelFeature
        from metaxy import BaseFeatureSpec, FeatureKey, FieldSpec, FieldKey
        from sqlmodel import Field

        class VideoFeature(
            SQLModelFeature,
            table=True,
            spec=BaseFeatureSpec(
                key=FeatureKey(["video"]),
                  # Root feature
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
    metaxy_provenance_by_field: str | None = Field(
        default=None,
        sa_type=JSON,
        sa_column_kwargs={"name": "provenance_by_field", "nullable": True},
    )

    metaxy_feature_version: str | None = Field(
        default=None, sa_column_kwargs={"name": "feature_version", "nullable": True}
    )

    metaxy_feature_spec_version: str | None = Field(
        default=None,
        sa_column_kwargs={"name": "feature_spec_version", "nullable": True},
    )

    metaxy_snapshot_version: str | None = Field(
        default=None, sa_column_kwargs={"name": "snapshot_version", "nullable": True}
    )
