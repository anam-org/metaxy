"""SQLModel integration for Metaxy.

This module provides a combined metaclass that allows Metaxy Feature classes
to also be SQLModel table classes, enabling seamless integration with SQLAlchemy/SQLModel ORMs.
"""

from typing import TYPE_CHECKING, Any

from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelMetaclass

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
        mcs: type[Any],
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *,
        spec: BaseFeatureSpecWithIDColumns | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a new combined SQLModel/Metaxy Feature class.

        This method handles:
        1. Automatic __tablename__ generation from feature key
        2. Validation that the feature has a spec
        3. Creation of both SQLModel table and Metaxy feature

        Args:
            name: Class name
            bases: Base classes
            namespace: Class namespace (attributes and methods)
            spec: Metaxy BaseFeatureSpec (required for concrete features)
            **kwargs: Additional keyword arguments (e.g., table=True for SQLModel)

        Returns:
            New class that is both a SQLModel table and Metaxy feature
        """
        # spec is already in kwargs, no need to extract again

        # Auto-generate __tablename__ if creating a table and not provided
        # Check config to see if infer_db_table_names is enabled
        if kwargs.get("table") and "__tablename__" not in namespace and spec:
            from metaxy.config import MetaxyConfig

            # Get config (may not be set, so use default if not available)
            try:
                config = MetaxyConfig.get()
                infer_db_table_names = config.ext.sqlmodel.infer_db_table_names
            except Exception:
                # If config not available or not set, default to True
                infer_db_table_names = True

            if infer_db_table_names:
                # Convert feature key to table name (using double underscores)
                table_name = "__".join(spec.key.parts)
                namespace["__tablename__"] = table_name

        # Call super().__new__ which follows MRO: MetaxyMeta -> SQLModelMetaclass -> ...
        # MetaxyMeta will consume the spec parameter and pass remaining kwargs to SQLModelMetaclass
        # Note: super().__new__ in metaclass context implicitly receives the metaclass
        new_class = super().__new__(
            mcs,
            name,  # pyright: ignore[reportCallIssue]
            bases,
            namespace,
            spec=spec,
            **kwargs,  # type: ignore[misc]
        )

        # After class creation, validate id_columns are not server-defined
        if kwargs.get("table") and spec is not None:
            mcs._validate_id_columns_not_server_defined(new_class, spec)

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

    Combines Metaxy Feature functionality with SQLModel ORM capabilities.
    Classes inheriting from this can be used both as Metaxy features and SQLAlchemy tables.

    Key features:
    - Automatic table name generation from feature key
    - Built-in Metaxy metadata columns (data_version, feature_version, etc.)
    - Full SQLModel ORM functionality (queries, relationships, etc.)
    - All Metaxy Feature methods (load_input, data_version, etc.)

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
