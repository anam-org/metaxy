"""SQLModel integration for Metaxy.

This module provides a combined metaclass that allows Metaxy Feature classes
to also be SQLModel table classes, enabling seamless integration with SQLAlchemy/SQLModel ORMs.
"""

from typing import Any

from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelMetaclass

from metaxy.config import MetaxyConfig
from metaxy.models.feature import Feature, MetaxyMeta
from metaxy.models.feature_spec import FeatureSpec


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
        >>> from metaxy.integrations.sqlmodel import SQLModelFeature
        >>> from metaxy import FeatureSpec, FeatureKey, FieldSpec, FieldKey
        >>> from sqlmodel import Field
        >>>
        >>> class MyFeature(
        ...     SQLModelFeature,
        ...     table=True,
        ...     spec=FeatureSpec(
        ...         key=FeatureKey(["my", "feature"]),
        ...         deps=None,
        ...         fields=[
        ...             FieldSpec(
        ...                 key=FieldKey(["data"]),
        ...                 code_version=1,
        ...             ),
        ...         ],
        ...     ),
        ... ):
        ...     __tablename__ = "my_feature"
        ...
        ...     uid: str = Field(primary_key=True)
        ...     data: str
    """

    def __new__(
        cls,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *,
        spec: FeatureSpec | None = None,
        **kwargs: Any,
    ) -> type[Any]:
        """Create a new SQLModel + Feature class.

        Args:
            cls_name: Name of the class being created
            bases: Base classes
            namespace: Class namespace (attributes and methods)
            spec: Metaxy FeatureSpec (required for concrete features)
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
        return super().__new__(cls, cls_name, bases, namespace, spec=spec, **kwargs)


# pyright: reportIncompatibleMethodOverride=false, reportIncompatibleVariableOverride=false
class SQLModelFeature(SQLModel, Feature, metaclass=SQLModelFeatureMeta, spec=None):  # type: ignore[misc]
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
        >>> from metaxy.integrations.sqlmodel import SQLModelFeature
        >>> from metaxy import FeatureSpec, FeatureKey, FieldSpec, FieldKey
        >>> from sqlmodel import Field
        >>>
        >>> class VideoFeature(
        ...     SQLModelFeature,
        ...     table=True,
        ...     spec=FeatureSpec(
        ...         key=FeatureKey(["video"]),
        ...         deps=None,  # Root feature
        ...         fields=[
        ...             FieldSpec(
        ...                 key=FieldKey(["video_file"]),
        ...                 code_version=1,
        ...             ),
        ...         ],
        ...     ),
        ... ):
        ...
        ...     uid: str = Field(primary_key=True)
        ...     path: str
        ...     duration: float
        ...
        ...     # Now you can use both Metaxy and SQLModel features:
        ...     # - VideoFeature.feature_version() -> Metaxy versioning
        ...     # - session.exec(select(VideoFeature)) -> SQLModel queries
    """

    # Override the frozen config from Feature's FrozenBaseModel
    # SQLModel instances need to be mutable for ORM operations
    model_config = {"frozen": False}  # pyright: ignore[reportAssignmentType]

    # System-managed metadata fields - these are optional and populated by Metaxy
    # The metadata store will populate these when reading/writing data
    # Users can override these definitions in subclasses if they need different constraints
    sample_uid: str | None = Field(default=None, nullable=True)

    # Using sa_column_kwargs to map to the actual column names used by Metaxy
    metaxy_data_version: str | None = Field(
        default=None,
        sa_type=JSON,
        sa_column_kwargs={"name": "data_version", "nullable": True},
    )

    metaxy_feature_version: str | None = Field(
        default=None, sa_column_kwargs={"name": "feature_version", "nullable": True}
    )

    metaxy_snapshot_version: str | None = Field(
        default=None, sa_column_kwargs={"name": "snapshot_version", "nullable": True}
    )

    # All Feature class methods and attributes are inherited from Feature base class:
    # - spec: ClassVar[FeatureSpec]
    # - graph: ClassVar[FeatureGraph]
    # - feature_version() -> str
    # - data_version() -> dict[str, str]
    # - load_input(...)
    # - resolve_data_version_diff(...)
