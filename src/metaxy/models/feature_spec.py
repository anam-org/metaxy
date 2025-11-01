from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
    runtime_checkable,
)

import pydantic
from pydantic import BeforeValidator
from pydantic.types import JsonValue
from typing_extensions import Self

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.field import FieldSpec, SpecialFieldDep
from metaxy.models.types import (
    CoercibleToFeatureKey,
    FeatureKey,
    FeatureKeyAdapter,
    FieldKey,
)
from metaxy.utils.hashing import truncate_hash

if TYPE_CHECKING:
    # yes, these are circular imports, the TYPE_CHECKING block hides them at runtime.
    # neither pyright not basedpyright allow ignoring `reportImportCycles` because they think it's a bad practice
    # and it would be very smart to force the user to restructure their project instead
    # context: https://github.com/microsoft/pyright/issues/1825
    # however, considering the recursive nature of graphs, and the syntactic sugar that we want to support,
    # I decided to just put these errors into `.basedpyright/baseline.json` (after ensuring this is the only error produced by basedpyright)
    from metaxy.models.feature import BaseFeature


# Runtime-checkable protocols for type checking without circular imports
@runtime_checkable
class FeatureSpecProtocol(Protocol):
    """Protocol for BaseFeatureSpec instances."""

    key: FeatureKey
    deps: list[Any] | None
    fields: list[FieldSpec]


@runtime_checkable
class FeatureClassProtocol(Protocol):
    """Protocol for BaseFeature classes."""

    @classmethod
    def spec(cls) -> FeatureSpecProtocol: ...


class FeatureDep(pydantic.BaseModel):
    """Feature dependency specification with optional column selection and renaming.

    Attributes:
        key: The feature key to depend on. Accepts string ("a/b/c"), list (["a", "b", "c"]),
            or FeatureKey instance.
        columns: Optional tuple of column names to select from upstream feature.
            - None (default): Keep all columns from upstream
            - Empty tuple (): Keep only system columns (sample_uid, data_version, etc.)
            - Tuple of names: Keep only specified columns (plus system columns)
        rename: Optional mapping of old column names to new names.
            Applied after column selection.

    Examples:
        >>> # Keep all columns (string key format)
        >>> FeatureDep(feature="upstream")

        >>> # Keep all columns (list format)
        >>> FeatureDep(feature=["upstream"])

        >>> # Keep all columns (FeatureKey instance)
        >>> FeatureDep(feature=FeatureKey(["upstream"]))

        >>> # Keep only specific columns
        >>> FeatureDep(
        ...     key="upstream/feature",
        ...     columns=("col1", "col2")
        ... )

        >>> # Rename columns to avoid conflicts
        >>> FeatureDep(
        ...     key="upstream/feature",
        ...     rename={"old_name": "new_name"}
        ... )

        >>> # Select and rename
        >>> FeatureDep(
        ...     key="upstream/feature",
        ...     columns=("col1", "col2"),
        ...     rename={"col1": "upstream_col1"}
        ... )
    """

    feature: Annotated[FeatureKey, BeforeValidator(FeatureKeyAdapter.validate_python)]
    columns: tuple[str, ...] | None = (
        None  # None = all columns, () = only system columns
    )
    rename: dict[str, str] | None = None  # Column renaming mapping

    @overload
    def __init__(
        self,
        *,
        feature: str,
        columns: tuple[str, ...] | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        """Initialize from string key."""
        ...

    @overload
    def __init__(
        self,
        *,
        feature: Sequence[str],
        columns: tuple[str, ...] | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(
        self,
        *,
        feature: FeatureKey,
        columns: tuple[str, ...] | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        """Initialize from FeatureKey instance."""
        ...

    @overload
    def __init__(
        self,
        *,
        feature: FeatureSpecProtocol,
        columns: tuple[str, ...] | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        """Initialize from BaseFeatureSpec instance."""
        ...

    @overload
    def __init__(
        self,
        *,
        feature: type[BaseFeature[Any]],
        columns: tuple[str, ...] | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        """Initialize from BaseFeature class."""
        ...

    def __init__(
        self,
        *,
        feature: CoercibleToFeatureKey | FeatureSpecProtocol | type[BaseFeature[Any]],
        columns: tuple[str, ...] | None = None,
        rename: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Handle different key types with proper type checking
        resolved_key: FeatureKey

        # Check if it's a BaseFeatureSpec instance (using Protocol)
        if isinstance(feature, FeatureSpecProtocol):
            resolved_key = feature.key
        # Check if it's a Feature class (using Protocol for runtime check)
        elif isinstance(feature, type) and hasattr(feature, "spec"):
            resolved_key = feature.spec().key
        # Check if it's already a FeatureKey
        elif isinstance(feature, FeatureKey):
            resolved_key = feature
        else:
            # Must be a CoercibleToFeatureKey (str or list of str)
            resolved_key = FeatureKeyAdapter.validate_python(feature)

        super().__init__(
            feature=resolved_key,
            columns=columns,
            rename=rename,
            **kwargs,
        )

    def table_name(self) -> str:
        """Get SQL-like table name for this feature spec."""
        return self.feature.table_name


IDColumns: TypeAlias = Sequence[
    str
]  # non-bound, should be used for feature specs with arbitrary id columns
IDColumnsT = TypeVar(
    "IDColumnsT", bound=IDColumns, covariant=True
)  # bound, should be used for generic


class _BaseFeatureSpec(FrozenBaseModel):
    key: Annotated[FeatureKey, BeforeValidator(FeatureKeyAdapter.validate_python)]
    deps: list[FeatureDep] | None = None
    fields: list[FieldSpec] = pydantic.Field(
        default_factory=lambda: [
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=SpecialFieldDep.ALL,
            )
        ]
    )
    metadata: dict[str, JsonValue] = pydantic.Field(
        default_factory=dict,
        description="Metadata attached to this feature.",
    )


class BaseFeatureSpec(_BaseFeatureSpec, Generic[IDColumnsT]):
    id_columns: pydantic.SkipValidation[IDColumnsT]

    @overload
    def __init__(
        self,
        key: str,
        *,
        deps: list[FeatureDep] | None = None,
        fields: list[FieldSpec] | None = None,
        id_columns: list[str] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Initialize from string key."""
        ...

    @overload
    def __init__(
        self,
        key: Sequence[str],
        *,
        deps: list[FeatureDep] | None = None,
        fields: list[FieldSpec] | None = None,
        id_columns: list[str] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(
        self,
        key: FeatureKey,
        *,
        deps: list[FeatureDep] | None = None,
        fields: list[FieldSpec] | None = None,
        id_columns: list[str] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Initialize from FeatureKey instance."""
        ...

    @overload
    def __init__(
        self,
        key: Self,
        *,
        deps: list[FeatureDep] | None = None,
        fields: list[FieldSpec] | None = None,
        id_columns: list[str] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Initialize from BaseFeatureSpec instance."""
        ...

    def __init__(self, key: CoercibleToFeatureKey | Self, **kwargs: Any):
        if isinstance(key, type(self)):
            key = key.key
        else:
            key = FeatureKeyAdapter.validate_python(key)

        assert isinstance(key, FeatureKey)

        super().__init__(key=key, **kwargs)

    @cached_property
    def fields_by_key(self) -> Mapping[FieldKey, FieldSpec]:
        return {c.key: c for c in self.fields}

    @cached_property
    def code_version(self) -> str:
        """Hash of this feature's field code_versions only (no dependencies)."""
        hasher = hashlib.sha256()

        # Sort fields by key for deterministic ordering
        sorted_fields = sorted(self.fields, key=lambda field: field.key.to_string())

        for field in sorted_fields:
            hasher.update(field.key.to_string().encode("utf-8"))
            hasher.update(str(field.code_version).encode("utf-8"))

        return truncate_hash(hasher.hexdigest())

    def table_name(self) -> str:
        """Get SQL-like table name for this feature spec."""
        return self.key.table_name

    @pydantic.model_validator(mode="after")
    def validate_unique_field_keys(self) -> BaseFeatureSpec[IDColumnsT]:
        """Validate that all fields have unique keys."""
        seen_keys: set[tuple[str, ...]] = set()
        for field in self.fields:
            # Convert to tuple for hashability in case it's a plain list
            key_tuple = tuple(field.key)
            if key_tuple in seen_keys:
                raise ValueError(
                    f"Duplicate field key found: {field.key}. "
                    f"All fields must have unique keys."
                )
            seen_keys.add(key_tuple)
        return self

    @pydantic.model_validator(mode="after")
    def validate_id_columns(self) -> BaseFeatureSpec[IDColumnsT]:
        """Validate that id_columns is non-empty if specified."""
        if self.id_columns is not None and len(self.id_columns) == 0:
            raise ValueError(
                "id_columns must be non-empty if specified. Use None for default."
            )
        return self

    @property
    def feature_spec_version(self) -> str:
        """Compute SHA256 hash of the complete feature specification.

        This property provides a deterministic hash of ALL specification properties,
        including key, deps, fields, and any metadata/tags.
        Used for audit trail and tracking specification changes.

        Unlike feature_version which only hashes computational properties
        (for migration triggering), feature_spec_version captures the entire specification
        for complete reproducibility and audit purposes.

        Returns:
            SHA256 hex digest of the specification

        Example:
            >>> spec = FeatureSpec(
            ...     key=FeatureKey(["my", "feature"]),
            ...
            ...     fields=[FieldSpec(key=FieldKey(["default"]))],
            ... )
            >>> spec.feature_spec_version
            'abc123...'  # 64-character hex string
        """
        # Use model_dump with mode="json" for deterministic serialization
        # This ensures all types (like FeatureKey) are properly serialized
        spec_dict = self.model_dump(mode="json")

        # Sort keys to ensure deterministic ordering
        spec_json = json.dumps(spec_dict, sort_keys=True)

        # Compute SHA256 hash
        hasher = hashlib.sha256()
        hasher.update(spec_json.encode("utf-8"))

        return hasher.hexdigest()


BaseFeatureSpecWithIDColumns: TypeAlias = BaseFeatureSpec[IDColumns]


DefaultFeatureCols: TypeAlias = tuple[Literal["sample_uid"],]


TestingUIDCols: TypeAlias = list[str]


class FeatureSpec(BaseFeatureSpec[DefaultFeatureCols]):
    """A default concrete implementation of BaseFeatureSpec that has a `sample_uid` ID column."""

    id_columns: DefaultFeatureCols = pydantic.Field(
        default=("sample_uid",),
        description="List of columns that uniquely identify a row. They will be used by Metaxy in joins.",
    )

    @overload
    def __init__(
        self,
        key: str,
        *,
        deps: list[FeatureDep] | None = None,
        fields: list[FieldSpec] | None = None,
        id_columns: Sequence[str] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Initialize from string key."""
        ...

    @overload
    def __init__(
        self,
        key: Sequence[str],
        *,
        deps: list[FeatureDep] | None = None,
        fields: list[FieldSpec] | None = None,
        id_columns: Sequence[str] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(
        self,
        key: FeatureKey,
        *,
        deps: list[FeatureDep] | None = None,
        fields: list[FieldSpec] | None = None,
        id_columns: Sequence[str] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Initialize from FeatureKey instance."""
        ...

    @overload
    def __init__(
        self,
        key: Self,
        *,
        deps: list[FeatureDep] | None = None,
        fields: list[FieldSpec] | None = None,
        id_columns: Sequence[str] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Initialize from FeatureSpec instance."""
        ...

    def __init__(self, key: CoercibleToFeatureKey | Self, **kwargs: Any):
        # id_columns is always set for FeatureSpec
        super().__init__(key=key, **kwargs)


class TestingFeatureSpec(BaseFeatureSpec[TestingUIDCols]):
    """A testing concrete implementation of BaseFeatureSpec that has a `sample_uid` ID column."""

    id_columns: TestingUIDCols = pydantic.Field(
        default=["sample_uid"],
        description="List of columns that uniquely identify a row. They will be used by Metaxy in joins.",
    )
