from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    overload,
    runtime_checkable,
)

import pydantic
from pydantic import BeforeValidator
from pydantic.types import JsonValue
from typing_extensions import Self

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.field import CoersibleToFieldSpecsTypeAdapter, FieldSpec
from metaxy.models.fields_mapping import FieldsMapping
from metaxy.models.lineage import LineageRelationship
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
    """Protocol for FeatureSpec instances."""

    key: FeatureKey
    deps: list[Any] | None
    fields: list[FieldSpec]


@runtime_checkable
class FeatureClassProtocol(Protocol):
    """Protocol for BaseFeature classes."""

    @classmethod
    def spec(cls) -> FeatureSpecProtocol: ...


def _validate_feature_dep_feature(value: Any) -> FeatureKey:
    """Coerce various input types to FeatureKey for FeatureDep."""
    if isinstance(value, FeatureKey):
        return value
    # Check if it's a FeatureSpec instance (using Protocol)
    elif isinstance(value, FeatureSpecProtocol):
        return value.key
    # Check if it's a Feature class (using Protocol for runtime check)
    elif isinstance(value, type) and hasattr(value, "spec"):
        return value.spec().key
    else:
        # Must be a CoercibleToFeatureKey (str or list of str)
        return FeatureKeyAdapter.validate_python(value)


class FeatureDep(pydantic.BaseModel):
    """Feature dependency specification with optional column selection and renaming.

    Attributes:
        key: The feature key to depend on. Accepts string ("a/b/c"), list (["a", "b", "c"]),
            or FeatureKey instance.
        columns: Optional tuple of column names to select from upstream feature.
            - None (default): Keep all columns from upstream
            - Empty tuple (): Keep only system columns (sample_uid, provenance_by_field, etc.)
            - Tuple of names: Keep only specified columns (plus system columns)
        rename: Optional mapping of old column names to new names.
            Applied after column selection.
        fields_mapping: Optional field mapping configuration for automatic field dependency resolution.
            When provided, fields without explicit deps will automatically map to matching upstream fields.
            Defaults to using `[FieldsMapping.default()][metaxy.models.fields_mapping.DefaultFieldsMapping]`.

    Examples:
        ```py
        # Keep all columns with default field mapping
        FeatureDep(feature="upstream")

        # Keep all columns with suffix matching
        FeatureDep(feature="upstream", fields_mapping=FieldsMapping.default(match_suffix=True))

        # Keep all columns with all fields mapping
        FeatureDep(feature="upstream", fields_mapping=FieldsMapping.all())

        # Keep only specific columns
        FeatureDep(
            feature="upstream/feature",
            columns=("col1", "col2")
        )

        # Rename columns to avoid conflicts
        FeatureDep(
            feature="upstream/feature",
            rename={"old_name": "new_name"}
        )

        # Select and rename
        FeatureDep(
            feature="upstream/feature",
            columns=("col1", "col2"),
            rename={"col1": "upstream_col1"}
        )
        ```
    """

    feature: Annotated[FeatureKey, BeforeValidator(_validate_feature_dep_feature)]
    columns: tuple[str, ...] | None = (
        None  # None = all columns, () = only system columns
    )
    rename: dict[str, str] | None = None  # Column renaming mapping
    fields_mapping: FieldsMapping = pydantic.Field(
        default_factory=FieldsMapping.default
    )

    if TYPE_CHECKING:

        def __init__(  # pyright: ignore[reportMissingSuperCall]
            self,
            *,
            feature: str
            | Sequence[str]
            | FeatureKey
            | FeatureSpecProtocol
            | type[BaseFeature],
            columns: tuple[str, ...] | None = None,
            rename: dict[str, str] | None = None,
            fields_mapping: FieldsMapping | None = None,
        ) -> None: ...  # pyright: ignore[reportMissingSuperCall]

    def table_name(self) -> str:
        """Get SQL-like table name for this feature spec."""
        return self.feature.table_name


IDColumns: TypeAlias = Sequence[
    str
]  # non-bound, should be used for feature specs with arbitrary id columns

CoercibleToFeatureDep: TypeAlias = (
    FeatureDep
    | type["BaseFeature"]
    | str
    | Sequence[str]
    | FeatureKey
    | FeatureSpecProtocol
)


def _validate_id_columns(value: Any) -> tuple[str, ...]:
    """Coerce id_columns to tuple."""
    if isinstance(value, tuple):
        return value
    return tuple(value)


def _validate_deps(value: Any) -> list[FeatureDep]:
    """Coerce deps list, converting Feature classes to FeatureDep instances."""
    # Import here to avoid circular dependency at module level
    from metaxy.models.feature import BaseFeature

    if not isinstance(value, list):
        value = list(value) if hasattr(value, "__iter__") else [value]

    result = []
    for item in value:
        if isinstance(item, FeatureDep):
            # Already a FeatureDep, keep as-is
            result.append(item)
        elif isinstance(item, dict):
            # It's a dict (from deserialization), let Pydantic construct FeatureDep from it
            result.append(FeatureDep.model_validate(item))
        elif isinstance(item, type) and issubclass(item, BaseFeature):
            # It's a Feature class, convert to FeatureDep
            result.append(FeatureDep(feature=item))
        else:
            # Try to construct FeatureDep from the item (handles FeatureSpec, etc.)
            result.append(FeatureDep(feature=item))

    return result


class FeatureSpec(FrozenBaseModel):
    key: Annotated[FeatureKey, BeforeValidator(FeatureKeyAdapter.validate_python)]
    id_columns: Annotated[tuple[str, ...], BeforeValidator(_validate_id_columns)] = (
        pydantic.Field(
            ...,
            description="Columns that uniquely identify a sample in this feature.",
        )
    )
    deps: Annotated[list[FeatureDep], BeforeValidator(_validate_deps)] = pydantic.Field(
        default_factory=list
    )
    fields: Annotated[
        list[FieldSpec],
        BeforeValidator(CoersibleToFieldSpecsTypeAdapter.validate_python),
    ] = pydantic.Field(
        default_factory=lambda: [
            FieldSpec(
                key=FieldKey(["default"]),
            )
        ],
    )
    lineage: LineageRelationship = pydantic.Field(
        default_factory=LineageRelationship.identity,
        description="Lineage relationship of this feature.",
    )
    metadata: dict[str, JsonValue] = pydantic.Field(
        default_factory=dict,
        description="Metadata attached to this feature.",
    )

    if TYPE_CHECKING:
        # Overload for common case: list of FeatureDep instances
        @overload
        def __init__(
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns,
            deps: list[FeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            lineage: LineageRelationship | None = None,
            metadata: Mapping[str, JsonValue] | None = None,
            **kwargs: Any,
        ) -> None: ...

        # Overload for flexible case: list of coercible types
        @overload
        def __init__(
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns,
            deps: list[CoercibleToFeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            lineage: LineageRelationship | None = None,
            metadata: Mapping[str, JsonValue] | None = None,
            **kwargs: Any,
        ) -> None: ...

        # Implementation signature
        def __init__(  # pyright: ignore[reportMissingSuperCall]
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns,
            deps: list[FeatureDep] | list[CoercibleToFeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            lineage: LineageRelationship | None = None,
            metadata: Mapping[str, JsonValue] | None = None,
            **kwargs: Any,
        ) -> None: ...  # pyright: ignore[reportMissingSuperCall]

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
    def validate_unique_field_keys(self) -> Self:
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
    def validate_id_columns(self) -> Self:
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
            ```py
            spec = FeatureSpec(
                key=FeatureKey(["my", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]))],
            )
            spec.feature_spec_version
            # 'abc123...'  # 64-character hex string
            ```
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


FeatureSpecWithIDColumns: TypeAlias = FeatureSpec


DefaultFeatureCols: TypeAlias = tuple[Literal["sample_uid"],]


TestingUIDCols: TypeAlias = list[str]

CoercibleToFieldSpec: TypeAlias = str | FieldSpec


def _validate_sample_feature_spec_id_columns(
    value: Any,
) -> list[str]:
    """Coerce id_columns to list for SampleFeatureSpec."""
    if value is None:
        return ["sample_uid"]
    if isinstance(value, list):
        return value
    return list(value)


class SampleFeatureSpec(FeatureSpec):
    """A testing implementation of FeatureSpec that has a `sample_uid` ID column. Has to be moved to tests."""

    id_columns: Annotated[  # pyright: ignore[reportIncompatibleVariableOverride]
        pydantic.SkipValidation[list[str]],
        BeforeValidator(_validate_sample_feature_spec_id_columns),
    ] = pydantic.Field(
        default_factory=lambda: ["sample_uid"],
        description="List of columns that uniquely identify a row. They will be used by Metaxy in joins.",
    )

    if TYPE_CHECKING:
        # Overload for common case: list of FeatureDep instances
        @overload
        def __init__(
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns | None = None,
            deps: list[FeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            metadata: Mapping[str, JsonValue] | None = None,
            **kwargs: Any,
        ) -> None: ...

        @overload
        def __init__(
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns | None = None,
            deps: list[CoercibleToFeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            metadata: Mapping[str, JsonValue] | None = None,
            **kwargs: Any,
        ) -> None: ...

        # Implementation signature
        def __init__(  # pyright: ignore[reportMissingSuperCall]
            self,
            *,
            key: CoercibleToFeatureKey,
            id_columns: IDColumns | None = None,
            deps: list[FeatureDep] | list[CoercibleToFeatureDep] | None = None,
            fields: Sequence[str | FieldSpec] | None = None,
            metadata: Mapping[str, JsonValue] | None = None,
            **kwargs: Any,
        ) -> None: ...  # pyright: ignore[reportMissingSuperCall]
