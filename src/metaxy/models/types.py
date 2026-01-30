"""Type definitions for metaxy models."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    NamedTuple,
    TypeAlias,
    overload,
)

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

from metaxy._decorators import public

if TYPE_CHECKING:
    from typing_extensions import Self

KEY_SEPARATOR = "/"

# backcompat
FEATURE_KEY_SEPARATOR = KEY_SEPARATOR
FIELD_KEY_SEPARATOR = KEY_SEPARATOR


@public
class SnapshotPushResult(NamedTuple):
    """Result of recording a feature graph snapshot.

    Attributes:
        snapshot_version: The deterministic hash of the graph snapshot
        already_pushed: True if this snapshot_version was already pushed previously
        updated_features: List of feature keys with updated information (changed definition_version)
    """

    snapshot_version: str
    already_pushed: bool
    updated_features: list[str]


_CoercibleToKey: TypeAlias = Sequence[str] | str


class _Key(RootModel[tuple[str, ...]]):
    """
    A common class for key-like objects that contain a sequence of string parts.

    Parts cannot contain forward slashes (/) or double underscores (__).

    Args:
        key: Feature key as string ("a/b/c"), sequence (["a", "b", "c"]), or FeatureKey instance.
             String format is split on "/" separator.
    """

    model_config = ConfigDict(
        frozen=True,
        repr=False,  # ty: ignore[invalid-key]
    )  # Make immutable for hashability, use custom __repr__

    root: tuple[str, ...]

    if TYPE_CHECKING:

        @overload
        def __init__(self, parts: str) -> None: ...

        @overload
        def __init__(self, parts: Sequence[str]) -> None: ...

        @overload
        def __init__(self, parts: Self) -> None: ...

        def __init__(
            self,
            parts: str | Sequence[str] | Self,
        ) -> None: ...

    @model_validator(mode="before")
    @classmethod
    def _validate_input(cls, data: Any) -> Any:
        """Convert various input types to tuple of strings."""
        # If it's already a tuple, validate and return it
        if isinstance(data, tuple):
            return data

        # Handle dict input (from Pydantic deserialization)
        if isinstance(data, dict):
            # RootModel deserialization passes dict with "root" key
            if "root" in data:
                root_value = data["root"]
                if isinstance(root_value, tuple):
                    return root_value
                elif isinstance(root_value, (list, Sequence)):
                    return tuple(root_value)
            # Legacy "parts" key for backward compatibility
            elif "parts" in data:
                parts = data["parts"]
                if isinstance(parts, (list, tuple)) and parts and isinstance(parts[0], dict):
                    # Handle incorrectly nested structure like {'parts': [{'parts': [...]}]}
                    if "parts" in parts[0]:
                        return tuple(parts[0]["parts"])
                    else:
                        raise ValueError(f"Invalid nested structure in parts: {parts}")
                return tuple(parts) if not isinstance(parts, tuple) else parts
            # Empty dict
            raise ValueError("Dict must contain 'root' or 'parts' key")

        # Handle string input - split on separator
        if isinstance(data, str):
            return tuple(data.split(KEY_SEPARATOR))

        # Handle instance of same class - extract root
        if isinstance(data, cls):
            return data.root

        # Handle sequence (list, etc.)
        if isinstance(data, Sequence):
            return tuple(data)

        raise ValueError(f"Cannot create {cls.__name__} from {type(data).__name__}")

    @field_validator("root", mode="after")
    @classmethod
    def _validate_root_content(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Validate that parts follow naming conventions.

        Rules:
        - First part must start with a lowercase letter (a-z) or underscore (_)
        - All parts can contain only lowercase letters, digits, underscores, or hyphens
        - No part can contain forward slashes (/) or double underscores (__)

        These rules ensure compatibility with SQL column naming conventions
        across databases (PostgreSQL, DuckDB, etc.) without requiring quoting.
        """
        import re

        # Pattern for first part: must start with lowercase letter or underscore
        first_part_pattern = re.compile(r"^[a-z_][a-z0-9_-]*$")
        # Pattern for subsequent parts: can start with letter, digit, or underscore
        other_part_pattern = re.compile(r"^[a-z0-9_][a-z0-9_-]*$")

        for i, part in enumerate(value):
            if not isinstance(part, str):
                raise ValueError(f"{cls.__name__} parts must be strings, got {type(part).__name__}")
            if not part:
                raise ValueError(f"{cls.__name__} parts cannot be empty strings")
            if "/" in part:
                raise ValueError(
                    f"{cls.__name__} part '{part}' cannot contain forward slashes (/). "
                    f"Forward slashes are reserved as the separator in to_string(). "
                    f"Use underscores or hyphens instead."
                )
            if "__" in part:
                raise ValueError(
                    f"{cls.__name__} part '{part}' cannot contain double underscores (__). "
                    f"Use single underscores or hyphens instead."
                )

            # First part must start with letter or underscore
            if i == 0:
                if not first_part_pattern.match(part):
                    raise ValueError(
                        f"{cls.__name__} first part '{part}' is invalid. "
                        f"The first part must start with a lowercase letter (a-z) or underscore (_), "
                        f"and contain only lowercase letters, digits, underscores, or hyphens."
                    )
            else:
                if not other_part_pattern.match(part):
                    raise ValueError(
                        f"{cls.__name__} part '{part}' is invalid. "
                        f"Parts must contain only lowercase letters, digits, underscores, or hyphens, "
                        f"and cannot start with a hyphen."
                    )
        return value

    @model_serializer(mode="plain")
    def _serialize_model(self) -> str:
        """Serialize to string format for JSON compatibility.

        Keys are serialized as strings (e.g., "a/b/c") so they can be used
        as dictionary keys in JSON, which requires string keys.
        """
        return self.to_string()

    @property
    def parts(self) -> tuple[str, ...]:
        """Backward compatibility property for accessing root as parts."""
        return self.root

    def to_string(self) -> str:
        """Convert to string representation with "/" separator."""
        return KEY_SEPARATOR.join(self.parts)

    def to_struct_key(self) -> str:
        """Convert to a name that can be used as struct key in databases"""
        return "_".join(self.parts)

    def to_column_suffix(self) -> str:
        """Convert to a suffix usable for database column names (typically temporary)."""
        return "__" + "_".join(self.parts)

    def __repr__(self) -> str:
        """Return string representation."""
        return self.to_string()

    def __str__(self) -> str:
        """Return string representation."""
        return self.to_string()

    def __lt__(self, other: Any) -> bool:
        """Less than comparison for sorting."""
        if isinstance(other, self.__class__):
            return self.parts < other.parts
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        """Less than or equal comparison for sorting."""
        if isinstance(other, self.__class__):
            return self.parts <= other.parts
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        """Greater than comparison for sorting."""
        if isinstance(other, self.__class__):
            return self.parts > other.parts
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        """Greater than or equal comparison for sorting."""
        if isinstance(other, self.__class__):
            return self.parts >= other.parts
        return NotImplemented

    def __iter__(self) -> Iterator[str]:  # ty: ignore[invalid-method-override]
        """Return iterator over parts."""
        return iter(self.parts)

    @property
    def table_name(self) -> str:
        """Get SQL-like table name for this feature key.

        Replaces hyphens with underscores for SQL compatibility.
        """
        return "__".join(part.replace("-", "_") for part in self.parts)

    # List-like interface for backward compatibility
    def __getitem__(self, index: int) -> str:
        """Get part by index."""
        return self.parts[index]

    def __len__(self) -> int:
        """Get number of parts."""
        return len(self.parts)

    def __contains__(self, item: str) -> bool:
        """Check if part is in key."""
        return item in self.parts

    def __reversed__(self):
        """Return reversed iterator over parts."""
        return reversed(self.parts)


# CoercibleToKey: TypeAlias = _CoercibleToKey | _Key


@public
class FeatureKey(_Key):
    """
    Feature key as a sequence of string parts.

    Hashable for use as dict keys in registries.
    Parts cannot contain forward slashes (/) or double underscores (__).

    Example:

        ```py
        FeatureKey("a/b/c")  # String format
        # FeatureKey(parts=['a', 'b', 'c'])

        FeatureKey(["a", "b", "c"])  # List format
        # FeatureKey(parts=['a', 'b', 'c'])

        FeatureKey(FeatureKey(["a", "b", "c"]))  # FeatureKey copy
        # FeatureKey(parts=['a', 'b', 'c'])
        ```
    """

    if TYPE_CHECKING:

        @overload
        def __init__(self, parts: str) -> None: ...

        @overload
        def __init__(self, parts: Sequence[str]) -> None: ...

        @overload
        def __init__(self, parts: FeatureKey) -> None: ...

        def __init__(
            self,
            parts: str | Sequence[str] | FeatureKey,
        ) -> None: ...

    def model_dump(self, **kwargs: Any) -> Any:
        """Serialize to string format for JSON dict key compatibility."""
        return self.to_string()

    def __hash__(self) -> int:
        """Return hash for use as dict keys."""
        return hash(self.parts)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another instance."""
        if isinstance(other, self.__class__):
            return self.parts == other.parts
        return super().__eq__(other)

    def to_column_suffix(self) -> str:
        """Convert to a suffix usable for database column names (typically temporary)."""
        return "__" + "_".join(self.parts)


@public
class FieldKey(_Key):
    """
    Field key as a sequence of string parts.

    Hashable for use as dict keys in registries.
    Parts cannot contain forward slashes (/) or double underscores (__).

    Example:

        ```py
        FieldKey("a/b/c")  # String format
        # FieldKey(parts=['a', 'b', 'c'])

        FieldKey(["a", "b", "c"])  # List format
        # FieldKey(parts=['a', 'b', 'c'])

        FieldKey(FieldKey(["a", "b", "c"]))  # FieldKey copy
        # FieldKey(parts=['a', 'b', 'c'])
        ```
    """

    if TYPE_CHECKING:

        @overload
        def __init__(self, parts: str) -> None: ...

        @overload
        def __init__(self, parts: Sequence[str]) -> None: ...

        @overload
        def __init__(self, parts: FieldKey) -> None: ...

        def __init__(
            self,
            parts: str | Sequence[str] | FieldKey,
        ) -> None: ...

    def model_dump(self, **kwargs: Any) -> Any:
        """Serialize to string format for JSON dict key compatibility."""
        return self.to_string()

    def __hash__(self) -> int:
        """Return hash for use as dict keys."""
        return hash(self.parts)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another instance."""
        if isinstance(other, self.__class__):
            return self.parts == other.parts
        return super().__eq__(other)


_CoercibleToFeatureKey: TypeAlias = _CoercibleToKey | FeatureKey

FeatureKeyAdapter = TypeAdapter(
    FeatureKey
)  # can call .validate_python() to transform acceptable types into a FeatureKey
FieldKeyAdapter = TypeAdapter(FieldKey)  # can call .validate_python() to transform acceptable types into a FieldKey


def _coerce_to_feature_key(value: Any) -> FeatureKey:
    """Convert various types to FeatureKey.

    Accepts:

    - slashed `str`: `"a/b/c"`

    - `Sequence[str]`: `["a", "b", "c"]`

    - `FeatureKey`: pass through

    - `type[BaseFeature]`: extracts .spec().key

    - `FeatureDefinition`: extracts .key

    Args:
        value: Value to coerce to `FeatureKey`

    Returns:
        canonical `FeatureKey` instance

    Raises:
        ValidationError: If value cannot be coerced to FeatureKey
    """
    if isinstance(value, FeatureKey):
        return value

    # Check if it's a FeatureDefinition
    from metaxy.models.feature_definition import FeatureDefinition

    if isinstance(value, FeatureDefinition):
        return value.key

    # Check if it's a BaseFeature class
    # Import here to avoid circular dependency at module level
    from metaxy.models.feature import BaseFeature

    if isinstance(value, type) and issubclass(value, BaseFeature):
        return value.spec().key

    # Handle str, Sequence[str]
    return FeatureKeyAdapter.validate_python(value)


def _coerce_to_field_key(value: Any) -> FieldKey:
    """Convert various types to FieldKey.

    Accepts:

        - slashed `str`: `"a/b/c"`

        - `Sequence[str]`: `["a", "b", "c"]`

        - `FieldKey`: pass through

    Args:
        value: Value to coerce to `FieldKey`

    Returns:
        canonical `FieldKey` instance

    Raises:
        ValidationError: If value cannot be coerced to `FieldKey`
    """
    if isinstance(value, FieldKey):
        return value

    # Handle str, Sequence[str]
    return FieldKeyAdapter.validate_python(value)


if TYPE_CHECKING:
    from metaxy.models.feature import BaseFeature
    from metaxy.models.feature_definition import FeatureDefinition

# Type unions - what inputs are accepted
# Note: FeatureDefinition is imported at runtime in _coerce_to_feature_key to avoid circular imports
CoercibleToFeatureKey: TypeAlias = "str | Sequence[str] | FeatureKey | type[BaseFeature] | FeatureDefinition"
"""Type alias for values that can be coerced to a [`FeatureKey`][metaxy.FeatureKey].

Accepted formats:

- `str`: Slash-separated string like `"raw/video"` or `"ml/embeddings/v2"`
- `Sequence[str]`: sequences of parts like `["user", "profile"]`
- [`FeatureKey`][metaxy.FeatureKey]: Pass through unchanged
- `type[BaseFeature]`: Any [`BaseFeature`][metaxy.BaseFeature] subclass - extracts its key via `.spec().key`
- [`FeatureDefinition`][metaxy.FeatureDefinition]: Extracts its key via `.key`

Example:
    ```python
    key1 = "raw/video"
    key2 = ["raw", "video"]
    key3 = mx.FeatureKey("raw/video")
    key4 = MyFeatureClass  # where MyFeatureClass is a BaseFeature subclass
    key5 = mx.FeatureDefinition("raw/video", ...)
    ```
"""

CoercibleToFieldKey: TypeAlias = str | Sequence[str] | FieldKey
"""Type alias for values that can be coerced to a [`FieldKey`][metaxy.FieldKey].

Accepted formats:

- `str`: Slash-separated string like `"audio/english"`
- `Sequence[str]`: sequence of parts like `["audio", "english"]`
- [`FieldKey`][metaxy.FieldKey]: Pass through unchanged

Example:
    ```python
    key1 = "audio/english"
    key2 = ["audio", "english"]
    key3 = mx.FieldKey("audio/english")
    ```
"""

# Annotated types for Pydantic field annotations - automatically validate
# After validation, these ARE FeatureKey/FieldKey (not unions)
ValidatedFeatureKey: TypeAlias = Annotated[
    FeatureKey,
    BeforeValidator(_coerce_to_feature_key),
    Field(
        description="Feature key. Accepts a slashed string ('a/b/c'), a sequence of strings, a FeatureKey instance, or a child class of BaseFeature"
    ),
]

ValidatedFieldKey: TypeAlias = Annotated[
    FieldKey,
    BeforeValidator(_coerce_to_field_key),
    Field(description="Field key. Accepts a slashed string ('a/b/c'), a sequence of strings, or a FieldKey instance."),
]

# TypeAdapters for non-Pydantic usage (e.g., in metadata_store/base.py)
ValidatedFeatureKeyAdapter: TypeAdapter[ValidatedFeatureKey] = TypeAdapter(ValidatedFeatureKey)
ValidatedFieldKeyAdapter: TypeAdapter[ValidatedFieldKey] = TypeAdapter(ValidatedFieldKey)


# Dagster-compatible version: validates FeatureKey semantics but stores as list[str]
# This is necessary because Dagster resolves types at import time and can't handle FeatureKey
def _coerce_to_feature_key_list(value: Any) -> list[str]:
    """Coerce to FeatureKey, validate, and return as list[str] for Dagster compatibility."""
    feature_key = _coerce_to_feature_key(value)
    return list(feature_key.parts)


ValidatedFeatureKeyList: TypeAlias = Annotated[
    list[str],
    BeforeValidator(_coerce_to_feature_key_list),
    Field(description="Feature key as list of strings (e.g., ['user', 'profile']). Validates using FeatureKey rules."),
]


# Collection types for common patterns - automatically validate sequences
# Pydantic will validate each element using ValidatedFeatureKey/ValidatedFieldKey
ValidatedFeatureKeySequence: TypeAlias = Annotated[
    Sequence[ValidatedFeatureKey],
    Field(description="Sequence items coerced into FeatureKey."),
]

ValidatedFieldKeySequence: TypeAlias = Annotated[
    Sequence[ValidatedFieldKey],
    Field(description="Sequence items coerced into FieldKey."),
]

# TypeAdapters for non-Pydantic usage
ValidatedFeatureKeySequenceAdapter: TypeAdapter[ValidatedFeatureKeySequence] = TypeAdapter(ValidatedFeatureKeySequence)
ValidatedFieldKeySequenceAdapter: TypeAdapter[ValidatedFieldKeySequence] = TypeAdapter(ValidatedFieldKeySequence)

FeatureDepMetadata: TypeAlias = dict[str, Any]
