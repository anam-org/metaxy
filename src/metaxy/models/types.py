"""Type definitions for metaxy models."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple, TypeAlias, overload

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

KEY_SEPARATOR = "/"

# backcompat
FEATURE_KEY_SEPARATOR = KEY_SEPARATOR
FIELD_KEY_SEPARATOR = KEY_SEPARATOR


class SnapshotPushResult(NamedTuple):
    """Result of recording a feature graph snapshot.

    Attributes:
        snapshot_version: The deterministic hash of the graph snapshot
        already_recorded: True if computational changes were already recorded
        metadata_changed: True if metadata-only changes were detected
        features_with_spec_changes: List of feature keys with spec version changes
    """

    snapshot_version: str
    already_recorded: bool
    metadata_changed: bool
    features_with_spec_changes: list[str]


_CoercibleToKey: TypeAlias = Sequence[str] | str


class _Key(BaseModel):
    """
    A common class for key-like objects that contain a sequence of string parts.

    Parts cannot contain forward slashes (/) or double underscores (__).

    Args:
        key: Feature key as string ("a/b/c"), sequence (["a", "b", "c"]), or FeatureKey instance.
             String format is split on "/" separator.
    """

    model_config = ConfigDict(frozen=True, repr=False)  # pyright: ignore[reportCallIssue]  # Make immutable for hashability, use custom __repr__

    parts: tuple[str, ...]

    @overload
    def __init__(self, key: str, /) -> None:
        """Initialize from string with "/" separator."""
        ...

    @overload
    def __init__(self, key: Sequence[str], /) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(self: Self, key: Self, /) -> None:
        """Initialize from another instance (copy)."""
        ...

    @overload
    def __init__(self, *parts: str) -> None:
        """Initialize from variadic string arguments."""
        ...

    @overload
    def __init__(self, *, parts: Sequence[str]) -> None:
        """Initialize from parts keyword argument."""
        ...

    def __init__(self, *args: str | _CoercibleToKey | Self, **kwargs: Any) -> None:
        """
        Initialize from various input types.

        Args:
            *args: Variadic positional arguments:
                - Single str: Split on "/" separator ("a/b/c" -> ["a", "b", "c"])
                - Single Sequence[str]: Use as parts (["a", "b", "c"])
                - Single Key instance: Copy parts
                - Multiple str: Use as parts ("a", "b", "c" -> ["a", "b", "c"])
            **kwargs: Additional keyword arguments for BaseModel (e.g., parts=...)

        Examples:
            >>> FeatureKey("a/b/c")  # String with separator
            >>> FeatureKey(["a", "b", "c"])  # List
            >>> FeatureKey("a", "b", "c")  # Variadic
            >>> FeatureKey(parts=["a", "b", "c"])  # Keyword argument
        """
        # Handle variadic or single argument construction
        if args:
            if len(args) == 1:
                key = args[0]
                # Single argument - could be str, sequence, or instance
                if isinstance(key, str):
                    kwargs["parts"] = tuple(key.split(KEY_SEPARATOR))
                elif isinstance(key, self.__class__):
                    kwargs["parts"] = key.parts
                elif isinstance(key, dict):
                    # Handle dict case (from Pydantic)
                    if "parts" in key:
                        parts_value = key["parts"]  # pyright: ignore[reportCallIssue, reportArgumentType]
                        kwargs["parts"] = (
                            tuple(parts_value)
                            if not isinstance(parts_value, tuple)
                            else parts_value
                        )
                    else:
                        raise ValueError("Dict must contain 'parts' key")
                elif isinstance(key, Sequence):
                    kwargs["parts"] = tuple(key)
                else:
                    raise ValueError(
                        f"Cannot create {self.__class__.__name__} from {type(key).__name__}"
                    )
            else:
                # Multiple arguments - treat as variadic parts
                # Validate all are strings
                if not all(isinstance(arg, str) for arg in args):
                    raise ValueError(
                        f"Variadic arguments to {self.__class__.__name__} must all be strings, "
                        f"got types: {[type(arg).__name__ for arg in args]}"
                    )
                kwargs["parts"] = tuple(args)  # type: ignore[arg-type]

        super().__init__(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def _validate_input(cls, data: Any) -> dict[str, Any]:
        """Convert various input types to dict with parts."""
        # If it's already a dict with parts, return it (normal Pydantic flow)

        if isinstance(data, dict):
            if "parts" in data:
                # Check if parts contains nested dicts (incorrect nesting from serialization)
                parts = data["parts"]
                if (
                    isinstance(parts, (list, tuple))
                    and parts
                    and isinstance(parts[0], dict)
                ):
                    # Handle incorrectly nested structure like {'parts': [{'parts': [...]}]}
                    # This can happen during certain deserialization paths
                    if "parts" in parts[0]:
                        data["parts"] = tuple(parts[0]["parts"])
                    else:
                        raise ValueError(f"Invalid nested structure in parts: {parts}")
                elif not isinstance(parts, tuple):
                    # Ensure parts is a tuple
                    data["parts"] = tuple(parts)
                return data
            # Handle dict without parts (shouldn't happen normally)
            return data
        # Handle different input types
        elif isinstance(data, str):
            parts = tuple(data.split(KEY_SEPARATOR))
        elif isinstance(data, cls):
            parts = data.parts
        elif isinstance(data, Sequence):
            parts = tuple(data)
        else:
            raise ValueError(f"Cannot create {cls.__name__} from {type(data).__name__}")

        return {"parts": parts}

    @field_validator("parts", mode="after")
    @classmethod
    def _validate_parts_content(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Validate that parts don't contain forbidden characters."""
        for part in value:
            if not isinstance(part, str):
                raise ValueError(
                    f"{cls.__name__} parts must be strings, got {type(part).__name__}"
                )
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
        return value

    @field_serializer("parts")
    @classmethod
    def _serialize_parts(cls, value: tuple[str, ...]) -> list[str]:
        """Serialize parts as list for backward compatibility."""
        return list(value)

    def to_string(self) -> str:
        """Convert to string representation with "/" separator."""
        return KEY_SEPARATOR.join(self.parts)

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

    def __iter__(self) -> Iterator[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return iterator over parts."""
        return iter(self.parts)

    @property
    def table_name(self) -> str:
        """Get SQL-like table name for this feature key."""
        return "__".join(self.parts)

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


class FeatureKey(_Key):
    """
    Feature key as a sequence of string parts.

    Hashable for use as dict keys in registries.
    Parts cannot contain forward slashes (/) or double underscores (__).

    Args:
        key: Feature key as string ("a/b/c"), sequence (["a", "b", "c"]), or FeatureKey instance.
             String format is split on "/" separator.

    Examples:
        >>> FeatureKey("a/b/c")  # String format
        FeatureKey(parts=['a', 'b', 'c'])
        >>> FeatureKey(["a", "b", "c"])  # List format
        FeatureKey(parts=['a', 'b', 'c'])
        >>> FeatureKey(FeatureKey(["a", "b", "c"]))  # FeatureKey copy
        FeatureKey(parts=['a', 'b', 'c'])
        >>> FeatureKey("a", "b", "c")  # Variadic format
        FeatureKey(parts=['a', 'b', 'c'])
    """

    @overload
    def __init__(self, key: str, /) -> None:
        """Initialize from string with "/" separator."""
        ...

    @overload
    def __init__(self, key: Sequence[str], /) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(self: Self, key: Self, /) -> None:
        """Initialize from another FeatureKey (copy)."""
        ...

    @overload
    def __init__(self, *parts: str) -> None:
        """Initialize from variadic string arguments."""
        ...

    @overload
    def __init__(self, *, parts: Sequence[str]) -> None:
        """Initialize from parts keyword argument."""
        ...

    def __init__(self, *args: str | _CoercibleToKey | Self, **kwargs: Any) -> None:
        """Initialize FeatureKey from various input types."""
        super().__init__(*args, **kwargs)

    @classmethod
    def __get_validators__(cls):
        """Pydantic validator for when used as a field type."""
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> FeatureKey:
        """Convert various inputs to FeatureKey."""
        if isinstance(value, cls):
            return value
        return cls(value)

    def model_dump(self, **kwargs: Any) -> Any:
        """Serialize to list format for backward compatibility."""
        # When serializing this key, return it as a list of parts
        # instead of the full Pydantic model structure
        return list(self.parts)

    @model_serializer
    def _serialize_model(self) -> list[str]:
        """Serialize to list when used as a field in another model."""
        return list(self.parts)

    def __hash__(self) -> int:
        """Return hash for use as dict keys."""
        return hash(self.parts)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another instance."""
        if isinstance(other, self.__class__):
            return self.parts == other.parts
        return super().__eq__(other)


class FieldKey(_Key):
    """
    Field key as a sequence of string parts.

    Hashable for use as dict keys in registries.
    Parts cannot contain forward slashes (/) or double underscores (__).

    Args:
        key: Field key as string ("a/b/c"), sequence (["a", "b", "c"]), or FieldKey instance.
             String format is split on "/" separator.

    Examples:
        >>> FieldKey("a/b/c")  # String format
        FieldKey(parts=['a', 'b', 'c'])
        >>> FieldKey(["a", "b", "c"])  # List format
        FieldKey(parts=['a', 'b', 'c'])
        >>> FieldKey(FieldKey(["a", "b", "c"]))  # FieldKey copy
        FieldKey(parts=['a', 'b', 'c'])
        >>> FieldKey("a", "b", "c")  # Variadic format
        FieldKey(parts=['a', 'b', 'c'])
    """

    @overload
    def __init__(self, key: str, /) -> None:
        """Initialize from string with "/" separator."""
        ...

    @overload
    def __init__(self, key: Sequence[str], /) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(self: Self, key: Self, /) -> None:
        """Initialize from another FieldKey (copy)."""
        ...

    @overload
    def __init__(self, *parts: str) -> None:
        """Initialize from variadic string arguments."""
        ...

    @overload
    def __init__(self, *, parts: Sequence[str]) -> None:
        """Initialize from parts keyword argument."""
        ...

    def __init__(self, *args: str | _CoercibleToKey | Self, **kwargs: Any) -> None:
        """Initialize FieldKey from various input types."""
        super().__init__(*args, **kwargs)

    @classmethod
    def __get_validators__(cls):
        """Pydantic validator for when used as a field type."""
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> FieldKey:
        """Convert various inputs to FieldKey."""
        if isinstance(value, cls):
            return value
        return cls(value)

    def model_dump(self, **kwargs: Any) -> Any:
        """Serialize to list format for backward compatibility."""
        # When serializing this key, return it as a list of parts
        # instead of the full Pydantic model structure
        return list(self.parts)

    @model_serializer
    def _serialize_model(self) -> list[str]:
        """Serialize to list when used as a field in another model."""
        return list(self.parts)

    def __hash__(self) -> int:
        """Return hash for use as dict keys."""
        return hash(self.parts)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another instance."""
        if isinstance(other, self.__class__):
            return self.parts == other.parts
        return super().__eq__(other)


CoercibleToFeatureKey: TypeAlias = _CoercibleToKey | FeatureKey
CoercibleToFieldKey: TypeAlias = _CoercibleToKey | FieldKey

FeatureKeyAdapter = TypeAdapter(
    FeatureKey
)  # can call .validate_python() to transform acceptable types into a FeatureKey
FieldKeyAdapter = TypeAdapter(
    FieldKey
)  # can call .validate_python() to transform acceptable types into a FieldKey

FeatureDepMetadata: TypeAlias = dict[str, Any]
