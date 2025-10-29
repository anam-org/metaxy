from typing import Any, NamedTuple, TypeAlias

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

FEATURE_KEY_SEPARATOR = "/"
FIELD_KEY_SEPARATOR = "/"


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


class FeatureKey(list):  # pyright: ignore[reportMissingTypeArgument]
    """
    Feature key as a list of strings.

    Hashable for use as dict keys in registries.

    Parts cannot contain forward slashes (/) or double underscores (__).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Validate that no part contains "/" or "__"
        for part in self:
            if not isinstance(part, str):
                raise ValueError(
                    f"FeatureKey parts must be strings, got {type(part).__name__}"
                )
            if "/" in part:
                raise ValueError(
                    f"FeatureKey part '{part}' cannot contain forward slashes (/). "
                    f"Forward slashes are reserved as the separator in to_string(). "
                    f"Use underscores or hyphens instead."
                )
            if "__" in part:
                raise ValueError(
                    f"FeatureKey part '{part}' cannot contain double underscores (__). "
                    f"Use single underscores or hyphens instead."
                )

    def to_string(self) -> str:
        return FEATURE_KEY_SEPARATOR.join(self)

    def __repr__(self) -> str:
        return self.to_string()

    def __hash__(self):  # pyright: ignore[reportIncompatibleVariableOverride]
        return hash(tuple(self))

    def __eq__(self, other):
        if isinstance(other, FeatureKey):
            return list.__eq__(self, other)
        return list.__eq__(self, other)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Pydantic schema that preserves FeatureKey type."""
        # python_schema = core_schema.is_instance_schema(cls)

        list_of_str_schema = core_schema.list_schema(core_schema.str_schema())

        return core_schema.no_info_wrap_validator_function(
            cls._validate,
            list_of_str_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: list(x)
            ),
        )

    @classmethod
    def _validate(cls, value, handler):
        """Validate and wrap in FeatureKey."""
        if isinstance(value, cls):
            return value
        # Let the list schema validate first
        validated = handler(value)
        # Wrap in FeatureKey
        return cls(validated)

    @property
    def table_name(self) -> str:
        """Get SQL-like table name for this feature key."""
        return "__".join(self)


class FieldKey(list):  # pyright: ignore[reportMissingTypeArgument]
    """
    Field key as a list of strings.

    Hashable for use as dict keys in registries.

    Parts cannot contain forward slashes (/) or double underscores (__).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # pyrefly: ignore[not-iterable]
        # Validate that no part contains "/" or "__"
        for part in self:
            if not isinstance(part, str):
                raise ValueError(
                    f"FieldKey parts must be strings, got {type(part).__name__}"
                )
            if "/" in part:
                raise ValueError(
                    f"FieldKey part '{part}' cannot contain forward slashes (/). "
                    f"Forward slashes are reserved as the separator in to_string(). "
                    f"Use underscores or hyphens instead."
                )
            if "__" in part:
                raise ValueError(
                    f"FieldKey part '{part}' cannot contain double underscores (__). "
                    f"Use single underscores or hyphens instead."
                )

    def to_string(self) -> str:
        return FIELD_KEY_SEPARATOR.join(self)

    def __repr__(self) -> str:
        return self.to_string()

    def __hash__(self):  # pyright: ignore[reportIncompatibleVariableOverride]
        return hash(tuple(self))

    def __eq__(self, other):
        if isinstance(other, FieldKey):
            return list.__eq__(self, other)
        return list.__eq__(self, other)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Pydantic schema that preserves FieldKey type."""
        # python_schema = core_schema.is_instance_schema(cls)

        list_of_str_schema = core_schema.list_schema(core_schema.str_schema())

        return core_schema.no_info_wrap_validator_function(
            cls._validate,
            list_of_str_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: list(x)
            ),
        )

    @classmethod
    def _validate(cls, value, handler):
        """Validate and wrap in FieldKey."""
        if isinstance(value, cls):
            return value
        # Let the list schema validate first
        validated = handler(value)
        # Wrap in FieldKey
        return cls(validated)


FeatureDepMetadata: TypeAlias = dict[str, Any]
