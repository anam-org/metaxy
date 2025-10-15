from typing import Any, TypeAlias

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

FEATURE_KEY_SEPARATOR = "__"
FIELD_KEY_SEPARATOR = "__"


class FeatureKey(list):  # type: ignore[type-arg]
    """
    Feature key as a list of strings.

    Hashable for use as dict keys in registries.

    Parts cannot contain double underscores (__) as that's used as the separator.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Validate that no part contains "__"
        for part in self:
            if not isinstance(part, str):
                raise ValueError(
                    f"FeatureKey parts must be strings, got {type(part).__name__}"
                )
            if "__" in part:
                raise ValueError(
                    f"FeatureKey part '{part}' cannot contain double underscores (__). "
                    f"Double underscores are reserved as the separator in to_string(). "
                    f"Use single underscores or hyphens instead."
                )

    def to_string(self) -> str:
        return FEATURE_KEY_SEPARATOR.join(self)

    def __hash__(self):
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


class FieldKey(list):
    """
    Field key as a list of strings.

    Hashable for use as dict keys in registries.

    Parts cannot contain double underscores (__) as that's used as the separator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # pyrefly: ignore[not-iterable]
        # Validate that no part contains "__"
        for part in self:
            if not isinstance(part, str):
                raise ValueError(
                    f"FieldKey parts must be strings, got {type(part).__name__}"
                )
            if "__" in part:
                raise ValueError(
                    f"FieldKey part '{part}' cannot contain double underscores (__). "
                    f"Double underscores are reserved as the separator in to_string(). "
                    f"Use single underscores or hyphens instead."
                )

    def to_string(self) -> str:
        return FIELD_KEY_SEPARATOR.join(self)

    def __hash__(self):
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
