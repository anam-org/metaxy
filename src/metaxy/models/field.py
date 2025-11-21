from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, overload

from pydantic import BaseModel, BeforeValidator, TypeAdapter
from pydantic import Field as PydanticField

from metaxy.models.constants import DEFAULT_CODE_VERSION
from metaxy.models.types import (
    CoercibleToFieldKey,
    FeatureKey,
    FeatureKeyAdapter,
    FieldKey,
    FieldKeyAdapter,
)

if TYPE_CHECKING:
    # yes, these are circular imports, the TYPE_CHECKING block hides them at runtime.
    # neither pyright not basedpyright allow ignoring `reportImportCycles` because they think it's a bad practice
    # and it would be very smart to force the user to restructure their project instead
    # context: https://github.com/microsoft/pyright/issues/1825
    # however, considering the recursive nature of graphs, and the syntactic sugar that we want to support,
    # I decided to just put these errors into `.basedpyright/baseline.json` (after ensuring this is the only error produced by basedpyright)
    from metaxy.models.feature import BaseFeature
    from metaxy.models.feature_spec import FeatureSpec


class SpecialFieldDep(Enum):
    ALL = "__METAXY_ALL_DEP__"


def _validate_field_dep_feature(value: Any) -> FeatureKey:
    """Coerce various input types to FeatureKey for FieldDep."""
    # Import here to avoid circular dependency at module level
    from metaxy.models.feature import BaseFeature
    from metaxy.models.feature_spec import FeatureSpec

    if isinstance(value, FeatureKey):
        return value
    elif isinstance(value, FeatureSpec):
        return value.key
    elif isinstance(value, type) and issubclass(value, BaseFeature):
        return value.spec().key
    else:
        # Handle str, Sequence[str], etc.
        return FeatureKeyAdapter.validate_python(value)


def _validate_field_dep_fields(
    value: Any,
) -> list[FieldKey] | Literal[SpecialFieldDep.ALL]:
    """Coerce list of field keys to validated FieldKey instances."""
    if value is SpecialFieldDep.ALL:
        return SpecialFieldDep.ALL
    if isinstance(value, str):
        if value == SpecialFieldDep.ALL.value:
            return SpecialFieldDep.ALL
        # Invalid string value - will be caught by Pydantic validation
        raise ValueError(
            f"String value must be {SpecialFieldDep.ALL.value}, got {value}"
        )
    # Validate as list of FieldKeys
    return TypeAdapter(list[FieldKey]).validate_python(value)


class FieldDep(BaseModel):
    feature: Annotated[FeatureKey, BeforeValidator(_validate_field_dep_feature)]
    fields: Annotated[
        list[FieldKey] | Literal[SpecialFieldDep.ALL],
        BeforeValidator(_validate_field_dep_fields),
    ] = SpecialFieldDep.ALL

    if TYPE_CHECKING:

        @overload
        @overload
        def __init__(
            self,
            feature: str,
            **kwargs: Any,
        ) -> None:
            """Initialize from string feature key."""
            ...

        @overload
        @overload
        def __init__(
            self,
            feature: Sequence[str],
            **kwargs: Any,
        ) -> None:
            """Initialize from sequence of parts."""
            ...

        @overload
        @overload
        def __init__(
            self,
            feature: FeatureKey,
            **kwargs: Any,
        ) -> None:
            """Initialize from FeatureKey instance."""
            ...

        @overload
        @overload
        def __init__(
            self,
            feature: "FeatureSpec",
            **kwargs: Any,
        ) -> None:
            """Initialize from FeatureSpec instance."""
            ...

        @overload
        @overload
        def __init__(
            self,
            feature: type["BaseFeature"],
            **kwargs: Any,
        ) -> None:
            """Initialize from BaseFeature class."""
            ...

        # Final signature combining all overloads
        def __init__(  # pyright: ignore[reportMissingSuperCall]
            self,
            feature: str
            | Sequence[str]
            | FeatureKey
            | "FeatureSpec"
            | type["BaseFeature"],
            fields: list[CoercibleToFieldKey]
            | Literal[SpecialFieldDep.ALL] = SpecialFieldDep.ALL,
            **kwargs: Any,
        ) -> None: ...  # pyright: ignore[reportMissingSuperCall]


def _validate_field_spec_from_string(value: Any) -> Any:
    """Validator function to convert string to FieldSpec dict.

    This allows FieldSpec to be constructed from just a string key:
    - "my_field" -> FieldSpec(key="my_field", code_version="1")

    Args:
        value: The value to validate (can be str, dict, or FieldSpec)

    Returns:
        Either the original value or a dict that Pydantic will use to construct FieldSpec
    """
    # If it's a string, convert to dict with key field
    if isinstance(value, str):
        return {"key": value}

    # Otherwise return as-is for normal Pydantic processing
    return value


def _validate_field_spec_key(value: Any) -> FieldKey:
    """Coerce various input types to FieldKey."""
    if isinstance(value, FieldKey):
        return value
    return FieldKeyAdapter.validate_python(value)


class FieldSpec(BaseModel):
    key: Annotated[FieldKey, BeforeValidator(_validate_field_spec_key)] = PydanticField(
        default_factory=lambda: FieldKey(["default"])
    )
    code_version: str = DEFAULT_CODE_VERSION

    # Field-level explicit dependencies
    # - SpecialFieldDep.ALL: explicitly depend on all upstream features and all their fields
    # - list[FieldDep]: depend on particular fields of specific features
    deps: SpecialFieldDep | list[FieldDep] = PydanticField(default_factory=list)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Add custom validator to coerce strings to FieldSpec."""
        from pydantic_core import core_schema

        # Get the default schema
        python_schema = handler(source_type)

        # Wrap it with a before validator that converts strings
        return core_schema.no_info_before_validator_function(
            _validate_field_spec_from_string,
            python_schema,
        )

    if TYPE_CHECKING:

        @overload
        def __init__(self, key: CoercibleToFieldKey, **kwargs) -> None:
            """Initialize from key and no other arguments."""
            ...

        @overload
        def __init__(
            self,
            key: str,
            code_version: str,
            deps: SpecialFieldDep | list[FieldDep] | None = None,
        ) -> None:
            """Initialize from string key."""
            ...

        @overload
        def __init__(
            self,
            key: Sequence[str],
            code_version: str,
            deps: SpecialFieldDep | list[FieldDep] | None = None,
        ) -> None:
            """Initialize from sequence of parts."""
            ...

        @overload
        def __init__(
            self,
            key: FieldKey,
            code_version: str,
            deps: SpecialFieldDep | list[FieldDep] | None = None,
        ) -> None:
            """Initialize from FieldKey instance."""
            ...

        # Final signature combining all overloads
        def __init__(  # pyright: ignore[reportMissingSuperCall]
            self,
            key: CoercibleToFieldKey,
            code_version: str = DEFAULT_CODE_VERSION,
            deps: SpecialFieldDep | list[FieldDep] | None = None,
            **kwargs: Any,
        ) -> None: ...

    # Runtime __init__ to handle positional arguments
    def __init__(
        self,
        key: CoercibleToFieldKey,
        code_version: str = DEFAULT_CODE_VERSION,
        deps: SpecialFieldDep | list[FieldDep] | None = None,
        *args,
        **kwargs: Any,
    ) -> None:
        validated_key = FieldKeyAdapter.validate_python(key)

        # Handle None deps - use empty list as default
        if deps is None:
            deps = []

        super().__init__(
            key=validated_key,
            code_version=code_version,
            deps=deps,
            *args,
            **kwargs,
        )


# Type adapter for validating FieldSpec with string coercion support

CoersibleToFieldSpecsTypeAdapter = TypeAdapter(list[FieldSpec])
