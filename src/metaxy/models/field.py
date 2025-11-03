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
    from metaxy.models.feature import Feature
    from metaxy.models.feature_spec import (
        CoercibleToFeatureKey,
        FeatureSpec,
    )


class SpecialFieldDep(Enum):
    ALL = "__METAXY_ALL_DEP__"


class FieldDep(BaseModel):
    feature: FeatureKey
    fields: list[FieldKey] | Literal[SpecialFieldDep.ALL] = SpecialFieldDep.ALL

    @overload
    def __init__(
        self,
        feature: str,
        **kwargs: Any,
    ) -> None:
        """Initialize from string feature key."""
        ...

    @overload
    def __init__(
        self,
        feature: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(
        self,
        feature: FeatureKey,
        **kwargs: Any,
    ) -> None:
        """Initialize from FeatureKey instance."""
        ...

    @overload
    def __init__(
        self,
        feature: "CoercibleToFeatureKey",
        **kwargs: Any,
    ) -> None:
        """Initialize from CoercibleToFeatureKey types."""
        ...

    @overload
    def __init__(
        self,
        feature: "FeatureSpec",
        **kwargs: Any,
    ) -> None:
        """Initialize from FeatureSpec instance."""
        ...

    @overload
    def __init__(
        self,
        feature: type["Feature"],
        **kwargs: Any,
    ) -> None:
        """Initialize from Feature instance."""
        ...

    def __init__(
        self,
        feature: "CoercibleToFeatureKey | FeatureSpec | type[Feature]",
        fields: list[CoercibleToFieldKey]
        | Literal[SpecialFieldDep.ALL] = SpecialFieldDep.ALL,
        *args,
        **kwargs,
    ):
        from metaxy.models.feature import Feature
        from metaxy.models.feature_spec import FeatureSpec

        if isinstance(feature, FeatureSpec):
            feature_key = feature.key
        elif isinstance(feature, type) and issubclass(feature, Feature):
            feature_key = feature.spec().key
        else:
            feature_key = FeatureKeyAdapter.validate_python(feature)

        assert isinstance(feature_key, FeatureKey)

        if isinstance(fields, list):
            validated_fields: Any = TypeAdapter(list[FieldKey]).validate_python(fields)
        else:
            validated_fields = fields  # Keep the enum value as-is

        super().__init__(feature=feature_key, fields=validated_fields, *args, **kwargs)


class FieldSpec(BaseModel):
    key: FieldKey = PydanticField(default_factory=lambda: FieldKey(["default"]))
    code_version: str = DEFAULT_CODE_VERSION

    # Field-level explicit dependencies
    # - SpecialFieldDep.ALL: explicitly depend on all upstream features and all their fields
    # - list[FieldDep]: depend on particular fields of specific features
    deps: SpecialFieldDep | list[FieldDep] = PydanticField(default_factory=list)

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


# Type adapter for validating FieldSpec with string coercion support
FieldSpecAdapter: TypeAdapter[FieldSpec] = TypeAdapter(
    Annotated[FieldSpec, BeforeValidator(_validate_field_spec_from_string)]
)
