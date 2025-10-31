from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic import BaseModel, TypeAdapter
from pydantic import Field as PydanticField

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
    feature_key: FeatureKey
    fields: list[FieldKey] | Literal[SpecialFieldDep.ALL] = SpecialFieldDep.ALL

    @overload
    def __init__(
        self,
        feature_key: str,
        **kwargs: Any,
    ) -> None:
        """Initialize from string feature key."""
        ...

    @overload
    def __init__(
        self,
        feature_key: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(
        self,
        feature_key: FeatureKey,
        **kwargs: Any,
    ) -> None:
        """Initialize from FeatureKey instance."""
        ...

    @overload
    def __init__(
        self,
        feature_key: "CoercibleToFeatureKey",
        **kwargs: Any,
    ) -> None:
        """Initialize from CoercibleToFeatureKey types."""
        ...

    @overload
    def __init__(
        self,
        feature_key: "FeatureSpec",
        **kwargs: Any,
    ) -> None:
        """Initialize from FeatureSpec instance."""
        ...

    @overload
    def __init__(
        self,
        feature_key: type["Feature"],
        **kwargs: Any,
    ) -> None:
        """Initialize from Feature instance."""
        ...

    def __init__(
        self,
        feature_key: "CoercibleToFeatureKey | FeatureSpec | type[Feature]",
        fields: list[CoercibleToFieldKey]
        | Literal[SpecialFieldDep.ALL] = SpecialFieldDep.ALL,
        *args,
        **kwargs,
    ):
        from metaxy.models.feature_spec import FeatureSpec

        if isinstance(feature_key, FeatureSpec):
            feature_key = feature_key.key
        elif isinstance(feature_key, type) and issubclass(feature_key, Feature):
            feature_key = feature_key.spec.key
        else:
            feature_key = FeatureKeyAdapter.validate_python(feature_key)

        assert isinstance(feature_key, FeatureKey)

        if isinstance(fields, list):
            validated_fields: Any = TypeAdapter(list[FieldKey]).validate_python(fields)
        else:
            validated_fields = fields  # Keep the enum value as-is

        super().__init__(
            feature_key=feature_key, fields=validated_fields, *args, **kwargs
        )


class FieldSpec(BaseModel):
    key: FieldKey = PydanticField(default_factory=lambda: FieldKey(["default"]))
    code_version: int = 1

    # field-level dependencies can be one of the following:
    # - the default SpecialFieldDep.ALL to depend on all upstream features and all their fields
    # - a list of FieldDep to depend on particular fields of specific features
    deps: SpecialFieldDep | list[FieldDep] = SpecialFieldDep.ALL

    @overload
    def __init__(
        self,
        key: str,
        code_version: int = 1,
        deps: SpecialFieldDep | list[FieldDep] = SpecialFieldDep.ALL,
    ) -> None:
        """Initialize from string key."""
        ...

    @overload
    def __init__(
        self,
        key: Sequence[str],
        code_version: int = 1,
        deps: SpecialFieldDep | list[FieldDep] = SpecialFieldDep.ALL,
    ) -> None:
        """Initialize from sequence of parts."""
        ...

    @overload
    def __init__(
        self,
        key: FieldKey,
        code_version: int = 1,
        deps: SpecialFieldDep | list[FieldDep] = SpecialFieldDep.ALL,
    ) -> None:
        """Initialize from FieldKey instance."""
        ...

    @overload
    def __init__(
        self,
        key: None,
        code_version: int = 1,
        deps: SpecialFieldDep | list[FieldDep] = SpecialFieldDep.ALL,
    ) -> None:
        """Initialize with None key (uses default)."""
        ...

    def __init__(
        self,
        key: CoercibleToFieldKey | None,
        code_version: int = 1,
        deps: SpecialFieldDep | list[FieldDep] = SpecialFieldDep.ALL,
        *args,
        **kwargs: Any,
    ) -> None:
        if key is None:
            super().__init__(
                key=FieldKey(["default"]),
                code_version=code_version,
                deps=deps,
                **kwargs,
            )
        else:
            super().__init__(
                key=FieldKeyAdapter.validate_python(key),
                code_version=code_version,
                deps=deps,
                *args,
                **kwargs,
            )
