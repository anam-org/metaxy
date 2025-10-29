from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, TypeAdapter
from pydantic import Field as PydanticField

from metaxy.models.types import (
    CoercibleToFeatureKey,
    CoercibleToFieldKey,
    FeatureKey,
    FeatureKeyAdapter,
    FieldKey,
    FieldKeyAdapter,
)


class SpecialFieldDep(Enum):
    ALL = "__METAXY_ALL_DEP__"


class FieldDep(BaseModel):
    feature_key: FeatureKey
    fields: list[FieldKey] | Literal[SpecialFieldDep.ALL] = SpecialFieldDep.ALL

    def __init__(
        self,
        feature_key: CoercibleToFeatureKey,
        fields: list[CoercibleToFieldKey]
        | Literal[SpecialFieldDep.ALL] = SpecialFieldDep.ALL,
        *args,
        **kwargs,
    ):
        feature_key = FeatureKeyAdapter.validate_python(feature_key)

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

    def __init__(
        self,
        key: CoercibleToFieldKey,
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
