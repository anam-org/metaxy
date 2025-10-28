from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import field_validator

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.types import FeatureKey, FieldKey


class SpecialFieldDep(Enum):
    ALL = "__METAXY_ALL_DEP__"


class FieldDep(FrozenBaseModel):
    feature_key: FeatureKey
    fields: list[FieldKey] | SpecialFieldDep = SpecialFieldDep.ALL

    @field_validator("feature_key", mode="before")
    @classmethod
    def _validate_feature_key(cls, value: Any) -> FeatureKey:
        return _coerce_feature_key(value)

    @field_validator("fields", mode="before")
    @classmethod
    def _validate_fields(cls, value: Any) -> SpecialFieldDep | list[FieldKey]:
        if value is SpecialFieldDep.ALL:
            return value
        if isinstance(value, SpecialFieldDep):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_coerce_field_key(item) for item in list(value)]
        raise TypeError("fields must be a list of field keys or SpecialFieldDep.ALL.")


class FieldSpec(FrozenBaseModel):
    key: FieldKey
    code_version: str = "1"

    # field-level dependencies can be one of the following:
    # - the default SpecialFieldDep.ALL to depend on all upstream features and all their fields
    # - a list of FieldDep to depend on particular fields of specific features
    deps: SpecialFieldDep | list[FieldDep] = SpecialFieldDep.ALL

    @field_validator("code_version", mode="before")
    @classmethod
    def _validate_code_version(cls, value: Any) -> str:
        if value is None:
            return "1"
        if isinstance(value, int):
            return str(value)
        if not isinstance(value, str):
            raise TypeError("code_version must be a string.")
        if not value:
            raise ValueError("code_version cannot be empty.")
        return value

    @field_validator("key", mode="before")
    @classmethod
    def _validate_key(cls, value: Any) -> FieldKey:
        return _coerce_field_key(value)


def _coerce_feature_key(value: Any) -> FeatureKey:
    if isinstance(value, FeatureKey):
        return value
    if isinstance(value, str):
        return FeatureKey([value])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if not all(isinstance(part, str) for part in parts):
            raise TypeError("Feature key parts must be strings.")
        return FeatureKey(parts)
    raise TypeError("feature_key must be a FeatureKey, string, or sequence of strings.")


def _coerce_field_key(value: Any) -> FieldKey:
    if isinstance(value, FieldKey):
        return value
    if isinstance(value, str):
        return FieldKey([value])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if not all(isinstance(part, str) for part in parts):
            raise TypeError("Field key parts must be strings.")
        return FieldKey(parts)
    raise TypeError("field key must be a FieldKey, string, or sequence of strings.")
