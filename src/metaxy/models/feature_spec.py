from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any

import pydantic
from pydantic import field_validator, model_validator

from metaxy.models.field import FieldSpec, SpecialFieldDep
from metaxy.models.types import FeatureDepMetadata, FeatureKey, FieldKey


def _coerce_feature_key(value: Any) -> FeatureKey:
    """Normalize incoming values to FeatureKey."""
    if isinstance(value, FeatureKey):
        return value
    if isinstance(value, str):
        return FeatureKey([value])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if not all(isinstance(part, str) for part in parts):
            raise TypeError("Feature key parts must be strings.")
        return FeatureKey(parts)
    raise TypeError("Feature key must be a FeatureKey, string, or sequence of strings.")


def _coerce_feature_dep_key(value: Any) -> FeatureKey:
    """Normalize FeatureDep key inputs including Feature classes/specs."""
    if hasattr(value, "spec"):
        spec = getattr(value, "spec")
        key = getattr(spec, "key", None)
        if isinstance(key, FeatureKey):
            return key
    if hasattr(value, "key"):
        key_candidate = getattr(value, "key")
        if isinstance(key_candidate, FeatureKey):
            return key_candidate
    return _coerce_feature_key(value)


class FeatureDep(pydantic.BaseModel):
    key: FeatureKey
    metadata: FeatureDepMetadata = pydantic.Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, value: Any):
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return value
        return {"key": value}

    @field_validator("key", mode="before")
    @classmethod
    def _validate_key(cls, value: Any) -> FeatureKey:
        return _coerce_feature_dep_key(value)


class FeatureSpec(pydantic.BaseModel):
    key: FeatureKey
    deps: list[FeatureDep] | None
    fields: list[FieldSpec] = pydantic.Field(
        default_factory=lambda: [
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=SpecialFieldDep.ALL,
            )
        ]
    )
    code_version: int = 1

    @field_validator("key", mode="before")
    @classmethod
    def _validate_key(cls, value: Any) -> FeatureKey:
        return _coerce_feature_key(value)

    @cached_property
    def fields_by_key(self) -> Mapping[FieldKey, FieldSpec]:
        return {c.key: c for c in self.fields}
