from collections.abc import Mapping
from functools import cached_property

import pydantic

from metaxy.models.field import FieldSpec, SpecialFieldDep
from metaxy.models.types import FeatureDepMetadata, FeatureKey, FieldKey


class FeatureDep(pydantic.BaseModel):
    key: FeatureKey
    metadata: FeatureDepMetadata = pydantic.Field(default_factory=dict)


class FeatureSpec(pydantic.BaseModel):
    key: FeatureKey
    deps: list[FeatureDep] | None
    fields: list[FieldSpec] = pydantic.Field(
        default_factory=lambda: [
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=SpecialFieldDep.ALL,
            )
        ]
    )
    code_version: int = 1

    @cached_property
    def fields_by_key(self) -> Mapping[FieldKey, FieldSpec]:
        return {c.key: c for c in self.fields}
