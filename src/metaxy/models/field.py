from enum import Enum
from typing import Any

import pydantic
from pydantic import Field as PydanticField

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.types import FeatureKey, FieldKey


class SpecialFieldDep(Enum):
    ALL = "__METAXY_ALL_DEP__"


class FieldDep(FrozenBaseModel):
    feature_key: FeatureKey
    fields: list[FieldKey] | SpecialFieldDep = SpecialFieldDep.ALL

    @pydantic.field_validator("feature_key", mode="before")
    @classmethod
    def _validate_feature_key(cls, value: Any) -> FeatureKey:
        """Convert various types to FeatureKey.

        Accepts:
        - Feature class: Extracts spec.key
        - FeatureSpec: Extracts key
        - str: Converts to FeatureKey([str])
        - list[str]: Converts to FeatureKey(list)
        - FeatureKey: Returns as-is
        """
        # FeatureKey is already validated by its own validator
        if isinstance(value, FeatureKey):
            return value

        # Accept Feature class and extract key
        if hasattr(value, "spec") and hasattr(value.spec, "key"):
            return value.spec.key

        # Accept FeatureSpec and extract key
        if hasattr(value, "key") and isinstance(value.key, FeatureKey):
            return value.key

        # Accept str and convert to FeatureKey
        if isinstance(value, str):
            return FeatureKey([value])

        # Accept list[str] and convert to FeatureKey
        if isinstance(value, list):
            return FeatureKey(value)

        # Let FeatureKey's validator handle other cases
        return value


class FieldSpec(FrozenBaseModel):
    key: FieldKey = PydanticField(default_factory=lambda: FieldKey(["default"]))
    code_version: int = 1

    # field-level dependencies can be one of the following:
    # - the default SpecialFieldDep.ALL to depend on all upstream features and all their fields
    # - a list of FieldDep to depend on particular fields of specific features
    deps: SpecialFieldDep | list[FieldDep] = SpecialFieldDep.ALL
