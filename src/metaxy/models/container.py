from enum import Enum

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.types import FeatureKey, FieldKey


class SpecialFieldDep(Enum):
    ALL = "__METAXY_ALL_DEP__"


class FieldDep(FrozenBaseModel):
    feature_key: FeatureKey
    fields: list[FieldKey] | SpecialFieldDep = SpecialFieldDep.ALL


class FieldSpec(FrozenBaseModel):
    key: FieldKey
    code_version: int = 1

    # field-level dependencies can be one of the following:
    # - the default SpecialFieldDep.ALL to depend on all upstream features and all their fields
    # - a list of FieldDep to depend on particular fields of specific features
    deps: SpecialFieldDep | list[FieldDep] = SpecialFieldDep.ALL
