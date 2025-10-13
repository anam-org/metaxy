from enum import Enum

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.types import ContainerKey, FeatureKey


class SpecialContainerDep(Enum):
    ALL = "__METAXY_ALL_DEP__"


class ContainerDep(FrozenBaseModel):
    feature_key: FeatureKey
    containers: list[ContainerKey] | SpecialContainerDep = SpecialContainerDep.ALL


class ContainerSpec(FrozenBaseModel):
    key: ContainerKey
    code_version: int = 1

    # container-level dependencies can be one of the following:
    # - the default SpecialContainerDep.ALL to depend on all upstream features and all their containers
    # - a list of ContainerDep to depend on particular containers of specific features
    deps: SpecialContainerDep | list[ContainerDep] = SpecialContainerDep.ALL
