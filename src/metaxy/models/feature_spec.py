from functools import cached_property
from metaxy.models.types import ContainerKey
from typing import Self
from typing import Mapping
import pydantic

from metaxy.models.container import ContainerSpec, SpecialContainerDep
from metaxy.models.types import FeatureDepMetadata, FeatureKey


class FeatureDep(pydantic.BaseModel):
    key: FeatureKey
    metadata: FeatureDepMetadata = pydantic.Field(default_factory=dict)


class FeatureSpec(pydantic.BaseModel):
    key: FeatureKey
    deps: list[FeatureDep] | None
    containers: list[ContainerSpec] = pydantic.Field(
        default_factory=lambda: [
            ContainerSpec(key=ContainerKey(["default"]), code_version=1, deps=SpecialContainerDep.ALL)
        ]
    )
    code_version: int = 1

    @cached_property
    def containers_by_key(self) -> Mapping[ContainerKey, ContainerSpec]:
        return {
            c.key: c for c in self.containers
        }
