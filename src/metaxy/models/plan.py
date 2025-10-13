import hashlib
from collections.abc import Mapping
from functools import cached_property

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.container import (
    ContainerDep,
    ContainerKey,
    ContainerSpec,
    SpecialContainerDep,
)
from metaxy.models.feature_spec import FeatureKey, FeatureSpec


class FQContainerKey(FrozenBaseModel):
    container: ContainerKey
    feature: FeatureKey

    def to_string(self) -> str:
        return f"{self.feature}/{self.container}"


class FeaturePlan(FrozenBaseModel):
    feature: FeatureSpec
    deps: list[FeatureSpec] | None

    @cached_property
    def parent_features_by_key(self) -> Mapping[FeatureKey, FeatureSpec]:
        return {feature.key: feature for feature in self.deps or []}

    @cached_property
    def all_parent_containers_by_key(self) -> Mapping[FQContainerKey, ContainerSpec]:
        res: dict[FQContainerKey, ContainerSpec] = {}

        for feature in self.deps or []:
            for container in feature.containers:
                res[FQContainerKey(container=container.key, feature=feature.key)] = (
                    container
                )

        return res

    @cached_property
    def parent_containers_by_key(self) -> Mapping[FQContainerKey, ContainerSpec]:
        res: dict[FQContainerKey, ContainerSpec] = {}

        for container in self.feature.containers:
            if container.deps == SpecialContainerDep.ALL:
                # we depend on all upstream features and their containers
                for feature in self.deps or []:
                    for container in feature.containers:
                        res[
                            FQContainerKey(container=container.key, feature=feature.key)
                        ] = container
            elif isinstance(container.deps, list):
                for container_dep in container.deps:
                    if container_dep.containers == SpecialContainerDep.ALL:
                        # we depend on all containers of the corresponding upstream feature
                        for parent_container in self.parent_features_by_key[
                            container_dep.feature_key
                        ].containers:
                            res[
                                FQContainerKey(
                                    container=parent_container.key,
                                    feature=container_dep.feature_key,
                                )
                            ] = parent_container

                    elif isinstance(container_dep, ContainerDep):
                        #
                        for container_key in container_dep.containers:
                            fq_key = FQContainerKey(
                                container=container_key,
                                feature=container_dep.feature_key,
                            )
                            res[fq_key] = self.parent_containers_by_key[fq_key]
                    else:
                        raise ValueError(
                            f"Unsupported dependency type: {type(container_dep)}"
                        )
            else:
                raise TypeError(
                    f"Unsupported dependencies type: {type(container.deps)}"
                )

        return res

    def get_parent_containers_for_container(
        self, key: ContainerKey
    ) -> Mapping[FQContainerKey, ContainerSpec]:
        res = {}

        for k, v in self.parent_containers_by_key.items():
            if k.container in self.feature.containers_by_key:
                res[FQContainerKey(feature=self.feature, container=k.container)] = v

        return res

    def get_container_data_version(self, key: ContainerKey) -> str:
        hasher = hashlib.sha256()

        container = self.feature.containers_by_key[key]

        hasher.update(key.to_string().encode())
        hasher.update(str(container.code_version).encode())

        for k, v in sorted(self.get_parent_containers_for_container(key).items()):
            hasher.update(k.to_string().encode())
            hasher.update(self.get_container_data_version(k.container).encode())

        return hasher.hexdigest()

    # @cached_property
    def data_version(self) -> dict[str, str]:
        """Computes the data version for the feature plan.

        Hash together container data versions versions with the feature code version.

        Returns:
            dict[str, str]: The data version for each container in the feature plan.
                Keys are container names as strings.
        """
        res = {}

        for k, v in self.feature.containers_by_key.items():
            hasher = hashlib.sha256()
            hasher.update(self.feature.key.to_string().encode())
            hasher.update(self.get_container_data_version(k).encode())
            res[k.to_string()] = hasher.hexdigest()  # Convert key to string

        return res
