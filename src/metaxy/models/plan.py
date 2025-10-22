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

    def __lt__(self, other: "FQContainerKey") -> bool:
        """Enable sorting of FQContainerKey objects."""
        return self.to_string() < other.to_string()

    def __le__(self, other: "FQContainerKey") -> bool:
        """Enable sorting of FQContainerKey objects."""
        return self.to_string() <= other.to_string()

    def __gt__(self, other: "FQContainerKey") -> bool:
        """Enable sorting of FQContainerKey objects."""
        return self.to_string() > other.to_string()

    def __ge__(self, other: "FQContainerKey") -> bool:
        """Enable sorting of FQContainerKey objects."""
        return self.to_string() >= other.to_string()


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
            res.update(self.get_parent_containers_for_container(container.key))

        return res

    def get_parent_containers_for_container(
        self, key: ContainerKey
    ) -> Mapping[FQContainerKey, ContainerSpec]:
        res = {}

        container = self.feature.containers_by_key[key]

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
                        res[fq_key] = self.all_parent_containers_by_key[fq_key]
                else:
                    raise ValueError(
                        f"Unsupported dependency type: {type(container_dep)}"
                    )
        else:
            raise TypeError(f"Unsupported dependencies type: {type(container.deps)}")

        return res

    @cached_property
    def container_dependencies(
        self,
    ) -> Mapping[ContainerKey, Mapping[FeatureKey, list[ContainerKey]]]:
        """Get dependencies for each container in this feature.

        Returns a mapping from container key to its upstream dependencies.
        Each dependency maps an upstream feature key to a list of container keys
        that this container depends on.

        This is the format needed by DataVersionResolver.

        Returns:
            Mapping of container keys to their dependency specifications.
            Format: {container_key: {upstream_feature_key: [upstream_container_keys]}}
        """
        result: dict[ContainerKey, dict[FeatureKey, list[ContainerKey]]] = {}

        for container in self.feature.containers:
            container_deps: dict[FeatureKey, list[ContainerKey]] = {}

            if container.deps == SpecialContainerDep.ALL:
                # Depend on all upstream features and all their containers
                for upstream_feature in self.deps or []:
                    container_deps[upstream_feature.key] = [
                        c.key for c in upstream_feature.containers
                    ]
            elif isinstance(container.deps, list):
                # Specific dependencies defined
                for container_dep in container.deps:
                    feature_key = container_dep.feature_key

                    if container_dep.containers == SpecialContainerDep.ALL:
                        # All containers from this upstream feature
                        upstream_feature_spec = self.parent_features_by_key[feature_key]
                        container_deps[feature_key] = [
                            c.key for c in upstream_feature_spec.containers
                        ]
                    elif isinstance(container_dep.containers, list):
                        # Specific containers
                        container_deps[feature_key] = container_dep.containers

            result[container.key] = container_deps

        return result
