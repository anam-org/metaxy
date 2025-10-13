from functools import cached_property
from typing import Any, ClassVar, Self

from pydantic._internal._model_construction import ModelMetaclass

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey


class FeatureRegistry:
    def __init__(self):
        self.features_by_key: dict[FeatureKey, type[Feature]] = {}
        self.feature_specs_by_key: dict[FeatureKey, FeatureSpec] = {}

    def add_feature(self, feature: type["Feature"]) -> None:
        self.features_by_key[feature.spec.key] = feature
        self.feature_specs_by_key[feature.spec.key] = feature.spec

    def get_feature_plan(self, key: FeatureKey) -> FeaturePlan:
        feature = self.feature_specs_by_key[key]

        return FeaturePlan(
            feature=feature,
            deps=[self.feature_specs_by_key[dep_key] for dep_key in feature.deps or []]
            or None,
        )

    def get_feature_data_version(self, key: FeatureKey) -> dict[str, str]:
        return self.get_feature_plan(key).data_version()


metaxy_registry = FeatureRegistry()


class _FeatureMeta(ModelMetaclass):
    def __new__(
        cls,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *,
        spec: FeatureSpec | None,
        **kwargs,
    ) -> type[Self]:
        cls = super().__new__(cls, cls_name, bases, namespace)
        cls.metaxy_registry = metaxy_registry

        if spec:
            cls.spec = spec
            metaxy_registry.add_feature(cls)
        else:
            pass  # TODO: set spec to a property that would raise an exception on access

        return cls


class Feature(FrozenBaseModel, metaclass=_FeatureMeta, spec=None):
    spec: ClassVar[FeatureSpec]
    metaxy_registry: ClassVar[FeatureRegistry]

    # @cached_property
    @classmethod
    def data_version(self) -> str:
        return self.metaxy_registry.get_feature_data_version(self.spec.key)
