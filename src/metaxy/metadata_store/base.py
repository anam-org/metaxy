from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager

from metaxy.provenance.polars import PolarsProvenanceTracker
from typing_extensions import Self

from metaxy.models.plan import FeaturePlan
from metaxy.provenance.tracker import ProvenanceTracker
import narwhals as nw
from metaxy import BaseFeature, FeatureKey, FeatureSpec, FeatureGraph

from typing import Sequence, overload, TypeAlias, TypeGuard, cast, Mapping
from metaxy.provenance.types import Increment, LazyIncrement, HashAlgorithm
import polars as pl
from narwhals.typing import IntoFrameT, FrameT, Frame
from metaxy.models.constants import METAXY_FEATURE_VERSION

CoercibleToFeature: TypeAlias = type[BaseFeature] | FeatureSpec | FeatureKey


class MetadataStore(ABC):
    def __init__(
        self,
        hash_algo: HashAlgorithm,
        hash_length: int
    ):
        self.hash_algo = hash_algo
        self.hash_length = hash_length

    @overload
    def resolve_update(
        self,
        feature: CoercibleToFeature,
        hash_algorithm: HashAlgorithm,
        hash_length: int,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None = None,
        sample: pl.DataFrame | None = None,
        lazy: bool = True
    ) -> LazyIncrement:
        ...

    @overload
    def resolve_update(
        self,
        feature: CoercibleToFeature,
        hash_algorithm: HashAlgorithm,
        hash_length: int,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None = None,
        sample: pl.DataFrame | None = None,
        lazy: bool = False
    ) -> Increment:
        ...

    def resolve_update(
        self,
        feature: CoercibleToFeature,
        hash_algorithm: HashAlgorithm,
        hash_length: int,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None = None,
        sample: pl.DataFrame | None = None,
        lazy: bool = True,
    ) -> Increment | LazyIncrement:
        filters = filters or {}

        feature = self.coerce_to_feature(feature)
        # feature = cast(type[BaseFeature], feature)
        plan = feature.graph.get_feature_plan(feature.spec.key)

        # 1. Load upstream columns
        upstream = {
            key: self.read_metadata(key, current_version=True)
            for key in plan.parent_features_by_key.keys()
        }

        # 2. Read current feature metadata
        current = self.read_metadata(feature, filters=filters.get(feature.spec.key, []), current_version=True)

        # 3. Create tracker
        if self.supports_native_tracker():
            tracker = self.create_tracker(plan)
        else:
            tracker = PolarsProvenanceTracker(plan)

        # 4. Run tracker

        return tracker.resolve_increment_with_provenance(
            upstream=upstream,
            current=current,
            hash_length=self.hash_length,
            hash_algorithm=self.hash_algorithm,
            filters=filters,
            sample=sample,
        )

    def coerce_to_feature(
        self,
        feature: CoercibleToFeature
    ) -> type[BaseFeature]:
        if isinstance(feature, FeatureKey):
            return FeatureGraph.get_active().get_feature_by_key(feature)
        elif isinstance(feature, FeatureSpec):
            return FeatureGraph.get_active().get_feature_by_key(feature.key)
        else:
            assert isinstance(feature, type) and issubclass(feature, BaseFeature)
            return feature

    @abstractmethod
    def supports_native_tracker(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_tracker(self, plan: FeaturePlan) -> ProvenanceTracker:
        raise NotImplementedError

    def read_metadata(self, feature: type[BaseFeature] | FeatureKey, filters: Sequence[nw.Expr] | None = None, current_version: bool = True) -> Frame:
        filters = filters or []
        feature = self.coerce_to_feature(feature)
        if current_version:
            filters.append(nw.col(METAXY_FEATURE_VERSION) == feature.feature_version())
        return self.read_metadata_impl(feature, filters)

    @abstractmethod
    def read_metadata_impl(self, feature: type[BaseFeature] | FeatureKey, filters: Sequence[nw.Expr] | None = None) -> Frame:
        ...

    @abstractmethod
    def write_metadata(self, feature: type[BaseFeature] | FeatureKey, data: Frame) -> None:
        ...

    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
