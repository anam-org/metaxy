from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TypeAlias, overload

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from typing_extensions import Self

from metaxy.metadata_store.system_tables import FEATURE_VERSIONS_KEY
from metaxy.models.constants import METAXY_FEATURE_VERSION
from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey
from metaxy.provenance.polars import PolarsProvenanceTracker
from metaxy.provenance.tracker import ProvenanceTracker
from metaxy.provenance.types import HashAlgorithm, Increment, LazyIncrement

CoercibleToFeature: TypeAlias = type[BaseFeature] | FeatureSpec | FeatureKey

__all__ = ["MetadataStore", "FEATURE_VERSIONS_KEY", "CoercibleToFeature"]


class MetadataStore(ABC):
    def __init__(
        self,
        hash_algo: HashAlgorithm,
        hash_length: int,
        auto_create_tables: bool = True,
    ):
        self.hash_algo = hash_algo
        self.hash_length = hash_length
        self.auto_create_tables = auto_create_tables

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
        plan = feature.graph.get_feature_plan(feature.spec().key)

        # 1. Load upstream columns
        upstream = {
            key: self.read_metadata(key, current_version=True)
            for key in plan.parent_features_by_key.keys()
        }

        # 2. Read current feature metadata
        current = self.read_metadata(feature, filters=filters.get(feature.spec().key, []), current_version=True)

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
            hash_algorithm=self.hash_algo,
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

    def read_metadata(self, feature: type[BaseFeature] | FeatureKey, filters: Sequence[nw.Expr] | None = None, current_version: bool = True) -> Frame | None:
        filters_list = list(filters or [])
        feature = self.coerce_to_feature(feature)

        if current_version:
            # Check if the table exists and has the metaxy_feature_version column
            # before adding the version filter
            temp_df = self.read_metadata_impl(feature, filters=None)
            if temp_df is not None:
                schema = temp_df.collect_schema()
                if METAXY_FEATURE_VERSION in schema.names():
                    filters_list.append(nw.col(METAXY_FEATURE_VERSION) == feature.feature_version())

        return self.read_metadata_impl(feature, filters_list)

    @abstractmethod
    def read_metadata_impl(self, feature: type[BaseFeature] | FeatureKey, filters: Sequence[nw.Expr] | None = None) -> Frame | None:
        ...

    @abstractmethod
    def write_metadata(self, feature: type[BaseFeature] | FeatureKey, data: Frame) -> None:
        ...

    @abstractmethod
    def open(self) -> AbstractContextManager[Self]:
        """Open the metadata store connection.

        Returns:
            A context manager that yields the store instance.

        Example:
            with store.open() as s:
                s.write_metadata(feature, data)
        """
        ...

    def __enter__(self) -> Self:
        """Enter the context manager by opening the store."""
        self._ctx = self.open()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        """Exit the context manager by closing the store."""
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)
