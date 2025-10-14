import sys
from typing import Any, ClassVar

import polars as pl

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

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
        """Add a feature to the registry.

        Args:
            feature: Feature class to register

        Raises:
            ValueError: If a feature with the same key is already registered
        """
        if feature.spec.key in self.features_by_key:
            existing = self.features_by_key[feature.spec.key]
            raise ValueError(
                f"Feature with key {feature.spec.key.to_string()} already registered. "
                f"Existing: {existing.__name__}, New: {feature.__name__}. "
                f"Each feature key must be unique within a registry."
            )

        self.features_by_key[feature.spec.key] = feature
        self.feature_specs_by_key[feature.spec.key] = feature.spec

    def get_feature_plan(self, key: FeatureKey) -> FeaturePlan:
        feature = self.feature_specs_by_key[key]

        return FeaturePlan(
            feature=feature,
            deps=[self.feature_specs_by_key[dep.key] for dep in feature.deps or []]
            or None,
        )

    def get_feature_data_version(self, key: FeatureKey) -> dict[str, str]:
        return self.get_feature_plan(key).data_version()

    def get_downstream_features(self, sources: list[FeatureKey]) -> list[FeatureKey]:
        """Get all features downstream of sources, topologically sorted.

        Performs a depth-first traversal of the dependency graph to find all
        features that transitively depend on any of the source features.

        Args:
            sources: List of source feature keys

        Returns:
            List of downstream feature keys in topological order (dependencies first).
            Does not include the source features themselves.

        Example:
            >>> # DAG: A -> B -> D
            >>> #      A -> C -> D
            >>> registry.get_downstream_features([FeatureKey(["A"])])
            [FeatureKey(["B"]), FeatureKey(["C"]), FeatureKey(["D"])]
        """
        source_set = set(sources)
        visited = set()
        post_order = []  # Reverse topological order

        def visit(key: FeatureKey):
            """DFS traversal."""
            if key in visited:
                return
            visited.add(key)

            # Find all features that depend on this one
            for feature_key, feature_spec in self.feature_specs_by_key.items():
                if feature_spec.deps:
                    for dep in feature_spec.deps:
                        if dep.key == key:
                            # This feature depends on 'key', so visit it
                            visit(feature_key)

            post_order.append(key)

        # Visit all sources
        for source in sources:
            visit(source)

        # Remove sources from result, reverse to get topological order
        result = [k for k in reversed(post_order) if k not in source_set]
        return result


metaxy_registry = FeatureRegistry()


class _FeatureMeta(ModelMetaclass):
    def __new__(
        cls,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *,
        spec: FeatureSpec | None,
        registry: FeatureRegistry = metaxy_registry,
        **kwargs,
    ) -> type[Self]:
        cls = super().__new__(cls, cls_name, bases, namespace)
        cls.registry = registry  # type: ignore[attr-defined]

        if spec:
            cls.spec = spec  # type: ignore[attr-defined]
            cls.registry.add_feature(cls)  # type: ignore[attr-defined]
        else:
            pass  # TODO: set spec to a property that would raise an exception on access

        return cls


class Feature(FrozenBaseModel, metaclass=_FeatureMeta, spec=None):
    spec: ClassVar[FeatureSpec]
    registry: ClassVar[FeatureRegistry]

    @classmethod
    def feature_version(cls) -> str:
        """Get hash of feature specification.

        Returns an 8-character hash representing the feature's complete configuration:
        - Feature key
        - Container definitions and code versions
        - Dependencies (feature-level and container-level)

        This hash changes when you modify:
        - Container code versions
        - Dependencies
        - Container definitions

        Used to distinguish current vs historical metadata versions.
        Stored in the 'feature_version' column of metadata DataFrames.

        Returns:
            8-character SHA256 hex digest (like git short hashes)

        Example:
            >>> class MyFeature(Feature, spec=FeatureSpec(
            ...     key=FeatureKey(["my", "feature"]),
            ...     containers=[ContainerSpec(key=ContainerKey(["default"]), code_version=1)],
            ... )):
            ...     pass
            >>> MyFeature.feature_version()
            'a3f8b2c1'
        """
        import hashlib

        from metaxy.models.container import SpecialContainerDep

        # Build deterministic representation
        components = [cls.spec.key.to_string()]

        # Container definitions (sorted for determinism)
        for container in sorted(cls.spec.containers, key=lambda c: c.key.to_string()):
            components.append(f"c:{container.key.to_string()}:{container.code_version}")

            # Include container dependencies
            if container.deps == SpecialContainerDep.ALL:
                components.append("cdeps:ALL")
            elif isinstance(container.deps, list):
                for dep in sorted(
                    container.deps, key=lambda d: d.feature_key.to_string()
                ):
                    components.append(f"cdep:{dep.feature_key.to_string()}")
                    if dep.containers == SpecialContainerDep.ALL:
                        components.append("conts:ALL")
                    elif isinstance(dep.containers, list):
                        for cont_key in sorted(
                            dep.containers, key=lambda k: k.to_string()
                        ):
                            components.append(f"cont:{cont_key.to_string()}")

        # Feature-level dependencies (sorted for determinism)
        if cls.spec.deps:
            for dep in sorted(cls.spec.deps, key=lambda d: d.key.to_string()):
                components.append(f"fdep:{dep.key.to_string()}")

        # Hash everything together
        hasher = hashlib.sha256()
        for component in components:
            hasher.update(component.encode())

        # Return first 8 characters (like git short hashes)
        return hasher.hexdigest()[:8]

    @classmethod
    def align_metadata_with_upstream(
        cls,
        current_metadata: pl.DataFrame,
        upstream_metadata: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Align metadata with upstream by joining on sample_id.

        Override this method to customize alignment logic when upstream features change.
        This is called during migration propagation to determine which samples to process.

        Common use cases:
        - One-to-many mappings (e.g., one video -> many frames)
        - Filtering based on upstream conditions
        - Cross-product joins for combinatorial features
        - Custom sample ID generation

        Default behavior: Inner join on 'sample_id' with all upstream features.
        Only preserves sample_id column - all other columns are dropped and will
        be recalculated during data version computation.

        Args:
            current_metadata: Existing metadata for this feature (may be empty)
            upstream_metadata: Dict mapping upstream feature keys to their metadata DataFrames.
                Each DataFrame includes sample_id, data_version, and other columns.

        Returns:
            DataFrame with 'sample_id' column (at minimum), ready for data version calculation.
            Other columns are preserved if present in current_metadata.

        Example - One-to-many (video frames):
            >>> @classmethod
            >>> def align_metadata_with_upstream(
            ...     cls,
            ...     current_metadata: pl.DataFrame,
            ...     upstream_metadata: dict[str, pl.DataFrame],
            ... ) -> pl.DataFrame:
            ...     # Each video produces 30 frames
            ...     video_samples = upstream_metadata["videos"]["sample_id"]
            ...     frames = []
            ...     for video_id in video_samples:
            ...         for frame_idx in range(30):
            ...             frames.append({
            ...                 "sample_id": f"{video_id}_frame_{frame_idx}",
            ...                 "video_id": video_id,
            ...                 "frame_idx": frame_idx,
            ...             })
            ...     return pl.DataFrame(frames)

        Example - Conditional filtering:
            >>> @classmethod
            >>> def align_metadata_with_upstream(
            ...     cls,
            ...     current_metadata: pl.DataFrame,
            ...     upstream_metadata: dict[str, pl.DataFrame],
            ... ) -> pl.DataFrame:
            ...     # Only process videos longer than 10 seconds
            ...     videos = upstream_metadata["videos"]
            ...     return videos.filter(pl.col("duration") > 10).select(["sample_id", "duration"])

        Example - Outer join (keep all upstream samples):
            >>> @classmethod
            >>> def align_metadata_with_upstream(
            ...     cls,
            ...     current_metadata: pl.DataFrame,
            ...     upstream_metadata: dict[str, pl.DataFrame],
            ... ) -> pl.DataFrame:
            ...     # Union of all upstream sample IDs
            ...     all_samples = set()
            ...     for upstream_df in upstream_metadata.values():
            ...         all_samples.update(upstream_df["sample_id"].to_list())
            ...     return pl.DataFrame({"sample_id": sorted(all_samples)})
        """
        if not upstream_metadata:
            # No upstream, return current metadata with only sample_id
            if len(current_metadata) > 0:
                return current_metadata.select(pl.col("sample_id"))
            else:
                return pl.DataFrame({"sample_id": []})

        # Default: inner join on sample_id across all upstream features
        # This ensures we only process samples that exist in ALL upstream features
        common_samples: set[int] | None = None
        for upstream_df in upstream_metadata.values():
            sample_ids = set(upstream_df["sample_id"].to_list())
            if common_samples is None:
                common_samples = sample_ids
            else:
                common_samples &= sample_ids  # Intersection

        if not common_samples:
            return pl.DataFrame({"sample_id": []})

        # Filter current metadata to common samples, preserving existing columns
        if len(current_metadata) > 0:
            return current_metadata.filter(
                pl.col("sample_id").is_in(list(common_samples))
            )
        else:
            # No current metadata, create from upstream sample IDs
            return pl.DataFrame({"sample_id": sorted(common_samples)})

    @classmethod
    def data_version(cls) -> dict[str, str]:
        """Get data version for all containers in this feature.

        Returns:
            Dictionary mapping container names (as strings) to their hash values.
        """
        return cls.registry.get_feature_data_version(cls.spec.key)
