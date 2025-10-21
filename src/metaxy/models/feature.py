from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import polars as pl
from pydantic._internal._model_construction import ModelMetaclass
from typing_extensions import Self

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey

# Type variable for backend-agnostic operations
TRef = TypeVar("TRef")

if TYPE_CHECKING:
    from metaxy.data_versioning.diff import DiffResult, MetadataDiffResolver
    from metaxy.data_versioning.joiners import UpstreamJoiner


# Context variable for active registry (module-level)
_active_registry: ContextVar["FeatureRegistry | None"] = ContextVar(
    "_active_registry", default=None
)


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

    def to_snapshot(self) -> dict[str, dict]:
        """Serialize registry to snapshot format.
        
        Returns a dict mapping feature_key (string) to feature data dict,
        including the import path of the Feature class for reconstruction.
        
        Returns:
            Dict of feature_key -> {
                feature_spec: dict, 
                feature_version: str,
                feature_class_path: str
            }
            
        Example:
            >>> snapshot = registry.to_snapshot()
            >>> snapshot["video_processing"]["feature_version"]
            'abc12345'
            >>> snapshot["video_processing"]["feature_class_path"]
            'myapp.features.video.VideoProcessing'
        """
        snapshot = {}
        
        for feature_key, feature_cls in self.features_by_key.items():
            feature_key_str = feature_key.to_string()
            feature_spec_dict = feature_cls.spec.model_dump(mode='json')  # type: ignore[attr-defined]
            feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]
            
            # Get class import path (module.ClassName)
            class_path = f"{feature_cls.__module__}.{feature_cls.__name__}"
            
            snapshot[feature_key_str] = {
                "feature_spec": feature_spec_dict,
                "feature_version": feature_version,
                "feature_class_path": class_path,
            }
        
        return snapshot
    
    @classmethod
    def from_snapshot(
        cls, 
        snapshot_data: dict[str, dict],
        *,
        class_path_overrides: dict[str, str] | None = None
    ) -> "FeatureRegistry":
        """Reconstruct registry from snapshot by importing Feature classes.
        
        Strictly requires Feature classes to exist at their recorded import paths.
        This ensures custom methods (like align_metadata_with_upstream) are available.
        
        If a feature has been moved/renamed, use class_path_overrides to specify
        the new location.
        
        Args:
            snapshot_data: Dict of feature_key -> {
                feature_spec: dict,
                feature_class_path: str,
                ...
            } (as returned by to_snapshot() or loaded from DB)
            class_path_overrides: Optional dict mapping feature_key to new class path
                                 for features that have been moved/renamed
        
        Returns:
            New FeatureRegistry with historical features
            
        Raises:
            ImportError: If feature class cannot be imported at recorded path
            
        Example:
            >>> # Load snapshot from metadata store
            >>> historical_registry = FeatureRegistry.from_snapshot(snapshot_data)
            >>> 
            >>> # With override for moved feature
            >>> historical_registry = FeatureRegistry.from_snapshot(
            ...     snapshot_data,
            ...     class_path_overrides={
            ...         "video_processing": "myapp.features_v2.VideoProcessing"
            ...     }
            ... )
        """
        from metaxy.models.feature_spec import FeatureSpec
        
        registry = cls()
        class_path_overrides = class_path_overrides or {}
        
        for feature_key_str, feature_data in snapshot_data.items():
            # Parse FeatureSpec for validation
            feature_spec_dict = feature_data["feature_spec"]
            feature_spec = FeatureSpec.model_validate(feature_spec_dict)
            
            # Get class path (check overrides first)
            if feature_key_str in class_path_overrides:
                class_path = class_path_overrides[feature_key_str]
            else:
                class_path = feature_data.get("feature_class_path")
                if not class_path:
                    raise ValueError(
                        f"Feature '{feature_key_str}' has no feature_class_path in snapshot. "
                        f"Cannot reconstruct historical registry."
                    )
            
            # Import the class
            try:
                module_path, class_name = class_path.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                feature_cls = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Cannot import Feature class '{class_path}' for historical migration. "
                    f"Feature '{feature_key_str}' is required for this migration but the class "
                    f"cannot be found at the recorded import path. "
                    f"\n\n"
                    f"Options:\n"
                    f"1. Restore the feature class at '{class_path}'\n"
                    f"2. If the feature was moved, add a class_path_override in the migration YAML:\n"
                    f"   feature_class_overrides:\n"
                    f"     {feature_key_str}: \"new.module.path.ClassName\"\n"
                    f"\n"
                    f"Original error: {e}"
                ) from e
            
            # Validate the imported class matches the stored spec
            if not hasattr(feature_cls, 'spec'):
                raise TypeError(
                    f"Imported class '{class_path}' is not a valid Feature class "
                    f"(missing 'spec' attribute)"
                )
            
            # Register it
            registry.features_by_key[feature_spec.key] = feature_cls
            registry.feature_specs_by_key[feature_spec.key] = feature_spec
        
        return registry

    @classmethod
    def get_active(cls) -> "FeatureRegistry":
        """Get the currently active registry.

        Returns the registry from the context variable if set, otherwise returns
        the default global registry.

        Returns:
            Active FeatureRegistry instance

        Example:
            >>> # Normal usage - returns default registry
            >>> reg = FeatureRegistry.get_active()
            >>>
            >>> # With custom registry in context
            >>> with my_registry.use():
            ...     reg = FeatureRegistry.get_active()  # Returns my_registry
        """
        return _active_registry.get() or registry

    @classmethod
    def set_active(cls, reg: "FeatureRegistry") -> None:
        """Set the active registry for the current context.

        This sets the context variable that will be returned by get_active().
        Typically used in application setup code or test fixtures.

        Args:
            reg: FeatureRegistry to activate

        Example:
            >>> # In application setup
            >>> my_registry = FeatureRegistry()
            >>> FeatureRegistry.set_active(my_registry)
            >>>
            >>> # Now all operations use my_registry
            >>> FeatureRegistry.get_active()  # Returns my_registry
        """
        _active_registry.set(reg)

    @contextmanager
    def use(self):
        """Context manager to temporarily use this registry as active.

        This is the recommended way to use custom registries, especially in tests.
        The registry is automatically restored when the context exits.

        Yields:
            This registry instance

        Example:
            >>> test_registry = FeatureRegistry()
            >>>
            >>> with test_registry.use():
            ...     # All operations use test_registry
            ...     class TestFeature(Feature, spec=...):
            ...         pass
            ...
            >>> # Outside context, back to previous registry
        """
        token = _active_registry.set(self)
        try:
            yield self
        finally:
            _active_registry.reset(token)


# Default global registry
registry = FeatureRegistry()


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
        new_cls = super().__new__(cls, cls_name, bases, namespace)

        if spec:
            # Get registry from context at class definition time
            active_registry = FeatureRegistry.get_active()
            new_cls.registry = active_registry  # type: ignore[attr-defined]
            new_cls.spec = spec  # type: ignore[attr-defined]
            active_registry.add_feature(new_cls)
        else:
            pass  # TODO: set spec to a property that would raise an exception on access

        return new_cls


class Feature(FrozenBaseModel, metaclass=_FeatureMeta, spec=None):
    spec: ClassVar[FeatureSpec]
    registry: ClassVar[FeatureRegistry]

    @classmethod
    def feature_version(cls) -> str:
        """Get hash of feature specification.

        Returns a hash representing the feature's complete configuration:
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
            SHA256 hex digest (like git short hashes)

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

        return hasher.hexdigest()

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
        """Get the code-level data version for this feature.

        This returns a static hash based on code versions and dependencies,
        not sample-level data versions.

        Returns:
            Dictionary mapping container keys to their data version hashes.
        """
        return cls.registry.get_feature_data_version(cls.spec.key)

    @classmethod
    def join_upstream_metadata(
        cls,
        joiner: "UpstreamJoiner[TRef]",
        upstream_refs: dict[str, TRef],
    ) -> tuple[TRef, dict[str, str]]:
        """Join upstream feature metadata.

        Override for custom join logic (1:many, different keys, filtering, etc.).

        Args:
            joiner: UpstreamJoiner from MetadataStore
            upstream_refs: Upstream feature metadata references (lazy where possible)

        Returns:
            (joined_upstream, upstream_column_mapping)
            - joined_upstream: All upstream data joined together
            - upstream_column_mapping: Maps upstream_key -> column name

        Example (default):
            >>> class MyFeature(Feature, spec=...):
            ...     pass  # Uses joiner's default implementation

        Example (custom 1:many join):
            >>> class VideoFramesFeature(Feature, spec=...):
            ...     @classmethod
            ...     def join_upstream_metadata(cls, joiner, upstream_refs):
            ...         # Custom join logic using joiner's methods
            ...         # This is backend-agnostic - works with Polars, Ibis, etc.!
            ...         return joiner.join_upstream(
            ...             upstream_refs=upstream_refs,
            ...             feature_spec=cls.spec,
            ...             feature_plan=cls.registry.get_feature_plan(cls.spec.key),
            ...         )
        """
        return joiner.join_upstream(
            upstream_refs=upstream_refs,
            feature_spec=cls.spec,
            feature_plan=cls.registry.get_feature_plan(cls.spec.key),
        )

    @classmethod
    def resolve_data_version_diff(
        cls,
        diff_resolver: "MetadataDiffResolver[TRef]",
        target_versions: TRef,
        current_metadata: TRef | None,
    ) -> "DiffResult":
        """Resolve differences between target and current data versions.

        Override for custom diff logic (ignore certain fields, custom rules, etc.).

        Args:
            diff_resolver: MetadataDiffResolver from MetadataStore
            target_versions: Calculated target data_versions (lazy ref)
            current_metadata: Current metadata for this feature (lazy ref, or None)

        Returns:
            DiffResult with added, changed, removed references

        Example (default):
            >>> class MyFeature(Feature, spec=...):
            ...     pass  # Uses diff resolver's default implementation

        Example (ignore certain container changes):
            >>> class MyFeature(Feature, spec=...):
            ...     @classmethod
            ...     def resolve_data_version_diff(cls, diff_resolver, target_versions, current_metadata):
            ...         # Get standard diff
            ...         result = diff_resolver.find_changes(target_versions, current_metadata)
            ...
            ...         # Custom: Only consider 'frames' container changes, ignore 'audio'
            ...         # (This example uses Polars - would work similarly with other backends)
            ...         # Users can filter/modify the diff result here
            ...
            ...         return result  # Return modified DiffResult
        """
        return diff_resolver.find_changes(
            target_versions=target_versions,
            current_metadata=current_metadata,
        )
