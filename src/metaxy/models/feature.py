import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
from pydantic._internal._model_construction import ModelMetaclass
from typing_extensions import Self

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.plan import FeaturePlan, FQFieldKey
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    import narwhals as nw

    from metaxy.data_versioning.diff import (
        DiffResult,
        LazyDiffResult,
        MetadataDiffResolver,
    )
    from metaxy.data_versioning.joiners import UpstreamJoiner


# Context variable for active graph (module-level)
_active_graph: ContextVar["FeatureGraph | None"] = ContextVar(
    "_active_graph", default=None
)


def get_feature_by_key(key: "FeatureKey") -> type["Feature"]:
    """Get a feature class by its key from the active graph.

    Convenience function that retrieves from the currently active graph.

    Args:
        key: Feature key to look up

    Returns:
        Feature class

    Raises:
        KeyError: If no feature with the given key is registered

    Example:
        >>> from metaxy import get_feature_by_key, FeatureKey
        >>> parent_key = FeatureKey(["examples", "parent"])
        >>> ParentFeature = get_feature_by_key(parent_key)
    """
    graph = FeatureGraph.get_active()
    return graph.get_feature_by_key(key)


class FeatureGraph:
    def __init__(self):
        self.features_by_key: dict[FeatureKey, type[Feature]] = {}
        self.feature_specs_by_key: dict[FeatureKey, FeatureSpec] = {}

    def add_feature(self, feature: type["Feature"]) -> None:
        """Add a feature to the graph.

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
                f"Each feature key must be unique within a graph."
            )

        self.features_by_key[feature.spec.key] = feature
        self.feature_specs_by_key[feature.spec.key] = feature.spec

    def remove_feature(self, key: FeatureKey) -> None:
        """Remove a feature from the graph.

        Args:
            key: Feature key to remove

        Raises:
            KeyError: If no feature with the given key is registered
        """
        if key not in self.features_by_key:
            raise KeyError(
                f"No feature with key {key.to_string()} found in graph. "
                f"Available keys: {[k.to_string() for k in self.features_by_key.keys()]}"
            )

        del self.features_by_key[key]
        del self.feature_specs_by_key[key]

    def get_feature_by_key(self, key: FeatureKey) -> type["Feature"]:
        """Get a feature class by its key.

        Args:
            key: Feature key to look up

        Returns:
            Feature class

        Raises:
            KeyError: If no feature with the given key is registered

        Example:
            >>> graph = FeatureGraph.get_active()
            >>> parent_key = FeatureKey(["examples", "parent"])
            >>> ParentFeature = graph.get_feature_by_key(parent_key)
        """
        if key not in self.features_by_key:
            raise KeyError(
                f"No feature with key {key.to_string()} found in graph. "
                f"Available keys: {[k.to_string() for k in self.features_by_key.keys()]}"
            )
        return self.features_by_key[key]

    def get_feature_plan(self, key: FeatureKey) -> FeaturePlan:
        feature = self.feature_specs_by_key[key]

        return FeaturePlan(
            feature=feature,
            deps=[self.feature_specs_by_key[dep.key] for dep in feature.deps or []]
            or None,
        )

    def get_field_version(self, key: "FQFieldKey") -> str:
        hasher = hashlib.sha256()

        plan = self.get_feature_plan(key.feature)
        field = plan.feature.fields_by_key[key.field]

        hasher.update(key.to_string().encode())
        hasher.update(str(field.code_version).encode())

        for k, v in sorted(plan.get_parent_fields_for_field(key.field).items()):
            hasher.update(self.get_field_version(k).encode())

        return hasher.hexdigest()

    def get_feature_version_by_field(self, key: FeatureKey) -> dict[str, str]:
        """Computes the feature data version.

        Hash together field data versions versions with the feature code version.

        Returns:
            dict[str, str]: The data version for each field in the feature plan.
                Keys are field names as strings.
        """
        res = {}

        plan = self.get_feature_plan(key)

        for k, v in plan.feature.fields_by_key.items():
            res[k.to_string()] = self.get_field_version(
                FQFieldKey(field=k, feature=key)
            )

        return res

    def get_feature_version(self, key: FeatureKey) -> str:
        """Computes the feature version as a single string"""
        hasher = hashlib.sha256()
        data_version = self.get_feature_version_by_field(key)
        for field_key in sorted(data_version):
            hasher.update(field_key.encode())
            hasher.update(data_version[field_key].encode())

        return hasher.hexdigest()

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
            >>> graph.get_downstream_features([FeatureKey(["A"])])
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

    @property
    def snapshot_version(self) -> str:
        """Generate a snapshot version representing the current topology + versions of the feature graph"""
        if len(self.feature_specs_by_key) == 0:
            return "empty"

        hasher = hashlib.sha256()
        for feature_key in sorted(self.feature_specs_by_key.keys()):
            hasher.update(feature_key.to_string().encode("utf-8"))
            hasher.update(self.get_feature_version(feature_key).encode("utf-8"))
        return hasher.hexdigest()

    def to_snapshot(self) -> dict[str, dict[str, Any]]:
        """Serialize graph to snapshot format.

        Returns a dict mapping feature_key (string) to feature data dict,
        including the import path of the Feature class for reconstruction.

        Returns:
            Dict of feature_key -> {
                feature_spec: dict,
                feature_version: str,
                feature_class_path: str
            }

        Example:
            >>> snapshot = graph.to_snapshot()
            >>> snapshot["video_processing"]["feature_version"]
            'abc12345'
            >>> snapshot["video_processing"]["feature_class_path"]
            'myapp.features.video.VideoProcessing'
        """
        snapshot = {}

        for feature_key, feature_cls in self.features_by_key.items():
            feature_key_str = feature_key.to_string()
            feature_spec_dict = feature_cls.spec.model_dump(mode="json")  # type: ignore[attr-defined]
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
        snapshot_data: dict[str, dict[str, Any]],
        *,
        class_path_overrides: dict[str, str] | None = None,
        force_reload: bool = False,
    ) -> "FeatureGraph":
        """Reconstruct graph from snapshot by importing Feature classes.

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
            force_reload: If True, reload modules from disk to get current code state.

        Returns:
            New FeatureGraph with historical features

        Raises:
            ImportError: If feature class cannot be imported at recorded path

        Example:
            >>> # Load snapshot from metadata store
            >>> historical_graph = FeatureGraph.from_snapshot(snapshot_data)
            >>>
            >>> # With override for moved feature
            >>> historical_graph = FeatureGraph.from_snapshot(
            ...     snapshot_data,
            ...     class_path_overrides={
            ...         "video_processing": "myapp.features_v2.VideoProcessing"
            ...     }
            ... )
        """
        import importlib
        import sys

        from metaxy.models.feature_spec import FeatureSpec

        graph = cls()
        class_path_overrides = class_path_overrides or {}

        # If force_reload, collect all module paths first to remove ALL features
        # from those modules before reloading (modules can have multiple features)
        modules_to_reload = set()
        if force_reload:
            for feature_key_str, feature_data in snapshot_data.items():
                class_path = class_path_overrides.get(
                    feature_key_str
                ) or feature_data.get("feature_class_path")
                if class_path:
                    module_path, _ = class_path.rsplit(".", 1)
                    if module_path in sys.modules:
                        modules_to_reload.add(module_path)

        # Use context manager to temporarily set the new graph as active
        # This ensures imported Feature classes register to the new graph, not the current one
        with graph.use():
            for feature_key_str, feature_data in snapshot_data.items():
                # Parse FeatureSpec for validation
                feature_spec_dict = feature_data["feature_spec"]
                FeatureSpec.model_validate(feature_spec_dict)

                # Get class path (check overrides first)
                if feature_key_str in class_path_overrides:
                    class_path = class_path_overrides[feature_key_str]
                else:
                    class_path = feature_data.get("feature_class_path")
                    if not class_path:
                        raise ValueError(
                            f"Feature '{feature_key_str}' has no feature_class_path in snapshot. "
                            f"Cannot reconstruct historical graph."
                        )

                # Import the class
                try:
                    module_path, class_name = class_path.rsplit(".", 1)

                    # Force reload module from disk if requested
                    # This is critical for migration detection - when code changes,
                    # we need fresh imports to detect the changes
                    if force_reload and module_path in modules_to_reload:
                        # Before first reload of this module, remove ALL features from this module
                        # (a module can define multiple features)
                        if module_path in modules_to_reload:
                            # Find all features from this module in snapshot and remove them
                            for fk_str, fd in snapshot_data.items():
                                fcp = class_path_overrides.get(fk_str) or fd.get(
                                    "feature_class_path"
                                )
                                if fcp and fcp.rsplit(".", 1)[0] == module_path:
                                    fspec_dict = fd["feature_spec"]
                                    fspec = FeatureSpec.model_validate(fspec_dict)
                                    if fspec.key in graph.features_by_key:
                                        graph.remove_feature(fspec.key)

                            # Mark module as processed so we don't remove features again
                            modules_to_reload.discard(module_path)

                        module = importlib.reload(sys.modules[module_path])
                    else:
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
                        f'     {feature_key_str}: "new.module.path.ClassName"\n'
                        f"\n"
                        f"Original error: {e}"
                    ) from e

                # Validate the imported class matches the stored spec
                if not hasattr(feature_cls, "spec"):
                    raise TypeError(
                        f"Imported class '{class_path}' is not a valid Feature class "
                        f"(missing 'spec' attribute)"
                    )

                # Register the imported feature to this graph if not already present
                # If the module was imported for the first time, the metaclass already registered it
                # If the module was previously imported, we need to manually register it
                if feature_cls.spec.key not in graph.features_by_key:
                    graph.add_feature(feature_cls)

        return graph

    @classmethod
    def get_active(cls) -> "FeatureGraph":
        """Get the currently active graph.

        Returns the graph from the context variable if set, otherwise returns
        the default global graph.

        Returns:
            Active FeatureGraph instance

        Example:
            >>> # Normal usage - returns default graph
            >>> reg = FeatureGraph.get_active()
            >>>
            >>> # With custom graph in context
            >>> with my_graph.use():
            ...     reg = FeatureGraph.get_active()  # Returns my_graph
        """
        return _active_graph.get() or graph

    @classmethod
    def set_active(cls, reg: "FeatureGraph") -> None:
        """Set the active graph for the current context.

        This sets the context variable that will be returned by get_active().
        Typically used in application setup code or test fixtures.

        Args:
            reg: FeatureGraph to activate

        Example:
            >>> # In application setup
            >>> my_graph = FeatureGraph()
            >>> FeatureGraph.set_active(my_graph)
            >>>
            >>> # Now all operations use my_graph
            >>> FeatureGraph.get_active()  # Returns my_graph
        """
        _active_graph.set(reg)

    @contextmanager
    def use(self):
        """Context manager to temporarily use this graph as active.

        This is the recommended way to use custom registries, especially in tests.
        The graph is automatically restored when the context exits.

        Yields:
            This graph instance

        Example:
            >>> test_graph = FeatureGraph()
            >>>
            >>> with test_graph.use():
            ...     # All operations use test_graph
            ...     class TestFeature(Feature, spec=...):
            ...         pass
            ...
            >>> # Outside context, back to previous graph
        """
        token = _active_graph.set(self)
        try:
            yield self
        finally:
            _active_graph.reset(token)


# Default global graph
graph = FeatureGraph()


class _FeatureMeta(ModelMetaclass):
    def __new__(
        cls,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *,
        spec: FeatureSpec | None,
        **kwargs,
    ) -> type[Self]:  # pyright: ignore[reportGeneralTypeIssues]
        new_cls = super().__new__(cls, cls_name, bases, namespace)

        if spec:
            # Get graph from context at class definition time
            active_graph = FeatureGraph.get_active()
            new_cls.graph = active_graph  # type: ignore[attr-defined]
            new_cls.spec = spec  # type: ignore[attr-defined]
            active_graph.add_feature(new_cls)
        else:
            pass  # TODO: set spec to a property that would raise an exception on access

        return new_cls


class Feature(FrozenBaseModel, metaclass=_FeatureMeta, spec=None):
    spec: ClassVar[FeatureSpec]
    graph: ClassVar[FeatureGraph]

    @classmethod
    def feature_version(cls) -> str:
        """Get hash of feature specification.

        Returns a hash representing the feature's complete configuration:
        - Feature key
        - Field definitions and code versions
        - Dependencies (feature-level and field-level)

        This hash changes when you modify:
        - Field code versions
        - Dependencies
        - Field definitions

        Used to distinguish current vs historical metadata versions.
        Stored in the 'feature_version' column of metadata DataFrames.

        Returns:
            SHA256 hex digest (like git short hashes)

        Example:
            >>> class MyFeature(Feature, spec=FeatureSpec(
            ...     key=FeatureKey(["my", "feature"]),
            ...     fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ... )):
            ...     pass
            >>> MyFeature.feature_version()
            'a3f8b2c1...'
        """
        return cls.graph.get_feature_version(cls.spec.key)

    @classmethod
    def align_metadata_with_upstream(
        cls,
        current_metadata: pl.DataFrame,
        upstream_metadata: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Align metadata with upstream by joining on sample_uid.

        Override this method to customize alignment logic when upstream features change.
        This is called during migration propagation to determine which samples to process.

        Common use cases:
        - One-to-many mappings (e.g., one video -> many frames)
        - Filtering based on upstream conditions
        - Cross-product joins for combinatorial features
        - Custom sample ID generation

        Default behavior: Inner join on 'sample_uid' with all upstream features.
        Only preserves sample_uid column - all other columns are dropped and will
        be recalculated during data version computation.

        Args:
            current_metadata: Existing metadata for this feature (may be empty)
            upstream_metadata: Dict mapping upstream feature keys to their metadata DataFrames.
                Each DataFrame includes sample_uid, data_version, and other columns.

        Returns:
            DataFrame with 'sample_uid' column (at minimum), ready for data version calculation.
            Other columns are preserved if present in current_metadata.

        Example - One-to-many (video frames):
            >>> @classmethod
            >>> def align_metadata_with_upstream(
            ...     cls,
            ...     current_metadata: pl.DataFrame,
            ...     upstream_metadata: dict[str, pl.DataFrame],
            ... ) -> pl.DataFrame:
            ...     # Each video produces 30 frames
            ...     video_samples = upstream_metadata["videos"]["sample_uid"]
            ...     frames = []
            ...     for video_id in video_samples:
            ...         for frame_idx in range(30):
            ...             frames.append({
            ...                 "sample_uid": f"{video_id}_frame_{frame_idx}",
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
            ...     return videos.filter(pl.col("duration") > 10).select(["sample_uid", "duration"])

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
            ...         all_samples.update(upstream_df["sample_uid"].to_list())
            ...     return pl.DataFrame({"sample_uid": sorted(all_samples)})
        """
        if not upstream_metadata:
            # No upstream, return current metadata with only sample_uid
            if len(current_metadata) > 0:
                return current_metadata.select(pl.col("sample_uid"))
            else:
                return pl.DataFrame({"sample_uid": []})

        # Default: inner join on sample_uid across all upstream features
        # This ensures we only process samples that exist in ALL upstream features
        common_samples: set[int] | None = None
        for upstream_df in upstream_metadata.values():
            sample_uids = set(upstream_df["sample_uid"].to_list())
            if common_samples is None:
                common_samples = sample_uids
            else:
                common_samples &= sample_uids  # Intersection

        if not common_samples:
            return pl.DataFrame({"sample_uid": []})

        # Filter current metadata to common samples, preserving existing columns
        if len(current_metadata) > 0:
            return current_metadata.filter(
                pl.col("sample_uid").is_in(list(common_samples))
            )
        else:
            # No current metadata, create from upstream sample IDs
            return pl.DataFrame({"sample_uid": sorted(common_samples)})

    @classmethod
    def data_version(cls) -> dict[str, str]:
        """Get the code-level data version for this feature.

        This returns a static hash based on code versions and dependencies,
        not sample-level data versions.

        Returns:
            Dictionary mapping field keys to their data version hashes.
        """
        return cls.graph.get_feature_version_by_field(cls.spec.key)

    @classmethod
    def join_upstream_metadata(
        cls,
        joiner: "UpstreamJoiner",
        upstream_refs: dict[str, "nw.LazyFrame[Any]"],
    ) -> tuple["nw.LazyFrame[Any]", dict[str, str]]:
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
            ...             feature_plan=cls.graph.get_feature_plan(cls.spec.key),
            ...         )
        """
        return joiner.join_upstream(
            upstream_refs=upstream_refs,
            feature_spec=cls.spec,
            feature_plan=cls.graph.get_feature_plan(cls.spec.key),
        )

    @classmethod
    def resolve_data_version_diff(
        cls,
        diff_resolver: "MetadataDiffResolver",
        target_versions: "nw.LazyFrame[Any]",
        current_metadata: "nw.LazyFrame[Any] | None",
        *,
        lazy: bool = False,
    ) -> "DiffResult | LazyDiffResult":
        """Resolve differences between target and current data versions.

        Override for custom diff logic (ignore certain fields, custom rules, etc.).

        Args:
            diff_resolver: MetadataDiffResolver from MetadataStore
            target_versions: Calculated target data_versions (Narwhals LazyFrame)
            current_metadata: Current metadata for this feature (Narwhals LazyFrame, or None).
                Should be pre-filtered by feature_version at the store level.
            lazy: If True, return LazyDiffResult. If False, return DiffResult.

        Returns:
            DiffResult (eager) or LazyDiffResult (lazy) with added, changed, removed

        Example (default):
            >>> class MyFeature(Feature, spec=...):
            ...     pass  # Uses diff resolver's default implementation

        Example (ignore certain field changes):
            >>> class MyFeature(Feature, spec=...):
            ...     @classmethod
            ...     def resolve_data_version_diff(cls, diff_resolver, target_versions, current_metadata, **kwargs):
            ...         # Get standard diff
            ...         result = diff_resolver.find_changes(target_versions, current_metadata)
            ...
            ...         # Custom: Only consider 'frames' field changes, ignore 'audio'
            ...         # Users can filter/modify the diff result here
            ...
            ...         return result  # Return modified DiffResult
        """
        # Diff resolver always returns LazyDiffResult - materialize if needed
        lazy_result = diff_resolver.find_changes(
            target_versions=target_versions,
            current_metadata=current_metadata,
        )

        # Materialize to DiffResult if lazy=False
        if not lazy:
            from metaxy.data_versioning.diff import DiffResult

            return DiffResult(
                added=lazy_result.added.collect(),
                changed=lazy_result.changed.collect(),
                removed=lazy_result.removed.collect(),
            )

        return lazy_result
