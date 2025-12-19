import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

import pydantic
from pydantic import AwareDatetime, Field, model_validator
from pydantic._internal._model_construction import ModelMetaclass
from typing_extensions import Self

from metaxy.models.constants import (
    METAXY_FEATURE_SPEC_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_FULL_DEFINITION_VERSION,
)
from metaxy.models.feature_spec import (
    FeatureSpec,
)
from metaxy.models.plan import FeaturePlan, FQFieldKey
from metaxy.models.types import (
    CoercibleToFeatureKey,
    FeatureKey,
    ValidatedFeatureKeyAdapter,
    ValidatedFeatureKeySequenceAdapter,
)
from metaxy.utils.hashing import truncate_hash

FEATURE_VERSION_COL = METAXY_FEATURE_VERSION
FEATURE_SPEC_VERSION_COL = METAXY_FEATURE_SPEC_VERSION
FEATURE_TRACKING_VERSION_COL = METAXY_FULL_DEFINITION_VERSION

if TYPE_CHECKING:
    import narwhals as nw

    from metaxy.versioning.types import Increment, LazyIncrement

    # TODO: These are no longer used - remove after refactoring
    # from metaxy.data_versioning.diff import MetadataDiffResolver
    # from metaxy.data_versioning.joiners import UpstreamJoiner

# Context variable for active graph (module-level)
_active_graph: ContextVar["FeatureGraph | None"] = ContextVar(
    "_active_graph", default=None
)


def get_feature_by_key(key: CoercibleToFeatureKey) -> type["BaseFeature"]:
    """Get a feature class by its key from the active graph.

    Convenience function that retrieves Metaxy feature class from the currently active [feature graph][metaxy.FeatureGraph]. Can be useful when receiving a feature key from storage or across process boundaries.

    Args:
        key: Feature key to look up. Accepts types that can be converted into a feature key..

    Returns:
        Feature class

    Raises:
        KeyError: If no feature with the given key is registered

    Example:
        ```py
        from metaxy import get_feature_by_key, FeatureKey
        parent_key = FeatureKey(["examples", "parent"])
        ParentFeature = get_feature_by_key(parent_key)

        # Or use string notation
        ParentFeature = get_feature_by_key("examples/parent")
        ```
    """
    graph = FeatureGraph.get_active()
    return graph.get_feature_by_key(key)


class SerializedFeature(TypedDict):
    feature_spec: dict[str, Any]
    feature_schema: dict[str, Any]
    metaxy_feature_version: str
    metaxy_feature_spec_version: str
    metaxy_full_definition_version: str
    feature_class_path: str
    project: str


class FeatureGraph:
    def __init__(self):
        self.features_by_key: dict[FeatureKey, type[BaseFeature]] = {}
        self.feature_specs_by_key: dict[FeatureKey, FeatureSpec] = {}
        # Standalone specs registered without Feature classes (for migrations)
        self.standalone_specs_by_key: dict[FeatureKey, FeatureSpec] = {}

    @property
    def all_specs_by_key(self) -> dict[FeatureKey, FeatureSpec]:
        return {**self.feature_specs_by_key, **self.standalone_specs_by_key}

    def add_feature(self, feature: type["BaseFeature"]) -> None:
        """Add a feature to the graph.

        Args:
            feature: Feature class to register

        Raises:
            ValueError: If a feature with a different import path but the same key is already registered
                       or if duplicate column names would result from renaming operations
        """
        if feature.spec().key in self.features_by_key:
            existing = self.features_by_key[feature.spec().key]
            # Allow quiet replacement if it's the same class (same import path)
            existing_path = f"{existing.__module__}.{existing.__name__}"
            new_path = f"{feature.__module__}.{feature.__name__}"
            if existing_path != new_path:
                raise ValueError(
                    f"Feature with key {feature.spec().key.to_string()} already registered. "
                    f"Existing: {existing_path}, New: {new_path}. "
                    f"Each feature key must be unique within a graph."
                )

        # Validation happens automatically when FeaturePlan is constructed via get_feature_plan()
        # We trigger it here to catch errors at definition time
        if feature.spec().deps:
            self._build_and_validate_plan(feature.spec())

        self.features_by_key[feature.spec().key] = feature
        self.feature_specs_by_key[feature.spec().key] = feature.spec()

    def add_feature_spec(self, spec: FeatureSpec) -> None:
        import warnings

        # Check if a Feature class already exists for this key
        if spec.key in self.features_by_key:
            warnings.warn(
                f"Feature class already exists for key {spec.key.to_string()}. "
                f"Standalone spec will be ignored - Feature class takes precedence.",
                stacklevel=2,
            )
            return

        # Check if a standalone spec already exists
        if spec.key in self.standalone_specs_by_key:
            existing = self.standalone_specs_by_key[spec.key]
            # Only warn if it's a different spec (by comparing feature_spec_version)
            if existing.feature_spec_version != spec.feature_spec_version:
                raise ValueError(
                    f"Standalone spec for key {spec.key.to_string()} already exists "
                    f"with a different version."
                )

        # Validation happens automatically when FeaturePlan is constructed
        if spec.deps:
            self._build_and_validate_plan(spec)

        # Store standalone spec
        self.standalone_specs_by_key[spec.key] = spec
        # Also add to feature_specs_by_key for methods that only need the spec
        self.feature_specs_by_key[spec.key] = spec

    def _build_and_validate_plan(self, spec: "FeatureSpec") -> None:
        """Build a FeaturePlan to trigger validation.

        FeaturePlan validates column configuration on construction.
        We skip validation if upstream specs are missing (will be validated later).
        """
        from metaxy.models.feature_spec import FeatureDep

        parent_specs = []
        for dep in spec.deps or []:
            if not isinstance(dep, FeatureDep):
                continue
            upstream_spec = self.feature_specs_by_key.get(dep.feature)
            if upstream_spec:
                parent_specs.append(upstream_spec)

        # Skip if any upstream spec is missing (will be validated later)
        feature_dep_count = len(
            [d for d in (spec.deps or []) if isinstance(d, FeatureDep)]
        )
        if len(parent_specs) != feature_dep_count:
            return

        # Constructing FeaturePlan triggers validation
        FeaturePlan(
            feature=spec,
            deps=parent_specs or None,
            feature_deps=spec.deps,
        )

    def remove_feature(self, key: CoercibleToFeatureKey) -> None:
        """Remove a feature from the graph.

        Removes Feature class or standalone spec (whichever exists).

        Args:
            key: Feature key to remove. Accepts types that can be converted into a feature key..

        Raises:
            KeyError: If no feature with the given key is registered
        """
        # Validate and coerce the key
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)

        # Check both Feature classes and standalone specs
        combined = {**self.feature_specs_by_key, **self.standalone_specs_by_key}

        if validated_key not in combined:
            raise KeyError(
                f"No feature with key {validated_key.to_string()} found in graph. "
                f"Available keys: {[k.to_string() for k in combined]}"
            )

        # Remove from all relevant dicts
        if validated_key in self.features_by_key:
            del self.features_by_key[validated_key]
        if validated_key in self.standalone_specs_by_key:
            del self.standalone_specs_by_key[validated_key]
        if validated_key in self.feature_specs_by_key:
            del self.feature_specs_by_key[validated_key]

    def get_feature_by_key(self, key: CoercibleToFeatureKey) -> type["BaseFeature"]:
        """Get a feature class by its key.

        Args:
            key: Feature key to look up. Accepts types that can be converted into a feature key..

        Returns:
            Feature class

        Raises:
            KeyError: If no feature with the given key is registered

        Example:
            ```py
            graph = FeatureGraph.get_active()
            parent_key = FeatureKey(["examples", "parent"])
            ParentFeature = graph.get_feature_by_key(parent_key)

            # Or use string notation
            ParentFeature = graph.get_feature_by_key("examples/parent")
            ```
        """
        # Validate and coerce the key
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)

        if validated_key not in self.features_by_key:
            raise KeyError(
                f"No feature with key {validated_key.to_string()} found in graph. "
                f"Available keys: {[k.to_string() for k in self.features_by_key.keys()]}"
            )
        return self.features_by_key[validated_key]

    def list_features(
        self,
        projects: list[str] | str | None = None,
        *,
        only_current_project: bool = True,
    ) -> list[FeatureKey]:
        """List all feature keys in the graph, optionally filtered by project(s).

        By default, filters features by the current project (first part of feature key).
        This prevents operations from affecting features in other projects.

        Args:
            projects: Project name(s) to filter by. Can be:
                - None: Use current project from MetaxyConfig (if only_current_project=True)
                - str: Single project name
                - list[str]: Multiple project names
            only_current_project: If True, filter by current/specified project(s).
                If False, return all features regardless of project.

        Returns:
            List of feature keys

        Example:
            ```py
            # Get all features for current project
            graph = FeatureGraph.get_active()
            features = graph.list_features()

            # Get features for specific project
            features = graph.list_features(projects="myproject")

            # Get features for multiple projects
            features = graph.list_features(projects=["project1", "project2"])

            # Get all features regardless of project
            all_features = graph.list_features(only_current_project=False)
            ```
        """
        if not only_current_project:
            # Return all features
            return list(self.features_by_key.keys())

        # Normalize projects to list
        project_list: list[str]
        if projects is None:
            # Try to get from config context
            try:
                from metaxy.config import MetaxyConfig

                config = MetaxyConfig.get()
                project_list = [config.project]
            except RuntimeError:
                # Config not initialized - in tests or non-CLI usage
                # Return all features (can't determine project)
                return list(self.features_by_key.keys())
        elif isinstance(projects, str):
            project_list = [projects]
        else:
            project_list = projects

        # Filter by project(s) using Feature.project attribute
        return [
            key
            for key in self.features_by_key.keys()
            if self.features_by_key[key].project in project_list
        ]

    def get_feature_plan(self, key: CoercibleToFeatureKey) -> FeaturePlan:
        """Get a feature plan for a given feature key.

        Args:
            key: Feature key to get plan for. Accepts types that can be converted into a feature key..

        Returns:
            FeaturePlan instance with feature spec and dependencies.
        """
        # Validate and coerce the key
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)

        spec = self.all_specs_by_key[validated_key]

        return FeaturePlan(
            feature=spec,
            deps=[self.feature_specs_by_key[dep.feature] for dep in spec.deps or []]
            or None,
            feature_deps=spec.deps,  # Pass the actual FeatureDep objects with field mappings
        )

    def get_field_version(self, key: "FQFieldKey") -> str:
        hasher = hashlib.sha256()

        plan = self.get_feature_plan(key.feature)
        field = plan.feature.fields_by_key[key.field]

        hasher.update(key.to_string().encode())
        hasher.update(str(field.code_version).encode())

        for k, v in sorted(plan.get_parent_fields_for_field(key.field).items()):
            hasher.update(self.get_field_version(k).encode())

        return truncate_hash(hasher.hexdigest())

    def get_feature_version_by_field(
        self, key: CoercibleToFeatureKey
    ) -> dict[str, str]:
        """Computes the field provenance map for a feature.

        Hash together field provenance entries with the feature code version.

        Args:
            key: Feature key to get field versions for. Accepts types that can be converted into a feature key..

        Returns:
            dict[str, str]: The provenance hash for each field in the feature plan.
                Keys are field names as strings.
        """
        # Validate and coerce the key
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)

        res = {}

        plan = self.get_feature_plan(validated_key)

        for k, v in plan.feature.fields_by_key.items():
            res[k.to_string()] = self.get_field_version(
                FQFieldKey(field=k, feature=validated_key)
            )

        return res

    def get_feature_version(self, key: CoercibleToFeatureKey) -> str:
        """Computes the feature version as a single string.

        Args:
            key: Feature key to get version for. Accepts types that can be converted into a feature key..

        Returns:
            Truncated SHA256 hash representing the feature version.
        """
        # Validate and coerce the key
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)

        hasher = hashlib.sha256()
        provenance_by_field = self.get_feature_version_by_field(validated_key)
        for field_key in sorted(provenance_by_field):
            hasher.update(field_key.encode())
            hasher.update(provenance_by_field[field_key].encode())

        return truncate_hash(hasher.hexdigest())

    def get_downstream_features(
        self, sources: Sequence[CoercibleToFeatureKey]
    ) -> list[FeatureKey]:
        """Get all features downstream of sources, topologically sorted.

        Performs a depth-first traversal of the dependency graph to find all
        features that transitively depend on any of the source features.

        Args:
            sources: List of source feature keys. Each element can be string, sequence, FeatureKey, or BaseFeature class.

        Returns:
            List of downstream feature keys in topological order (dependencies first).
            Does not include the source features themselves.

        Example:
            ```py
            # DAG: A -> B -> D
            #      A -> C -> D
            graph.get_downstream_features([FeatureKey(["A"])])
            # [FeatureKey(["B"]), FeatureKey(["C"]), FeatureKey(["D"])]

            # Or use string notation
            graph.get_downstream_features(["A"])
            ```
        """
        # Validate and coerce the source keys
        validated_sources = ValidatedFeatureKeySequenceAdapter.validate_python(sources)

        source_set = set(validated_sources)
        visited = set()
        post_order = []
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
                        if dep.feature == key:
                            # This feature depends on 'key', so visit it
                            visit(feature_key)

            post_order.append(key)

        # Visit all sources
        for source in validated_sources:
            visit(source)

        # Remove sources from result, reverse to get topological order
        result = [k for k in reversed(post_order) if k not in source_set]
        return result

    def topological_sort_features(
        self,
        feature_keys: Sequence[CoercibleToFeatureKey] | None = None,
        *,
        descending: bool = False,
    ) -> list[FeatureKey]:
        """Sort feature keys in topological order.

        Uses stable alphabetical ordering when multiple nodes are at the same level.
        This ensures deterministic output for diff comparisons and migrations.

        Implemented using depth-first search with post-order traversal.

        Args:
            feature_keys: List of feature keys to sort. Each element can be string, sequence,
                FeatureKey, or BaseFeature class. If None, sorts all features
                (both Feature classes and standalone specs) in the graph.
            descending: If False (default), dependencies appear before dependents.
                For a chain A -> B -> C, returns [A, B, C].
                If True, dependents appear before dependencies.
                For a chain A -> B -> C, returns [C, B, A].

        Returns:
            List of feature keys sorted in topological order

        Example:
            ```py
            graph = FeatureGraph.get_active()
            # Sort specific features (dependencies first)
            sorted_keys = graph.topological_sort_features([
                FeatureKey(["video", "raw"]),
                FeatureKey(["video", "scene"]),
            ])

            # Or use string notation
            sorted_keys = graph.topological_sort_features(["video/raw", "video/scene"])

            # Sort all features in the graph (including standalone specs)
            all_sorted = graph.topological_sort_features()

            # Sort with dependents first (useful for processing leaf nodes before roots)
            reverse_sorted = graph.topological_sort_features(descending=True)
            ```
        """
        # Determine which features to sort
        if feature_keys is None:
            # Include both Feature classes and standalone specs
            keys_to_sort = set(self.feature_specs_by_key.keys())
        else:
            # Validate and coerce the feature keys
            validated_keys = ValidatedFeatureKeySequenceAdapter.validate_python(
                feature_keys
            )
            keys_to_sort = set(validated_keys)

        visited = set()
        result = []  # Topological order (dependencies first)

        def visit(key: FeatureKey):
            """DFS visit with post-order traversal."""
            if key in visited or key not in keys_to_sort:
                return
            visited.add(key)

            # Get dependencies from feature spec
            spec = self.feature_specs_by_key.get(key)
            if spec and spec.deps:
                # Sort dependencies alphabetically for deterministic ordering
                sorted_deps = sorted(
                    (dep.feature for dep in spec.deps),
                    key=lambda k: k.to_string().lower(),
                )
                for dep_key in sorted_deps:
                    if dep_key in keys_to_sort:
                        visit(dep_key)

            # Add to result after visiting dependencies (post-order)
            result.append(key)

        # Visit all keys in sorted order for deterministic traversal
        for key in sorted(keys_to_sort, key=lambda k: k.to_string().lower()):
            visit(key)

        # Post-order DFS gives topological order (dependencies before dependents)
        if descending:
            return list(reversed(result))
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
        return truncate_hash(hasher.hexdigest())

    def to_snapshot(self) -> dict[str, SerializedFeature]:
        """Serialize graph to snapshot format.

        Returns a dict mapping feature_key (string) to feature data dict,
        including the import path of the Feature class for reconstruction.

        Returns: dictionary mapping feature_key (string) to feature data dict

        Example:
            ```py
            snapshot = graph.to_snapshot()
            snapshot["video_processing"]["metaxy_feature_version"]
            # 'abc12345'
            snapshot["video_processing"]["metaxy_feature_spec_version"]
            # 'def67890'
            snapshot["video_processing"]["metaxy_full_definition_version"]
            # 'xyz98765'
            snapshot["video_processing"]["feature_class_path"]
            # 'myapp.features.video.VideoProcessing'
            snapshot["video_processing"]["project"]
            # 'myapp'
            ```
        """
        snapshot: dict[str, SerializedFeature] = {}

        for feature_key, feature_cls in self.features_by_key.items():
            feature_key_str = feature_key.to_string()
            feature_spec_dict = feature_cls.spec().model_dump(mode="json")
            feature_schema_dict = feature_cls.model_json_schema()
            feature_version = feature_cls.feature_version()
            feature_spec_version = feature_cls.spec().feature_spec_version
            full_definition_version = feature_cls.full_definition_version()
            project = feature_cls.project

            # Get class import path (module.ClassName)
            class_path = f"{feature_cls.__module__}.{feature_cls.__name__}"

            snapshot[feature_key_str] = {  # ty: ignore[invalid-assignment]
                "feature_spec": feature_spec_dict,
                "feature_schema": feature_schema_dict,
                FEATURE_VERSION_COL: feature_version,
                FEATURE_SPEC_VERSION_COL: feature_spec_version,
                FEATURE_TRACKING_VERSION_COL: full_definition_version,
                "feature_class_path": class_path,
                "project": project,
            }

        return snapshot

    @classmethod
    def from_snapshot(
        cls,
        snapshot_data: Mapping[str, Mapping[str, Any]],
        *,
        class_path_overrides: dict[str, str] | None = None,
        force_reload: bool = False,
    ) -> "FeatureGraph":
        """Reconstruct graph from snapshot by importing Feature classes.

        Strictly requires Feature classes to exist at their recorded import paths.
        This ensures custom methods (like load_input) are available.

        If a feature has been moved/renamed, use class_path_overrides to specify
        the new location.

        Args:
            snapshot_data: Dict of feature_key -> dict containing
                feature_spec (dict), feature_class_path (str), and other fields
                as returned by to_snapshot() or loaded from DB
            class_path_overrides: Optional dict mapping feature_key to new class path
                                 for features that have been moved/renamed
            force_reload: If True, reload modules from disk to get current code state.

        Returns:
            New FeatureGraph with historical features

        Raises:
            ImportError: If feature class cannot be imported at recorded path

        Example:
            ```py
            # Load snapshot from metadata store
            historical_graph = FeatureGraph.from_snapshot(snapshot_data)

            # With override for moved feature
            historical_graph = FeatureGraph.from_snapshot(
                snapshot_data,
                class_path_overrides={
                    "video_processing": "myapp.features_v2.VideoProcessing"
                }
            )
            ```
        """
        import importlib
        import sys

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
                except (ImportError, AttributeError):
                    # Feature class not importable - add as standalone spec instead
                    # This allows migrations to work even when old Feature classes are deleted/moved
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.exception(
                        f"Cannot import Feature class '{class_path}' for '{feature_key_str}'. "
                        f"Adding only the FeatureSpec. "
                    )

                    feature_spec = FeatureSpec.model_validate(feature_spec_dict)
                    # Add the spec as a standalone spec
                    graph.add_feature_spec(feature_spec)
                    continue

                # Validate the imported class matches the stored spec
                if not hasattr(feature_cls, "spec"):
                    raise TypeError(
                        f"Imported class '{class_path}' is not a valid Feature class "
                        f"(missing 'spec' attribute)"
                    )

                # Register the imported feature to this graph if not already present
                # If the module was imported for the first time, the metaclass already registered it
                # If the module was previously imported, we need to manually register it
                if feature_cls.spec().key not in graph.features_by_key:
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
            ```py
            # Normal usage - returns default graph
            reg = FeatureGraph.get_active()

            # With custom graph in context
            with my_graph.use():
                reg = FeatureGraph.get_active()  # Returns my_graph
            ```
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
            ```py
            # In application setup
            my_graph = FeatureGraph()
            FeatureGraph.set_active(my_graph)

            # Now all operations use my_graph
            FeatureGraph.get_active()  # Returns my_graph
            ```
        """
        _active_graph.set(reg)

    @contextmanager
    def use(self) -> Iterator[Self]:
        """Context manager to temporarily use this graph as active.

        This is the recommended way to use custom registries, especially in tests.
        The graph is automatically restored when the context exits.

        Yields:
            FeatureGraph: This graph instance

        Example:
            ```py
            test_graph = FeatureGraph()

            with test_graph.use():
                # All operations use test_graph
                class TestFeature(Feature, spec=...):
                    pass

            # Outside context, back to previous graph
            ```
        """
        token = _active_graph.set(self)
        try:
            yield self
        finally:
            _active_graph.reset(token)


def current_graph() -> FeatureGraph:
    """Get the currently active graph.

    Returns:
        FeatureGraph: The currently active graph.
    """
    return FeatureGraph.get_active()


# Default global graph
graph = FeatureGraph()


class MetaxyMeta(ModelMetaclass):
    def __new__(
        cls,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *,
        spec: FeatureSpec | None = None,
        **kwargs,
    ) -> type[Self]:
        # Inject frozen config if not already specified in namespace
        if "model_config" not in namespace:
            from pydantic import ConfigDict

            namespace["model_config"] = ConfigDict(frozen=True, extra="forbid")

        new_cls = super().__new__(cls, cls_name, bases, namespace, **kwargs)

        if spec:
            # Get graph from context at class definition time
            active_graph = FeatureGraph.get_active()
            new_cls.graph = active_graph
            new_cls._spec = spec

            # Determine project for this feature using intelligent detection
            project = cls._detect_project(new_cls)
            new_cls.project = project

            active_graph.add_feature(new_cls)
        else:
            pass  # TODO: set spec to a property that would raise an exception on access

        return new_cls

    @staticmethod
    def _detect_project(feature_cls: type) -> str:
        """Detect project for a feature class.

        Detection order:
        1. Try to auto-load MetaxyConfig from metaxy.toml/pyproject.toml
           starting from the feature's file location
        2. Use config.project if available
        3. Check metaxy.projects entry points as fallback
        4. Fall back to "default" with a warning

        Args:
            feature_cls: The Feature class being registered

        Returns:
            Project name string
        """
        import inspect
        import warnings
        from pathlib import Path

        from metaxy._packaging import detect_project_from_entrypoints
        from metaxy.config import MetaxyConfig

        module_name = feature_cls.__module__

        # Strategy 1: Try to load config if not already set
        if not MetaxyConfig.is_set():
            # Get the file where the feature class is defined
            feature_file = inspect.getfile(feature_cls)
            feature_dir = Path(feature_file).parent

            # Attempt to auto-load config from metaxy.toml or pyproject.toml
            # starting from the feature's directory
            config = MetaxyConfig.load(
                search_parents=True, auto_discovery_start=feature_dir
            )
            return config.project
        else:
            # Config already set, use it
            config = MetaxyConfig.get()
            return config.project

        # Strategy 2: Check metaxy.projects entry points as fallback
        project = detect_project_from_entrypoints(module_name)
        if project is not None:
            return project

        # Strategy 3: Fall back to "default" with a warning
        warnings.warn(
            f"Could not detect project for feature '{feature_cls.__name__}' "
            f"from module '{module_name}'. No metaxy.toml found and no entry point configured. "
            f"Using 'default' as project name. This may cause issues with metadata isolation. "
            f"Please ensure features are imported after init_metaxy() or configure a metaxy.toml file.",
            stacklevel=3,
        )
        return "default"


class _FeatureSpecDescriptor:
    """Descriptor that returns the feature spec of the feature."""

    def __get__(self, instance, owner) -> str:
        if owner.spec is None:
            raise ValueError(f"Feature '{owner.__name__}' has no spec defined.")
        return owner.spec


class BaseFeature(pydantic.BaseModel, metaclass=MetaxyMeta, spec=None):
    _spec: ClassVar[FeatureSpec]

    graph: ClassVar[FeatureGraph]
    project: ClassVar[str]

    # System columns - automatically managed by Metaxy
    # Most of them are optional since Metaxy injects them into dataframes at some point
    metaxy_provenance_by_field: dict[str, str] = Field(
        default_factory=dict,
        description="Field-level provenance hashes (maps field names to hashes)",
    )
    metaxy_provenance: str | None = Field(
        default=None,
        description="Hash of metaxy_provenance_by_field",
    )
    metaxy_feature_version: str | None = Field(
        default=None,
        description="Hash of the feature definition (dependencies + fields + code_versions)",
    )
    metaxy_snapshot_version: str | None = Field(
        default=None,
        description="Hash of the entire feature graph snapshot",
    )
    metaxy_data_version_by_field: dict[str, str] | None = Field(
        default=None,
        description="Field-level data version hashes (maps field names to version hashes)",
    )
    metaxy_data_version: str | None = Field(
        default=None,
        description="Hash of metaxy_data_version_by_field",
    )
    metaxy_created_at: AwareDatetime | None = Field(
        default=None,
        description="Timestamp when the metadata row was created (UTC)",
    )
    metaxy_materialization_id: str | None = Field(
        default=None,
        description="External orchestration run ID (e.g., Dagster Run ID)",
    )

    @model_validator(mode="after")
    def _validate_id_columns_exist(self) -> Self:
        """Validate that all id_columns from spec are present in model fields."""
        spec = self.__class__.spec()
        model_fields = set(self.__class__.model_fields.keys())

        missing_columns = set(spec.id_columns) - model_fields
        if missing_columns:
            raise ValueError(
                f"ID columns {missing_columns} specified in spec are not present in model fields. "
                f"Available fields: {model_fields}"
            )
        return self

    @classmethod
    def spec(cls) -> FeatureSpec:
        return cls._spec

    @classmethod
    def table_name(cls) -> str:
        """Get SQL-like table name for this feature.

        Converts feature key to SQL-compatible table name by joining
        parts with double underscores, consistent with IbisMetadataStore.

        Returns:
            Table name string (e.g., "my_namespace__my_feature")

        Example:
            ```py
            class VideoFeature(Feature, spec=FeatureSpec(
                key=FeatureKey(["video", "processing"]),
                ...
            )):
                pass
            VideoFeature.table_name()
            # 'video__processing'
            ```
        """
        return cls.spec().table_name()

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

        Used to distinguish current vs historical metafield provenance hashes.
        Stored in the 'metaxy_feature_version' column of metadata DataFrames.

        Returns:
            SHA256 hex digest (like git short hashes)

        Example:
            ```py
            class MyFeature(Feature, spec=FeatureSpec(
                key=FeatureKey(["my", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            )):
                pass
            MyFeature.feature_version()
            # 'a3f8b2c1...'
            ```
        """
        return cls.graph.get_feature_version(cls.spec().key)

    @classmethod
    def feature_spec_version(cls) -> str:
        """Get hash of the complete feature specification.

        Returns a hash representing ALL specification properties including:
        - Feature key
        - Dependencies
        - Fields
        - Code versions
        - Any future metadata, tags, or other properties

        Unlike feature_version which only hashes computational properties
        (for migration triggering), feature_spec_version captures the entire specification
        for complete reproducibility and audit purposes.

        Stored in the 'metaxy_feature_spec_version' column of metadata DataFrames.

        Returns:
            SHA256 hex digest of the complete specification

        Example:
            ```py
            class MyFeature(Feature, spec=FeatureSpec(
                key=FeatureKey(["my", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            )):
                pass
            MyFeature.feature_spec_version()
            # 'def456...'  # Different from feature_version
            ```
        """
        return cls.spec().feature_spec_version

    @classmethod
    def full_definition_version(cls) -> str:
        """Get hash of the complete feature definition including Pydantic schema.

        This method computes a hash of the entire feature class definition, including:
        - Pydantic model schema
        - Project name

        Used in the `metaxy_full_definition_version` column of system tables.

        Returns:
            SHA256 hex digest of the complete definition
        """
        import json

        hasher = hashlib.sha256()

        # Hash the Pydantic schema (includes field types, descriptions, validators, etc.)
        schema = cls.model_json_schema()
        schema_json = json.dumps(schema, sort_keys=True)
        hasher.update(schema_json.encode())

        # Hash the feature specification
        hasher.update(cls.feature_spec_version().encode())

        # Hash the project name
        hasher.update(cls.project.encode())

        return truncate_hash(hasher.hexdigest())

    @classmethod
    def provenance_by_field(cls) -> dict[str, str]:
        """Get the code-level field provenance for this feature.

        This returns a static hash based on code versions and dependencies,
        not sample-level field provenance computed from upstream data.

        Returns:
            Dictionary mapping field keys to their provenance hashes.
        """
        return cls.graph.get_feature_version_by_field(cls.spec().key)

    @classmethod
    def load_input(
        cls,
        joiner: Any,
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
        """
        from metaxy.models.feature_spec import FeatureDep

        # Extract columns and renames from deps
        upstream_columns: dict[str, tuple[str, ...] | None] = {}
        upstream_renames: dict[str, dict[str, str] | None] = {}

        deps = cls.spec().deps
        if deps:
            for dep in deps:
                if isinstance(dep, FeatureDep):
                    dep_key_str = dep.feature.to_string()
                    upstream_columns[dep_key_str] = dep.columns
                    upstream_renames[dep_key_str] = dep.rename

        return joiner.join_upstream(
            upstream_refs=upstream_refs,
            feature_spec=cls.spec(),
            feature_plan=cls.graph.get_feature_plan(cls.spec().key),
            upstream_columns=upstream_columns,
            upstream_renames=upstream_renames,
        )

    @classmethod
    def resolve_data_version_diff(
        cls,
        diff_resolver: Any,
        target_provenance: "nw.LazyFrame[Any]",
        current_metadata: "nw.LazyFrame[Any] | None",
        *,
        lazy: bool = False,
    ) -> "Increment | LazyIncrement":
        """Resolve differences between target and current field provenance.

        Override for custom diff logic (ignore certain fields, custom rules, etc.).

        Args:
            diff_resolver: MetadataDiffResolver from MetadataStore
            target_provenance: Calculated target field provenance (Narwhals LazyFrame)
            current_metadata: Current metadata for this feature (Narwhals LazyFrame, or None).
                Should be pre-filtered by feature_version at the store level.
            lazy: If True, return LazyIncrement. If False, return Increment.

        Returns:
            Increment (eager) or LazyIncrement (lazy) with added, changed, removed

        Example (default):
            ```py
            class MyFeature(Feature, spec=...):
                pass  # Uses diff resolver's default implementation
            ```

        Example (ignore certain field changes):
            ```py
            class MyFeature(Feature, spec=...):
                @classmethod
                def resolve_data_version_diff(cls, diff_resolver, target_provenance, current_metadata, **kwargs):
                    # Get standard diff
                    result = diff_resolver.find_changes(target_provenance, current_metadata, cls.spec().id_columns)

                    # Custom: Only consider 'frames' field changes, ignore 'audio'
                    # Users can filter/modify the increment here

                    return result  # Return modified Increment
            ```
        """
        # Diff resolver always returns LazyIncrement - materialize if needed
        lazy_result = diff_resolver.find_changes(
            target_provenance=target_provenance,
            current_metadata=current_metadata,
            id_columns=cls.spec().id_columns,  # Pass ID columns from feature spec
        )

        # Materialize to Increment if lazy=False
        if not lazy:
            from metaxy.versioning.types import Increment

            return Increment(
                added=lazy_result.added.collect(),
                changed=lazy_result.changed.collect(),
                removed=lazy_result.removed.collect(),
            )

        return lazy_result
