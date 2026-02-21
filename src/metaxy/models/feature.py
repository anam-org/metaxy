import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict

import pydantic
from pydantic import AwareDatetime, Field, model_validator
from pydantic._internal._model_construction import ModelMetaclass
from typing_extensions import Self

from metaxy._decorators import public
from metaxy._hashing import truncate_hash
from metaxy.models.constants import (
    METAXY_DEFINITION_VERSION,
    METAXY_FEATURE_VERSION,
)
from metaxy.models.feature_definition import FeatureDefinition
from metaxy.models.feature_spec import (
    FeatureSpec,
)
from metaxy.models.plan import FeaturePlan, FQFieldKey
from metaxy.models.types import (
    CascadeMode,
    CoercibleToFeatureKey,
    FeatureKey,
    ValidatedFeatureKeyAdapter,
    ValidatedFeatureKeySequenceAdapter,
)

FEATURE_VERSION_COL = METAXY_FEATURE_VERSION
FEATURE_TRACKING_VERSION_COL = METAXY_DEFINITION_VERSION

if TYPE_CHECKING:
    import narwhals as nw


# Context variable for active graph (module-level)
_active_graph: ContextVar["FeatureGraph | None"] = ContextVar("_active_graph", default=None)


class SerializedFeature(TypedDict):
    feature_spec: dict[str, Any]
    feature_schema: dict[str, Any]
    metaxy_feature_version: str
    metaxy_definition_version: str
    feature_class_path: str
    project: str


@public
class FeatureGraph:
    def __init__(self):
        # Primary storage: FeatureDefinition objects
        self.feature_definitions_by_key: dict[FeatureKey, FeatureDefinition] = {}

    def add_feature(self, feature: type["BaseFeature"]) -> None:
        """Add a feature class to the graph.

        Creates a FeatureDefinition from the class and delegates to add_feature_definition.

        Args:
            feature: Feature class to register

        Raises:
            ValueError: If a feature with a different import path but the same key is already registered
                       or if duplicate column names would result from renaming operations
        """
        definition = FeatureDefinition.from_feature_class(feature)
        self.add_feature_definition(definition)

    def add_feature_definition(
        self, definition: FeatureDefinition, on_conflict: Literal["raise", "ignore"] = "raise"
    ) -> None:
        """Add a feature to the graph.

        !!! note "Interactions with External Features"

            Normal features take priority over external features with the same key.

        Args:
            definition: FeatureDefinition to register
            on_conflict: What to do if a feature with the same key is already registered

        Raises:
            ValueError: If a non-external feature with a different import path but
                the same key is already registered and `on_conflict` is `"raise"`
        """
        key = definition.key

        if key not in self.feature_definitions_by_key:
            self.feature_definitions_by_key[key] = definition
        elif definition.is_external and not self.feature_definitions_by_key[key].is_external:
            # External features never overwrite non-external features
            return
        elif not definition.is_external and self.feature_definitions_by_key[key].is_external:
            # Non-external features always replace external features
            # Note: version mismatch checking is done in load_feature_definitions,
            # not here, because we need the full graph context to compute
            # provenance-carrying versions.
            self.feature_definitions_by_key[key] = definition
        elif definition.feature_class_path == self.feature_definitions_by_key[key].feature_class_path:
            # Same class path - allow quiet replacement
            self.feature_definitions_by_key[key] = definition
        elif on_conflict == "ignore":
            # Conflict exists but we're ignoring - keep existing definition
            return
        elif definition.is_external:
            # Both external with different class paths - raise to be safe
            raise ValueError(f"External feature with key {key.to_string()} is already registered.")
        else:
            # Both non-external with different class paths
            raise ValueError(
                f"Feature with key {key.to_string()} already registered. "
                f"Existing: {self.feature_definitions_by_key[key].feature_class_path}, "
                f"New: {definition.feature_class_path}. "
                f"Each feature key must be unique within a graph."
            )

    def get_feature_definition(self, key: CoercibleToFeatureKey) -> FeatureDefinition:
        """Get a FeatureDefinition by its key.

        This is the primary method for accessing feature information.

        Args:
            key: Feature key to look up

        Returns:
            FeatureDefinition for the feature

        Raises:
            KeyError: If no feature with the given key is registered
        """
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)

        if validated_key not in self.feature_definitions_by_key:
            raise KeyError(
                f"No feature with key {validated_key.to_string()} found in graph. "
                f"Available keys: {[k.to_string() for k in self.feature_definitions_by_key.keys()]}"
            )
        return self.feature_definitions_by_key[validated_key]

    def remove_feature(self, key: CoercibleToFeatureKey) -> None:
        """Remove a feature from the graph.

        Args:
            key: Feature key to remove. Accepts types that can be converted into a feature key..

        Raises:
            KeyError: If no feature with the given key is registered
        """
        # Validate and coerce the key
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)

        if validated_key not in self.feature_definitions_by_key:
            raise KeyError(
                f"No feature with key {validated_key.to_string()} found in graph. "
                f"Available keys: {[k.to_string() for k in self.feature_definitions_by_key]}"
            )

        del self.feature_definitions_by_key[validated_key]

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
            # Get features for specific project
            features = graph.list_features(projects="myproject")

            # Get all features regardless of project
            all_features = graph.list_features(only_current_project=False)
            ```
        """
        if not only_current_project:
            # Return all features (both class-based and definition-only)
            return list(self.feature_definitions_by_key.keys())

        # Normalize projects to list
        project_list: list[str]
        if projects is None:
            # Try to get from config context
            try:
                from metaxy.config import MetaxyConfig

                config = MetaxyConfig.get()
                if config.project is None:
                    # No project configured - return all features
                    return list(self.feature_definitions_by_key.keys())
                project_list = [config.project]
            except RuntimeError:
                # Config not initialized - in tests or non-CLI usage
                # Return all features (can't determine project)
                return list(self.feature_definitions_by_key.keys())
        elif isinstance(projects, str):
            project_list = [projects]
        else:
            project_list = projects

        # Filter by project(s) using FeatureDefinition.project
        return [key for key, defn in self.feature_definitions_by_key.items() if defn.project in project_list]

    def get_feature_plan(self, key: CoercibleToFeatureKey) -> FeaturePlan:
        """Get a feature plan for a given feature key.

        Args:
            key: Feature key to get plan for. Accepts types that can be converted into a feature key.

        Returns:
            FeaturePlan instance with feature spec and dependencies.

        Raises:
            MetaxyMissingFeatureDependency: If any dependency is not in the graph.
        """
        from metaxy.utils.exceptions import MetaxyMissingFeatureDependency

        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)

        definition = self.feature_definitions_by_key[validated_key]
        spec = definition.spec

        # Check all dependencies are present and collect their specs
        dep_specs = []
        for dep in spec.deps or []:
            if dep.feature not in self.feature_definitions_by_key:
                raise MetaxyMissingFeatureDependency(
                    f"Feature '{validated_key.to_string()}' depends on '{dep.feature.to_string()}' "
                    f"which is not in the graph."
                )
            dep_specs.append(self.feature_definitions_by_key[dep.feature].spec)

        return FeaturePlan(
            feature=spec,
            deps=dep_specs or None,
            feature_deps=spec.deps,
        )

    def get_field_version(self, key: "FQFieldKey") -> str:
        definition = self.feature_definitions_by_key.get(key.feature)

        # check for special case: external feature with provenance override
        # TODO: this has to be done more elegantly
        # this check doesn't really belong here
        if definition and definition.is_external and definition.has_provenance_override:
            provenance = definition.provenance_by_field_override
            field_str = key.field.to_string()
            if provenance and field_str in provenance:
                return provenance[field_str]

        hasher = hashlib.sha256()

        plan = self.get_feature_plan(key.feature)
        field = plan.feature.fields_by_key[key.field]

        hasher.update(key.to_string().encode())
        hasher.update(str(field.code_version).encode())

        for k, v in sorted(plan.get_parent_fields_for_field(key.field).items()):
            hasher.update(self.get_field_version(k).encode())

        return truncate_hash(hasher.hexdigest())

    def get_feature_version_by_field(self, key: CoercibleToFeatureKey) -> dict[str, str]:
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
            res[k.to_string()] = self.get_field_version(FQFieldKey(field=k, feature=validated_key))

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

    def get_downstream_features(self, sources: Sequence[CoercibleToFeatureKey]) -> list[FeatureKey]:
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
            # Build a DAG: a -> b -> d, a -> c -> d
            class FeatureA(mx.BaseFeature, spec=mx.FeatureSpec(key="a", id_columns=["id"])):
                id: str


            class FeatureB(
                mx.BaseFeature, spec=mx.FeatureSpec(key="b", id_columns=["id"], deps=[mx.FeatureDep(feature=FeatureA)])
            ):
                id: str


            class FeatureC(
                mx.BaseFeature, spec=mx.FeatureSpec(key="c", id_columns=["id"], deps=[mx.FeatureDep(feature=FeatureA)])
            ):
                id: str


            class FeatureD(
                mx.BaseFeature,
                spec=mx.FeatureSpec(
                    key="d", id_columns=["id"], deps=[mx.FeatureDep(feature=FeatureB), mx.FeatureDep(feature=FeatureC)]
                ),
            ):
                id: str


            graph.get_downstream_features(["a"])
            # [FeatureKey(['b']), FeatureKey(['c']), FeatureKey(['d'])]
            ```
        """
        # Validate and coerce the source keys
        validated_sources = ValidatedFeatureKeySequenceAdapter.validate_python(sources)

        source_set = set(validated_sources)
        visited: set[FeatureKey] = set()
        post_order: list[FeatureKey] = []

        # Build reverse dependency index for efficient lookup: feature_key -> list of dependents
        # This avoids O(V) scans per node during traversal
        reverse_deps: dict[FeatureKey, list[FeatureKey]] = {}
        for feature_key, definition in self.feature_definitions_by_key.items():
            if definition.spec.deps:
                for dep in definition.spec.deps:
                    dep_key = dep.feature
                    if dep_key not in reverse_deps:
                        reverse_deps[dep_key] = []
                    reverse_deps[dep_key].append(feature_key)

        def visit(key: FeatureKey) -> None:
            """DFS traversal."""
            if key in visited:
                return
            visited.add(key)

            # Use the pre-built reverse dependency index for O(1) lookup
            if key in reverse_deps:
                for dependent in reverse_deps[key]:
                    visit(dependent)

            post_order.append(key)

        # Visit all sources
        for source in validated_sources:
            visit(source)

        # Remove sources from result, reverse to get topological order
        result = [k for k in reversed(post_order) if k not in source_set]
        return result

    def get_upstream_features(self, sources: Sequence[CoercibleToFeatureKey]) -> list[FeatureKey]:
        """Get all features upstream of sources (their dependencies), topologically sorted.

        Performs a depth-first traversal of the dependency graph to find all
        features that the source features transitively depend on.

        Args:
            sources: List of source feature keys. Each element can be string, sequence, FeatureKey, or BaseFeature class.

        Returns:
            List of upstream feature keys in topological order (dependencies first).
            Does not include the source features themselves unless they are dependencies of other sources.

        Example:
            ```py
            # DAG: a -> b -> d
            #      a -> c -> d
            graph.get_upstream_features([FeatureKey(["d"])])
            # [FeatureKey(["a"]), FeatureKey(["b"]), FeatureKey(["c"])]

            # Or use string notation
            graph.get_upstream_features(["d"])
            ```
        """
        validated_sources = ValidatedFeatureKeySequenceAdapter.validate_python(sources)

        visited: set[FeatureKey] = set()
        post_order: list[FeatureKey] = []

        def visit(key: FeatureKey) -> None:
            """DFS traversal following dependencies (upstream)."""
            if key in visited:
                return
            visited.add(key)

            definition = self.feature_definitions_by_key.get(key)
            if definition and definition.spec.deps:
                for dep in definition.spec.deps:
                    visit(dep.feature)

            post_order.append(key)

        # Start DFS from the dependencies of each source, not from the sources themselves.
        # This naturally excludes sources unless they're reached as dependencies of another source.
        for source in validated_sources:
            definition = self.feature_definitions_by_key.get(source)
            if definition and definition.spec.deps:
                for dep in definition.spec.deps:
                    visit(dep.feature)

        return post_order

    def get_cascade_features(
        self,
        source: CoercibleToFeatureKey,
        cascade: CascadeMode | str,
    ) -> list[FeatureKey]:
        """Get features to process for cascade operations.

        Determines the ordered list of features based on cascade direction.

        Args:
            source: Source feature key. Can be string, sequence, FeatureKey, or BaseFeature class.
            cascade: Cascade mode determining which features to include. Can be:
                - CascadeMode.NONE or "none": Only the source feature
                - CascadeMode.DOWNSTREAM or "downstream": Get dependents, order leaf-first (for deletion)
                - CascadeMode.UPSTREAM or "upstream": Get dependencies, order root-first
                - CascadeMode.BOTH or "both": Get both upstream and downstream

        Returns:
            List of feature keys in topological order appropriate for the cascade direction.
            Includes the source feature.

        Raises:
            ValueError: If cascade is not a valid option.

        Example:
            ```py
            from metaxy.models.types import CascadeMode

            # DAG: a -> b -> c
            graph.get_cascade_features("b", CascadeMode.DOWNSTREAM)
            # [FeatureKey(["c"]), FeatureKey(["b"])]  # dependents first

            graph.get_cascade_features("b", "upstream")
            # [FeatureKey(["a"]), FeatureKey(["b"])]  # dependencies first

            graph.get_cascade_features("b", CascadeMode.NONE)
            # [FeatureKey(["b"])]  # only source
            ```
        """
        # Convert string to CascadeMode if needed
        if isinstance(cascade, str):
            try:
                cascade_mode = CascadeMode(cascade.lower())
            except ValueError:
                valid_options = [mode.value for mode in CascadeMode]
                raise ValueError(
                    f"Invalid cascade option: '{cascade}'. Valid options: {', '.join(valid_options)}"
                ) from None
        else:
            if not isinstance(cascade, CascadeMode):
                valid_options = [mode.value for mode in CascadeMode]
                raise ValueError(
                    f"Invalid cascade option type: {type(cascade)!r}. "
                    f"Expected CascadeMode or str with one of: {', '.join(valid_options)}"
                )
            cascade_mode = cascade

        # Validate and coerce the source key
        validated_source = ValidatedFeatureKeyAdapter.validate_python(source)

        if cascade_mode == CascadeMode.NONE:
            return [validated_source]

        if cascade_mode == CascadeMode.DOWNSTREAM:
            downstream = self.get_downstream_features([validated_source])
            features = downstream + [validated_source]
            return self.topological_sort_features(features, descending=True)

        if cascade_mode == CascadeMode.UPSTREAM:
            upstream = self.get_upstream_features([validated_source])
            features = upstream + [validated_source]
            return self.topological_sort_features(features, descending=False)

        # cascade_mode == CascadeMode.BOTH
        upstream = self.get_upstream_features([validated_source])
        downstream = self.get_downstream_features([validated_source])
        features = upstream + [validated_source] + downstream
        return self.topological_sort_features(features, descending=False)

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
            class VideoRaw(mx.BaseFeature, spec=mx.FeatureSpec(key="video/raw", id_columns=["id"])):
                id: str


            class VideoScene(
                mx.BaseFeature,
                spec=mx.FeatureSpec(key="video/scene", id_columns=["id"], deps=[mx.FeatureDep(feature=VideoRaw)]),
            ):
                id: str


            graph.topological_sort_features(["video/raw", "video/scene"])
            # [FeatureKey(['video', 'raw']), FeatureKey(['video', 'scene'])]
            ```
        """
        # Determine which features to sort
        if feature_keys is None:
            # Include all features
            keys_to_sort = set(self.feature_definitions_by_key.keys())
        else:
            # Validate and coerce the feature keys
            validated_keys = ValidatedFeatureKeySequenceAdapter.validate_python(feature_keys)
            keys_to_sort = set(validated_keys)

        visited = set()
        result = []  # Topological order (dependencies first)

        def visit(key: FeatureKey):
            """DFS visit with post-order traversal."""
            if key in visited or key not in keys_to_sort:
                return
            visited.add(key)

            # Get dependencies from feature definition
            definition = self.feature_definitions_by_key.get(key)
            if definition and definition.spec.deps:
                # Sort dependencies alphabetically for deterministic ordering
                sorted_deps = sorted(
                    (dep.feature for dep in definition.spec.deps),
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

    def get_cascade_deletion_order(
        self,
        feature_key: FeatureKey,
        cascade: CascadeMode,
    ) -> list[FeatureKey]:
        """Get features to delete in cascade order.

        Determines which features to delete based on cascade mode and returns them
        in the correct deletion order (dependents before dependencies for safe deletion).

        Args:
            feature_key: The primary feature to delete
            cascade: Cascade mode determining which related features to include

        Returns:
            List of features in deletion order (dependents first, dependencies last)

        Example:
            ```py
            graph = FeatureGraph.get_active()
            # Get deletion order for downstream cascade
            features = graph.get_cascade_deletion_order(FeatureKey(["video", "raw"]), CascadeMode.DOWNSTREAM)
            ```
        """
        features_to_delete: list[FeatureKey] = []

        if cascade == CascadeMode.DOWNSTREAM:
            # Get all downstream features (dependents)
            downstream = self.get_downstream_features([feature_key])
            # Include self
            features_to_delete = downstream + [feature_key]
            # Sort in reverse topological order for safe deletion (dependents before dependencies)
            features_to_delete = self.topological_sort_features(features_to_delete, descending=True)

        elif cascade == CascadeMode.UPSTREAM:
            # Get all upstream features (dependencies) recursively
            upstream = self.get_upstream_features([feature_key])
            # Include self
            features_to_delete = upstream + [feature_key]
            # Sort in reverse topological order for safe deletion (dependents before dependencies)
            features_to_delete = self.topological_sort_features(features_to_delete, descending=True)

        elif cascade == CascadeMode.BOTH:
            # Get both upstream and downstream
            upstream_keys = self.get_upstream_features([feature_key])
            downstream_keys = self.get_downstream_features([feature_key])

            # Combine: dependents -> self -> deps
            features_to_delete = downstream_keys + [feature_key] + upstream_keys
            # Sort in reverse topological order for safe deletion (dependents before dependencies)
            features_to_delete = self.topological_sort_features(features_to_delete, descending=True)

        else:  # CascadeMode.NONE
            features_to_delete = [feature_key]

        return features_to_delete

    def _compute_project_version(
        self,
        features: list[tuple[FeatureKey, FeatureDefinition]],
    ) -> str:
        """Compute a project version hash for a list of features.

        Args:
            features: List of (key, definition) tuples, should be sorted by key.
        """
        if not features:
            return "empty"

        hasher = hashlib.sha256()
        for feature_key, definition in features:
            hasher.update(feature_key.to_string().encode("utf-8"))
            hasher.update(definition.feature_definition_version.encode("utf-8"))

        return truncate_hash(hasher.hexdigest())

    def get_project_version(self, project: str) -> str:
        """Generate a project version for features belonging to a specific project.

        Uses feature_definition_version (spec + schema only), excluding external features.
        This makes the project version independent of external feature changes.

        Args:
            project: The project name to compute version for.

        Returns:
            A hash representing the project's feature definitions.
        """
        project_features = sorted(
            (
                (key, defn)
                for key, defn in self.feature_definitions_by_key.items()
                if defn.project == project and not defn.is_external
            ),
            key=lambda x: x[0],
        )
        return self._compute_project_version(project_features)

    @property
    def project_version(self) -> str:
        """Generate a project version for the current project's features.

        Uses feature_definition_version (spec + schema only), excluding external features.
        The project is determined from MetaxyConfig.project if set, otherwise from the graph's
        single project (via the `project` property).

        Raises:
            RuntimeError: If MetaxyConfig.project is not set and the graph is empty or spans multiple projects.
        """
        from metaxy.config import MetaxyConfig

        return self.get_project_version(MetaxyConfig.get().project or self.project)

    @property
    def has_external_features(self) -> bool:
        """Check if any feature in the graph is an external feature."""
        return any(d.is_external for d in self.feature_definitions_by_key.values())

    @property
    def project(self) -> str:
        """The single project for all non-external features in this graph.

        Returns the project name if all non-external features belong to a single project.

        Raises:
            RuntimeError: If the graph is empty or features span multiple projects.
        """
        projects = {defn.project for defn in self.feature_definitions_by_key.values() if not defn.is_external}
        if len(projects) == 0:
            raise RuntimeError("FeatureGraph contains no non-external features.")
        if len(projects) == 1:
            return projects.pop()
        raise RuntimeError(
            f"FeatureGraph contains features from multiple projects ({', '.join(sorted(projects))}). "
            "Use get_project_version(project) to get the version for a specific project."
        )

    def to_snapshot(self, *, project: str | None = None) -> dict[str, SerializedFeature]:
        """Serialize graph to snapshot format.

        Returns a dict mapping feature_key (string) to feature data dict,
        including the import path of the Feature class for reconstruction.

        External features are excluded from the snapshot as they should not be
        pushed to the metadata store.

        Args:
            project: Only include features from this project. If not provided,
                uses the graph's single project (via the `project` property).

        Returns:
            Dictionary mapping feature_key (string) to feature data dict.

        Raises:
            RuntimeError: If no project is provided and features span multiple projects.
        """
        if project is None:
            project = self.project

        snapshot: dict[str, SerializedFeature] = {}

        for feature_key, definition in self.feature_definitions_by_key.items():
            # Skip external features - they should not be pushed to the metadata store
            if definition.is_external:
                continue

            # Skip features from other projects
            if definition.project != project:
                continue

            feature_key_str = feature_key.to_string()
            feature_spec_dict = definition.spec.model_dump(mode="json")
            feature_schema_dict = definition.feature_schema
            feature_version = self.get_feature_version(feature_key)
            definition_version = definition.feature_definition_version
            project = definition.project
            class_path = definition.feature_class_path
            assert class_path is not None, "feature_class_path must be set for serialization"

            snapshot[feature_key_str] = {
                "feature_spec": feature_spec_dict,
                "feature_schema": feature_schema_dict,
                FEATURE_VERSION_COL: feature_version,
                FEATURE_TRACKING_VERSION_COL: definition_version,
                "feature_class_path": class_path,
                "project": project,
            }

        return snapshot

    @classmethod
    def from_snapshot(
        cls,
        snapshot_data: Mapping[str, Mapping[str, Any]],
    ) -> "FeatureGraph":
        """Reconstruct graph from snapshot by creating FeatureDefinition objects.

        This method creates FeatureDefinition objects directly from the snapshot data
        without any dynamic imports. The resulting graph contains all feature metadata
        needed for operations like migrations and comparisons.

        Args:
            snapshot_data: Dict of feature_key -> dict containing all required fields:
                - feature_spec (dict): The feature specification
                - feature_schema (dict): The JSON schema for the feature
                - feature_class_path (str): The import path of the feature class
                - project (str): The project name

        Returns:
            New FeatureGraph with FeatureDefinition objects

        Raises:
            KeyError: If required fields are missing from snapshot data

        Example:
            ```py
            snapshot_data = {}  # Loaded from metadata store
            # Load snapshot from metadata store
            historical_graph = FeatureGraph.from_snapshot(snapshot_data)
            ```
        """
        graph = cls()

        required_fields = ("feature_spec", "feature_schema", "feature_class_path", "project")

        for feature_key_str, feature_data in snapshot_data.items():
            # Validate all required fields are present
            missing_fields = [f for f in required_fields if f not in feature_data]
            if missing_fields:
                raise KeyError(
                    f"Feature '{feature_key_str}' snapshot is missing required fields: {missing_fields}. "
                    f"All snapshots must include: {required_fields}"
                )

            definition = FeatureDefinition.from_stored_data(
                feature_spec=feature_data["feature_spec"],
                feature_schema=feature_data["feature_schema"],
                feature_class_path=feature_data["feature_class_path"],
                project=feature_data["project"],
                source="snapshot",
            )
            graph.add_feature_definition(definition)

        return graph

    @classmethod
    def get(cls) -> "FeatureGraph":
        """Get the currently active graph.

        Returns the graph from the context variable if set, otherwise returns
        the default global graph.

        Returns:
            Active FeatureGraph instance

        Example:
            ```py
            graph = mx.FeatureGraph.get_active()
            ```
        """
        return _active_graph.get() or graph

    @classmethod
    def get_active(cls) -> "FeatureGraph":
        # backcompat, todo: delete
        return cls.get()

    @classmethod
    def set_active(cls, reg: "FeatureGraph") -> None:
        """Set the active graph for the current context.

        This sets the context variable that will be returned by get_active().
        Typically used in application setup code or test fixtures.

        Args:
            reg: FeatureGraph to activate

        Example:
            ```py
            my_graph = mx.FeatureGraph()
            mx.FeatureGraph.set_active(my_graph)
            mx.FeatureGraph.get_active()  # Returns my_graph
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
            with graph.use():

                class TestFeature(mx.BaseFeature, spec=mx.FeatureSpec(key="test", id_columns=["id"])):
                    id: str
            ```
        """
        token = _active_graph.set(self)
        try:
            yield self
        finally:
            _active_graph.reset(token)


@public
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
            new_cls.graph = active_graph  # ty: ignore[unresolved-attribute]
            new_cls._spec = spec  # ty: ignore[unresolved-attribute]

            # Determine project for this feature
            # Use explicit class attribute if defined, otherwise auto-detect from package
            if "__metaxy_project__" in namespace:
                new_cls.__metaxy_project__ = namespace["__metaxy_project__"]  # ty: ignore[unresolved-attribute]
            else:
                new_cls.__metaxy_project__ = cls._detect_project(new_cls)  # ty: ignore[unresolved-attribute]

            active_graph.add_feature(new_cls)
        else:
            pass  # TODO: set spec to a property that would raise an exception on access

        return new_cls  # ty: ignore[invalid-return-type]

    @staticmethod
    def _detect_project(feature_cls: type) -> str:
        """Detect project for a feature class from its Python package.

        Detection order:
        1. Check for __metaxy_project__ in top-level package
        2. Use top-level package name
        """
        from metaxy._packaging import detect_project_from_package

        module_name = feature_cls.__module__
        return detect_project_from_package(module_name)


class _FeatureSpecDescriptor:
    """Descriptor that returns the feature spec of the feature."""

    def __get__(self, instance, owner) -> str:
        if owner.spec is None:
            raise ValueError(f"Feature '{owner.__name__}' has no spec defined.")
        return owner.spec


@public
class BaseFeature(pydantic.BaseModel, metaclass=MetaxyMeta, spec=None):
    _spec: ClassVar[FeatureSpec]

    graph: ClassVar[FeatureGraph]
    __metaxy_project__: ClassVar[str]

    @classmethod
    def metaxy_project(cls) -> str:
        """Return the project this feature belongs to."""
        return cls.__metaxy_project__

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
    metaxy_project_version: str | None = Field(
        default=None,
        description="Hash of the entire feature graph project version",
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
            class VideoFeature(mx.BaseFeature, spec=mx.FeatureSpec(key="video/processing", id_columns=["id"])):
                id: str


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
            class MyFeature(mx.BaseFeature, spec=mx.FeatureSpec(key="my/feature", id_columns=["id"])):
                id: str


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
        for reproducibility and audit purposes.

        Returns:
            SHA256 hex digest of the complete specification

        Example:
            ```py
            class MyFeature(mx.BaseFeature, spec=mx.FeatureSpec(key="my/feature2", id_columns=["id"])):
                id: str


            MyFeature.feature_spec_version()
            # 'def456...'  # Different from feature_version
            ```
        """
        return cls.spec().feature_spec_version

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
                    upstream_columns[dep_key_str] = dep.select
                    upstream_renames[dep_key_str] = dep.rename

        return joiner.join_upstream(
            upstream_refs=upstream_refs,
            feature_spec=cls.spec(),
            feature_plan=cls.graph.get_feature_plan(cls.spec().key),
            upstream_columns=upstream_columns,
            upstream_renames=upstream_renames,
        )
