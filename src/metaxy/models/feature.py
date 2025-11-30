import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar

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

    from metaxy.models.feature_definition import FeatureDefinition
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


class FeatureGraph:
    def __init__(self):
        import uuid

        # Unique identifier for this graph instance
        self.instance_uid: str = str(uuid.uuid4())
        # Primary storage: FeatureDefinition objects for ALL features
        self.definitions_by_key: dict[FeatureKey, FeatureDefinition] = {}
        # Feature class references (for get_feature_by_key) - will be removed later
        self._feature_classes: dict[FeatureKey, type[BaseFeature]] = {}
        # Pending specs: temporarily stores specs during add_feature before definition is created
        # This is needed because version computation requires the spec to be accessible
        self._pending_specs: dict[FeatureKey, FeatureSpec] = {}

    @property
    def features_by_key(self) -> dict[FeatureKey, type["BaseFeature"]]:
        """Access to feature classes (for backward compatibility).

        Deprecated: Use get_feature_by_key() instead. Will be removed later.
        """
        return self._feature_classes

    @property
    def feature_specs_by_key(self) -> dict[FeatureKey, FeatureSpec]:
        """Get specs derived from definitions (for backward compatibility).

        Also includes pending specs that are being processed during add_feature.
        """
        result = {key: defn.spec for key, defn in self.definitions_by_key.items()}
        # Include pending specs (during add_feature before definition is created)
        result.update(self._pending_specs)
        return result

    @property
    def all_specs_by_key(self) -> dict[FeatureKey, FeatureSpec]:
        """Alias for feature_specs_by_key (backward compatibility)."""
        return self.feature_specs_by_key

    @property
    def standalone_specs_by_key(self) -> dict[FeatureKey, FeatureSpec]:
        """Specs without Feature classes (backward compatibility).

        Returns specs for definitions that don't have a Feature class.
        """
        return {
            key: defn.spec
            for key, defn in self.definitions_by_key.items()
            if key not in self._feature_classes
        }

    def add_feature(
        self,
        feature: "type[BaseFeature] | FeatureDefinition | FeatureSpec",
    ) -> None:
        """Add a feature to the graph.

        Unified method that accepts:
        - Feature class (type[BaseFeature]) - creates FeatureDefinition and stores both
        - FeatureDefinition - stores directly (for external features)
        - FeatureSpec - creates minimal FeatureDefinition (for migrations)

        Local features (Feature classes) always take precedence over external definitions.

        Args:
            feature: Feature class, FeatureDefinition, or FeatureSpec to register
        """
        import warnings

        from metaxy.models.feature_definition import FeatureDefinition

        # Determine the key and what we're adding
        if isinstance(feature, FeatureDefinition):
            key = feature.key
            definition = feature
            feature_class = None
        elif isinstance(feature, FeatureSpec):
            key = feature.key
            # Create minimal FeatureDefinition from spec
            definition = FeatureDefinition(
                spec=feature,
                feature_schema={},  # No schema for standalone specs
                project_name="",  # Unknown project
                feature_version=feature.feature_spec_version,  # Use spec version
                feature_code_version=feature.code_version,
                feature_definition_version=feature.feature_spec_version,
                feature_class_path="",  # No class path
            )
            feature_class = None
        else:
            # It's a Feature class - need to temporarily add spec for version computation
            key = feature.spec().key
            spec = feature.spec()

            # Add spec to pending before creating definition (needed for version computation)
            self._pending_specs[key] = spec
            try:
                definition = feature.to_definition()
            finally:
                # Remove from pending after definition is created
                self._pending_specs.pop(key, None)

            feature_class = feature

        # Check for duplicates
        if key in self.definitions_by_key:
            # If we're adding a Feature class and one already exists
            if feature_class is not None and key in self._feature_classes:
                existing = self._feature_classes[key]
                warnings.warn(
                    f"Feature with key {key.to_string()} already registered. "
                    f"Existing: {existing.__name__}, New: {feature_class.__name__}. "
                    f"Ignoring duplicate registration.",
                    stacklevel=2,
                )
                return
            # If we're adding a Feature class and only definition exists, class takes precedence
            elif feature_class is not None:
                # Update definition and add class
                pass  # Continue to store
            # If we're adding a definition/spec and class already exists, class takes precedence
            elif key in self._feature_classes:
                warnings.warn(
                    f"Feature class already exists for key {key.to_string()}. "
                    f"External definition will be ignored - local features take precedence.",
                    stacklevel=2,
                )
                return
            # If we're adding a definition/spec and one already exists
            else:
                warnings.warn(
                    f"Definition for key {key.to_string()} already exists. "
                    f"Ignoring duplicate definition.",
                    stacklevel=2,
                )
                return

        # Validate that there are no duplicate column names across dependencies
        if definition.spec.deps:
            self._validate_no_duplicate_columns(definition.spec)

        # Store the definition (primary storage)
        self.definitions_by_key[key] = definition

        # Store the Feature class reference if available (secondary storage)
        if feature_class is not None:
            self._feature_classes[key] = feature_class

    def get_definition(self, key: CoercibleToFeatureKey) -> "FeatureDefinition":
        """Get FeatureDefinition for a feature key.

        Args:
            key: Feature key to look up. Accepts types that can be converted
                into a feature key.

        Returns:
            FeatureDefinition for the requested feature

        Raises:
            KeyError: If no feature definition exists for the given key.
                This can happen when the feature is from an external project
                and hasn't been loaded. Use a MetadataStore to access external
                feature definitions.
        """
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)
        if validated_key not in self.definitions_by_key:
            raise KeyError(
                f"No feature definition for '{validated_key.to_string()}' in this graph. "
                f"If this is an external feature, use MetadataStore.graph to access "
                f"feature definitions that include external dependencies loaded from storage."
            )
        return self.definitions_by_key[validated_key]

    def has_feature_class(self, key: CoercibleToFeatureKey) -> bool:
        """Check if a Feature class is available (vs external definition only).

        This is useful for distinguishing between features that have a local
        Python class available versus features loaded from external projects
        that only have a FeatureDefinition.

        Args:
            key: Feature key to check. Accepts types that can be converted
                into a feature key.

        Returns:
            True if a Feature class exists for this key, False if only a
            FeatureDefinition is available.
        """
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)
        return validated_key in self._feature_classes

    def get_external_dependency_projects(self, current_project: str) -> set[str]:
        """Get the set of external project names that features in this graph depend on.

        This identifies which external projects need to have their feature definitions
        loaded from the metadata store before version computation can proceed.

        Args:
            current_project: The current project name (features from this project are not external)

        Returns:
            Set of project names for external dependencies
        """
        from metaxy.models.feature_spec import FeatureDep

        external_projects: set[str] = set()

        for key, spec in self.feature_specs_by_key.items():
            if not spec.deps:
                continue

            for dep in spec.deps:
                if not isinstance(dep, FeatureDep):
                    continue

                dep_key = dep.feature

                # Check if the dependency is in the graph
                if dep_key in self.definitions_by_key:
                    # Get the project from the definition
                    dep_definition = self.definitions_by_key[dep_key]
                    if dep_definition.project_name != current_project:
                        external_projects.add(dep_definition.project_name)
                elif dep_key not in self.feature_specs_by_key:
                    # Dependency not in graph at all - this is an unresolved external dependency
                    # We don't know the project yet, but we'll need to load it
                    # For now, mark it as needing external resolution
                    # The project will be determined when we query the system table
                    pass

        return external_projects

    def get_unresolved_dependencies(self) -> set[FeatureKey]:
        """Get feature keys for dependencies that are not in the graph.

        These are dependencies that need to be loaded from external sources.

        Returns:
            Set of FeatureKey objects for unresolved dependencies
        """
        from metaxy.models.feature_spec import FeatureDep

        unresolved: set[FeatureKey] = set()

        for key, spec in self.feature_specs_by_key.items():
            if not spec.deps:
                continue

            for dep in spec.deps:
                if not isinstance(dep, FeatureDep):
                    continue

                dep_key = dep.feature

                # Check if the dependency is in the graph
                if dep_key not in self.definitions_by_key:
                    unresolved.add(dep_key)

        return unresolved

    def _validate_no_duplicate_columns(self, spec: "FeatureSpec") -> None:
        """Validate that there are no duplicate column names across dependencies after renaming.

        This method checks that after all column selection and renaming operations,
        no two columns have the same name (except for ID columns which are expected to be the same).
        Also validates that columns are not renamed to system column names.

        Args:
            spec: Feature specification to validate

        Raises:
            ValueError: If duplicate column names would result from the dependency configuration
                       or if columns are renamed to system column names
        """
        from metaxy.models.constants import ALL_SYSTEM_COLUMNS
        from metaxy.models.feature_spec import FeatureDep

        if not spec.deps:
            return

        # First, validate each dependency individually
        for dep in spec.deps:
            if not isinstance(dep, FeatureDep):
                continue

            if dep.rename:
                # Get the upstream feature's spec to check its ID columns
                upstream_spec = self.feature_specs_by_key.get(dep.feature)
                upstream_id_columns = upstream_spec.id_columns if upstream_spec else []

                # Check for renaming to system columns or upstream's ID columns
                for old_name, new_name in dep.rename.items():
                    if new_name in ALL_SYSTEM_COLUMNS:
                        raise ValueError(
                            f"Cannot rename column '{old_name}' to system column name '{new_name}' "
                            f"in dependency '{dep.feature.to_string()}'. "
                            f"System columns: {sorted(ALL_SYSTEM_COLUMNS)}"
                        )

                    # Check against upstream feature's ID columns
                    if new_name in upstream_id_columns:
                        raise ValueError(
                            f"Cannot rename column '{old_name}' to ID column '{new_name}' "
                            f"from upstream feature '{dep.feature.to_string()}'. "
                            f"ID columns for '{dep.feature.to_string()}': {upstream_id_columns}"
                        )

                # Check for duplicate column names within this dependency
                renamed_values = list(dep.rename.values())
                if len(renamed_values) != len(set(renamed_values)):
                    # Find the duplicate(s)
                    seen = set()
                    duplicates = set()
                    for name in renamed_values:
                        if name in seen:
                            duplicates.add(name)
                        seen.add(name)
                    raise ValueError(
                        f"Duplicate column names after renaming in dependency '{dep.feature.to_string()}': "
                        f"{sorted(duplicates)}. Cannot rename multiple columns to the same name within a single dependency."
                    )

        # Track all column names and their sources
        column_sources: dict[str, list[str]] = {}  # column_name -> [source_features]
        id_columns_set = set(spec.id_columns)

        for dep in spec.deps:
            if not isinstance(dep, FeatureDep):
                continue

            dep_key_str = dep.feature.to_string()

            # Get the upstream feature spec if available
            upstream_spec = self.feature_specs_by_key.get(dep.feature)
            if not upstream_spec:
                # If upstream feature isn't registered yet, skip validation
                # This can happen during circular imports or when features are defined in different modules
                continue

            # Determine which columns will be present from this dependency
            if dep.columns is None:
                # All columns from upstream (except droppable system columns)
                # We don't know exactly which columns without the actual data,
                # but we can check the renamed columns at least
                if dep.rename:
                    for old_name, new_name in dep.rename.items():
                        if (
                            new_name not in id_columns_set
                        ):  # ID columns are expected to be the same
                            if new_name not in column_sources:
                                column_sources[new_name] = []
                            column_sources[new_name].append(
                                f"{dep_key_str} (renamed from '{old_name}')"
                            )
                # For non-renamed columns, we can't validate without knowing the actual columns
                # This validation will happen at runtime in the joiner
            elif dep.columns == ():
                # Only system columns - no user columns to track
                pass
            else:
                # Specific columns selected
                for col in dep.columns:
                    # Check if this column is renamed
                    if dep.rename and col in dep.rename:
                        new_name = dep.rename[col]
                        if new_name not in id_columns_set:
                            if new_name not in column_sources:
                                column_sources[new_name] = []
                            column_sources[new_name].append(
                                f"{dep_key_str} (renamed from '{col}')"
                            )
                    else:
                        # Column keeps its original name
                        if col not in id_columns_set:
                            if col not in column_sources:
                                column_sources[col] = []
                            column_sources[col].append(dep_key_str)

        # Check for duplicates
        duplicates = {
            col: sources for col, sources in column_sources.items() if len(sources) > 1
        }

        if duplicates:
            # Format error message
            error_lines = []
            for col, sources in sorted(duplicates.items()):
                error_lines.append(
                    f"  - Column '{col}' appears in: {', '.join(sources)}"
                )

            raise ValueError(
                f"Feature '{spec.key.to_string()}' would have duplicate column names after renaming:\n"
                + "\n".join(error_lines)
                + "\n\nUse the 'rename' parameter in FeatureDep to resolve conflicts, "
                "or use 'columns' to select only the columns you need."
            )

    def remove_feature(self, key: CoercibleToFeatureKey) -> None:
        """Remove a feature from the graph.

        Removes the FeatureDefinition and Feature class reference (if any).

        Args:
            key: Feature key to remove. Accepts types that can be converted into a feature key.

        Raises:
            KeyError: If no feature with the given key is registered
        """
        # Validate and coerce the key
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)

        if validated_key not in self.definitions_by_key:
            raise KeyError(
                f"No feature with key {validated_key.to_string()} found in graph. "
                f"Available keys: {[k.to_string() for k in self.definitions_by_key]}"
            )

        # Remove from primary storage
        del self.definitions_by_key[validated_key]

        # Remove from secondary storage (Feature class reference) if exists
        if validated_key in self._feature_classes:
            del self._feature_classes[validated_key]

    def get_feature_by_key(self, key: CoercibleToFeatureKey) -> type["BaseFeature"]:
        """Get a feature class by its key.

        Args:
            key: Feature key to look up. Accepts types that can be converted into a feature key.

        Returns:
            Feature class

        Raises:
            KeyError: If no feature class with the given key is registered
                (note: this only returns Feature classes, not external definitions)

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

        if validated_key not in self._feature_classes:
            raise KeyError(
                f"No feature class with key {validated_key.to_string()} found in graph. "
                f"Available feature classes: {[k.to_string() for k in self._feature_classes.keys()]}"
            )
        return self._feature_classes[validated_key]

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
            # Return all features (including external definitions)
            return list(self.definitions_by_key.keys())

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
                return list(self.definitions_by_key.keys())
        elif isinstance(projects, str):
            project_list = [projects]
        else:
            project_list = projects

        # Filter by project(s) using FeatureDefinition.project_name
        return [
            key
            for key, defn in self.definitions_by_key.items()
            if defn.project_name in project_list
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

        If a FeatureDefinition exists (which has pre-computed versions), returns
        the stored version directly. Otherwise, computes it from field versions.

        Args:
            key: Feature key to get version for. Accepts types that can be converted into a feature key..

        Returns:
            Truncated SHA256 hash representing the feature version.
        """
        # Validate and coerce the key
        validated_key = ValidatedFeatureKeyAdapter.validate_python(key)

        # If we have a FeatureDefinition, use its pre-computed version
        if validated_key in self.definitions_by_key:
            return self.definitions_by_key[validated_key].feature_version

        # Otherwise, compute from field versions (for standalone specs)
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

    def to_snapshot(self) -> dict[str, "FeatureDefinition"]:
        """Serialize graph to snapshot format using FeatureDefinitions.

        Returns a dict mapping feature_key (string) to FeatureDefinition objects.
        Iterates over definitions_by_key to include all features (both those
        with Feature classes and external definitions).

        Returns: dictionary mapping feature_key (string) to FeatureDefinition

        Example:
            ```py
            snapshot = graph.to_snapshot()
            snapshot["video_processing"].feature_version
            # 'abc12345'
            snapshot["video_processing"].spec.feature_spec_version
            # 'def67890'
            snapshot["video_processing"].feature_definition_version
            # 'xyz98765'
            snapshot["video_processing"].feature_class_path
            # 'myapp.features.video.VideoProcessing'
            snapshot["video_processing"].project_name
            # 'myapp'
            ```
        """
        snapshot: dict[str, FeatureDefinition] = {}

        for key, definition in self.definitions_by_key.items():
            feature_key_str = key.to_string()
            snapshot[feature_key_str] = definition

        return snapshot

    @classmethod
    def from_snapshot(
        cls,
        snapshot_data: Mapping[str, "FeatureDefinition | Mapping[str, Any]"],
        *,
        class_path_overrides: dict[str, str] | None = None,
        force_reload: bool = False,
    ) -> "FeatureGraph":
        """Reconstruct graph from snapshot.

        Creates FeatureDefinitions for all features first, then attempts to import
        Feature classes for features that have class paths. This allows the graph
        to work even when Feature classes are not importable.

        If a feature has been moved/renamed, use class_path_overrides to specify
        the new location.

        Args:
            snapshot_data: Dict of feature_key -> FeatureDefinition or dict containing
                feature_spec (dict), feature_class_path (str), and other fields.
                Accepts both FeatureDefinition objects (from to_snapshot()) and
                raw dicts (from DB storage).
            class_path_overrides: Optional dict mapping feature_key to new class path
                                 for features that have been moved/renamed
            force_reload: If True, reload modules from disk to get current code state.

        Returns:
            New FeatureGraph with historical features

        Example:
            ```py
            # Load snapshot from to_snapshot()
            snapshot = active_graph.to_snapshot()
            historical_graph = FeatureGraph.from_snapshot(snapshot)

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

        from metaxy.models.feature_definition import FeatureDefinition

        graph = cls()
        class_path_overrides = class_path_overrides or {}

        def _get_class_path(
            feature_key_str: str, feature_data: FeatureDefinition | Mapping[str, Any]
        ) -> str:
            """Extract class path from feature data, considering overrides."""
            if feature_key_str in class_path_overrides:
                return class_path_overrides[feature_key_str]
            if isinstance(feature_data, FeatureDefinition):
                return feature_data.feature_class_path
            return feature_data.get("feature_class_path", "")

        def _get_feature_spec(
            feature_data: FeatureDefinition | Mapping[str, Any],
        ) -> FeatureSpec:
            """Extract FeatureSpec from feature data."""
            if isinstance(feature_data, FeatureDefinition):
                return feature_data.spec
            return FeatureSpec.model_validate(feature_data["feature_spec"])

        def _to_definition(
            feature_key_str: str,
            feature_data: FeatureDefinition | Mapping[str, Any],
            class_path: str,
        ) -> FeatureDefinition:
            """Convert feature data to FeatureDefinition."""
            if isinstance(feature_data, FeatureDefinition):
                # If class path override is provided, create new definition with updated path
                if class_path != feature_data.feature_class_path:
                    return FeatureDefinition(
                        spec=feature_data.spec,
                        feature_schema=feature_data.feature_schema,
                        project_name=feature_data.project_name,
                        feature_version=feature_data.feature_version,
                        feature_code_version=feature_data.feature_code_version,
                        feature_definition_version=feature_data.feature_definition_version,
                        feature_class_path=class_path,
                    )
                return feature_data
            # From dict (legacy format from DB)
            return FeatureDefinition.from_stored_metadata(
                {
                    **dict(feature_data),
                    "feature_class_path": class_path,
                }
            )

        # If force_reload, collect all module paths first to remove ALL features
        # from those modules before reloading (modules can have multiple features)
        modules_to_reload = set()
        if force_reload:
            for feature_key_str, feature_data in snapshot_data.items():
                class_path = _get_class_path(feature_key_str, feature_data)
                if class_path:
                    module_path, _ = class_path.rsplit(".", 1)
                    if module_path in sys.modules:
                        modules_to_reload.add(module_path)

        # Use context manager to temporarily set the new graph as active
        # This ensures imported Feature classes register to the new graph, not the current one
        with graph.use():
            for feature_key_str, feature_data in snapshot_data.items():
                # Get class path (check overrides first, then stored)
                class_path = _get_class_path(feature_key_str, feature_data)

                # First, create FeatureDefinition
                # This ensures we always have a definition even if import fails
                definition = _to_definition(feature_key_str, feature_data, class_path)

                # Try to import the Feature class if path is available
                if class_path:
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
                                    fcp = _get_class_path(fk_str, fd)
                                    if fcp and fcp.rsplit(".", 1)[0] == module_path:
                                        fspec = _get_feature_spec(fd)
                                        if fspec.key in graph.features_by_key:
                                            graph.remove_feature(fspec.key)

                                # Mark module as processed so we don't remove features again
                                modules_to_reload.discard(module_path)

                            module = importlib.reload(sys.modules[module_path])
                        else:
                            module = __import__(module_path, fromlist=[class_name])

                        feature_cls = getattr(module, class_name)

                        # Validate the imported class is a valid Feature class
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

                        # Feature class successfully loaded - continue to next feature
                        continue

                    except (ImportError, AttributeError):
                        # Feature class not importable - fall through to add definition
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Cannot import Feature class '{class_path}' for '{feature_key_str}'. "
                            f"Adding FeatureDefinition instead."
                        )

                # Add the FeatureDefinition (either no class path or import failed)
                # Skip if already added by metaclass during import
                if definition.key not in graph.definitions_by_key:
                    graph.add_feature(definition)

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
    ) -> type[Self]:  # pyright: ignore[reportGeneralTypeIssues]
        # Inject frozen config if not already specified in namespace
        if "model_config" not in namespace:
            from pydantic import ConfigDict

            namespace["model_config"] = ConfigDict(frozen=True)

        new_cls = super().__new__(cls, cls_name, bases, namespace, **kwargs)

        if spec:
            # Get graph from context at class definition time
            active_graph = FeatureGraph.get_active()
            new_cls.graph = active_graph  # type: ignore[attr-defined]
            new_cls._spec = spec  # type: ignore[attr-defined]

            # Determine project for this feature using intelligent detection
            project = cls._detect_project(new_cls)
            new_cls.project = project  # type: ignore[attr-defined]

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
    def spec(cls) -> FeatureSpec:  # type: ignore[override]
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
    def to_definition(cls) -> "FeatureDefinition":
        """Convert this Feature class to a FeatureDefinition.

        Creates a FeatureDefinition containing all metadata needed to work
        with this feature without requiring the Feature class. This is the
        bridge between Feature classes and the internal representation used
        by Metaxy machinery.

        Returns:
            FeatureDefinition containing complete feature metadata.

        Example:
            ```python
            class MyFeature(Feature, spec=FeatureSpec(...)):
                value: str

            definition = MyFeature.to_definition()
            print(definition.key)  # Same as MyFeature.spec().key
            print(definition.project_name)  # From MetaxyConfig
            ```
        """
        from metaxy.models.feature_definition import FeatureDefinition

        return FeatureDefinition.from_feature_class(cls)

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
