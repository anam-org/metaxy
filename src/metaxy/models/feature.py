import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, Generic

import pydantic
from pydantic._internal._model_construction import ModelMetaclass
from typing_extensions import Self

from metaxy.models.feature_spec import (
    BaseFeatureSpec,
    BaseFeatureSpecWithIDColumns,
    DefaultFeatureCols,
    IDColumns,
    IDColumnsT,
    TestingUIDCols,
)
from metaxy.models.plan import FeaturePlan, FQFieldKey
from metaxy.models.types import FeatureKey
from metaxy.utils.code_version_descriptor import _CodeVersionDescriptor
from metaxy.utils.hashing import truncate_hash

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


def get_feature_by_key(key: "FeatureKey") -> type["BaseFeature[IDColumns]"]:
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
        self.features_by_key: dict[FeatureKey, type[BaseFeature[IDColumns]]] = {}
        self.feature_specs_by_key: dict[FeatureKey, BaseFeatureSpecWithIDColumns] = {}

    def add_feature(self, feature: type["BaseFeature[IDColumns]"]) -> None:
        """Add a feature to the graph.

        Args:
            feature: Feature class to register

        Raises:
            ValueError: If a feature with the same key is already registered
                       or if duplicate column names would result from renaming operations
        """
        if feature.spec().key in self.features_by_key:
            existing = self.features_by_key[feature.spec().key]
            raise ValueError(
                f"Feature with key {feature.spec().key.to_string()} already registered. "
                f"Existing: {existing.__name__}, New: {feature.__name__}. "
                f"Each feature key must be unique within a graph."
            )

        # Validate that there are no duplicate column names across dependencies after renaming
        if feature.spec().deps:
            self._validate_no_duplicate_columns(feature.spec())

        self.features_by_key[feature.spec().key] = feature
        self.feature_specs_by_key[feature.spec().key] = feature.spec()

    def _validate_no_duplicate_columns(
        self, spec: "BaseFeatureSpecWithIDColumns"
    ) -> None:
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
                upstream_spec = self.feature_specs_by_key.get(dep.key)
                upstream_id_columns = upstream_spec.id_columns if upstream_spec else []

                # Check for renaming to system columns or upstream's ID columns
                for old_name, new_name in dep.rename.items():
                    if new_name in ALL_SYSTEM_COLUMNS:
                        raise ValueError(
                            f"Cannot rename column '{old_name}' to system column name '{new_name}' "
                            f"in dependency '{dep.key.to_string()}'. "
                            f"System columns: {sorted(ALL_SYSTEM_COLUMNS)}"
                        )

                    # Check against upstream feature's ID columns
                    if new_name in upstream_id_columns:
                        raise ValueError(
                            f"Cannot rename column '{old_name}' to ID column '{new_name}' "
                            f"from upstream feature '{dep.key.to_string()}'. "
                            f"ID columns for '{dep.key.to_string()}': {upstream_id_columns}"
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
                        f"Duplicate column names after renaming in dependency '{dep.key.to_string()}': "
                        f"{sorted(duplicates)}. Cannot rename multiple columns to the same name within a single dependency."
                    )

        # Track all column names and their sources
        column_sources: dict[str, list[str]] = {}  # column_name -> [source_features]
        id_columns_set = set(spec.id_columns)

        for dep in spec.deps:
            if not isinstance(dep, FeatureDep):
                continue

            dep_key_str = dep.key.to_string()

            # Get the upstream feature spec if available
            upstream_spec = self.feature_specs_by_key.get(dep.key)
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

    def get_feature_by_key(self, key: FeatureKey) -> type["BaseFeature[IDColumns]"]:
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

        return truncate_hash(hasher.hexdigest())

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

        return truncate_hash(hasher.hexdigest())

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
        return truncate_hash(hasher.hexdigest())

    def to_snapshot(self) -> dict[str, dict[str, Any]]:
        """Serialize graph to snapshot format.

        Returns a dict mapping feature_key (string) to feature data dict,
        including the import path of the Feature class for reconstruction.

        Returns:
            Dict of feature_key -> {
                feature_spec: dict,
                feature_version: str,
                feature_spec_version: str,
                feature_tracking_version: str,
                feature_class_path: str,
                project: str
            }

        Example:
            >>> snapshot = graph.to_snapshot()
            >>> snapshot["video_processing"]["feature_version"]
            'abc12345'
            >>> snapshot["video_processing"]["feature_spec_version"]
            'def67890'
            >>> snapshot["video_processing"]["feature_tracking_version"]
            'xyz98765'
            >>> snapshot["video_processing"]["feature_class_path"]
            'myapp.features.video.VideoProcessing'
            >>> snapshot["video_processing"]["project"]
            'myapp'
        """
        snapshot = {}

        for feature_key, feature_cls in self.features_by_key.items():
            feature_key_str = feature_key.to_string()
            feature_spec_dict = feature_cls.spec().model_dump(mode="json")  # type: ignore[attr-defined]
            feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]
            feature_spec_version = feature_cls.spec().feature_spec_version  # type: ignore[attr-defined]
            feature_tracking_version = feature_cls.feature_tracking_version()  # type: ignore[attr-defined]
            project = feature_cls.project  # type: ignore[attr-defined]

            # Get class import path (module.ClassName)
            class_path = f"{feature_cls.__module__}.{feature_cls.__name__}"

            snapshot[feature_key_str] = {
                "feature_spec": feature_spec_dict,
                "feature_version": feature_version,
                "feature_spec_version": feature_spec_version,
                "feature_tracking_version": feature_tracking_version,
                "feature_class_path": class_path,
                "project": project,
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
        This ensures custom methods (like load_input) are available.

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

        from metaxy.models.feature_spec import BaseFeatureSpec

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
                # Parse BaseFeatureSpec for validation
                feature_spec_dict = feature_data["feature_spec"]
                BaseFeatureSpec.model_validate(feature_spec_dict)

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
                                    fspec = BaseFeatureSpec.model_validate(fspec_dict)
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
                        f"Cannot import Feature class '{class_path}' for feature graph reconstruction from snapshot. "
                        f"Feature '{feature_key_str}' is required to reconstruct the graph, but the class "
                        f"cannot be found at the recorded import path. "
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


class MetaxyMeta(ModelMetaclass):
    def __new__(
        cls,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *,
        spec: BaseFeatureSpecWithIDColumns | None = None,
        **kwargs,
    ) -> type[Self]:  # pyright: ignore[reportGeneralTypeIssues]
        new_cls = super().__new__(cls, cls_name, bases, namespace, **kwargs)

        if spec:
            # Get graph from context at class definition time
            active_graph = FeatureGraph.get_active()
            new_cls.graph = active_graph
            new_cls._spec = spec

            # Determine project for this feature using intelligent detection
            project = cls._detect_project(new_cls)
            new_cls.project = project  # type: ignore[attr-defined]
            active_graph.add_feature(new_cls)
        else:
            pass  # TODO: set spec to a property that would raise an exception on access

        return new_cls

    @staticmethod
    def _detect_project(feature_cls: type) -> str:
        """Detect project for a feature class using multiple strategies.

        Detection order:
        1. If loaded via entrypoint, extract project from package name
        2. Extract package name from module path (e.g., "my_package.features" -> "my_package")
        3. For test/example modules, use global config (before searching filesystem)
        4. Look for pyproject.toml in parent directories to get project name
        5. Fall back to global config

        Args:
            feature_cls: The Feature class being registered

        Returns:
            Project name string
        """
        import sys
        from pathlib import Path

        module_name = feature_cls.__module__

        # Get the top-level package name
        root_package = module_name.split(".")[0]

        # Strategy 1: Check if loaded via entrypoint
        # When loaded via entrypoints, sys.modules will have metadata about the entry point
        # This is the most reliable method for installed packages
        # Check if this was loaded as an entrypoint
        # Entry points are typically registered in the package's metadata
        import importlib.metadata

        # Try to find the distribution that owns this module
        for dist in importlib.metadata.distributions():
            # Use bracket notation for metadata access (works across Python versions)
            try:
                name = dist.metadata["Name"]
            except KeyError:
                continue
            if name.replace("-", "_") == root_package:
                # Found the distribution - use its name as the project
                # Normalize project name (replace hyphens with underscores)
                return name.replace("-", "_")

        # Strategy 2: Extract package name from module path
        # For modules like "my_package.features.video", extract "my_package"
        if "." in module_name:
            # Only use this if it's not a common/generic name
            if root_package not in ("metaxy", "tests", "examples", "__main__"):
                return root_package

        # Strategy 3: For test/example modules, prioritize global config
        # This ensures tests use the configured test project
        # But skip this for modules that are mocked in tests (they have MagicMock in sys.modules)
        if root_package in ("tests", "test", "examples") or (
            root_package.startswith("test_") and module_name != "test_module"
        ):
            # Check if this is a real test module or a mock used in testing
            module = sys.modules.get(module_name)
            if module and not hasattr(module, "_mock_name"):  # Not a MagicMock
                from metaxy.config import MetaxyConfig

                config = MetaxyConfig.get()
                return config.project

        # Strategy 4: Look for pyproject.toml in parent directories
        # Get the module's file path
        module = sys.modules.get(module_name)
        if module and hasattr(module, "__file__") and module.__file__:
            module_path = Path(module.__file__).parent

            # Search up the directory tree for pyproject.toml
            current_dir = module_path
            while current_dir != current_dir.parent:
                pyproject_path = current_dir / "pyproject.toml"
                if pyproject_path.exists():
                    import tomli as tomllib

                    with open(pyproject_path, "rb") as f:
                        pyproject_data = tomllib.load(f)

                    # Extract project name from pyproject.toml
                    if "project" in pyproject_data:
                        project_name = pyproject_data["project"].get("name")
                        if project_name:
                            # Normalize project name
                            return project_name.replace("-", "_")

                current_dir = current_dir.parent

        # Strategy 5: Fall back to global config
        # This is used for any remaining cases
        from metaxy.config import MetaxyConfig

        config = MetaxyConfig.get()
        return config.project


class BaseFeature(
    pydantic.BaseModel,
    Generic[IDColumnsT],
    metaclass=MetaxyMeta,
    spec=None,
):
    _spec: ClassVar[Any]
    graph: ClassVar[FeatureGraph]
    project: ClassVar[str]

    @classmethod
    def spec(cls) -> BaseFeatureSpec[IDColumnsT]:
        return cls._spec  # always set by metaclass

    @classmethod
    def id_columns(cls) -> IDColumnsT:
        return cls.spec().id_columns

    @classmethod
    def table_name(cls) -> str:
        """Get SQL-like table name for this feature.

        Converts feature key to SQL-compatible table name by joining
        parts with double underscores, consistent with IbisMetadataStore.

        Returns:
            Table name string (e.g., "my_namespace__my_feature")

        Example:
            >>> class VideoFeature(Feature, spec=BaseFeatureSpec(
            ...     key=FeatureKey(["video", "processing"]),
            ...     ...
            ... )):
            ...     pass
            >>> VideoFeature.table_name()
            'video__processing'
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

        Used to distinguish current vs historical metadata versions.
        Stored in the 'feature_version' column of metadata DataFrames.

        Returns:
            SHA256 hex digest (like git short hashes)

        Example:
            >>> class MyFeature(Feature, spec=BaseFeatureSpec(
            ...     key=FeatureKey(["my", "feature"]),
            ...     fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ... )):
            ...     pass
            >>> MyFeature.feature_version()
            'a3f8b2c1...'
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

        Stored in the 'feature_spec_version' column of metadata DataFrames.

        Returns:
            SHA256 hex digest of the complete specification

        Example:
            >>> class MyFeature(Feature, spec=BaseFeatureSpec(
            ...     key=FeatureKey(["my", "feature"]),
            ...     fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ... )):
            ...     pass
            >>> MyFeature.feature_spec_version()
            'def456...'  # Different from feature_version
        """
        return cls.spec().feature_spec_version

    @classmethod
    def feature_tracking_version(cls) -> str:
        """Get hash combining feature spec version and project.

        This version is used in system tables to track when features move between projects
        or when their specifications change. It combines:
        - feature_spec_version: Complete feature specification hash
        - project: The project this feature belongs to

        This allows the migration system to detect when a feature moves from one project
        to another, triggering appropriate migrations.

        Returns:
            SHA256 hex digest of feature_spec_version + project

        Example:
            >>> class MyFeature(Feature, spec=FeatureSpec(...)):
            ...     pass
            >>> MyFeature.feature_tracking_version()  # Combines spec + project
            'abc789...'
        """
        hasher = hashlib.sha256()
        hasher.update(cls.feature_spec_version().encode())
        hasher.update(cls.project.encode())
        return truncate_hash(hasher.hexdigest())

    @classmethod
    def data_version(cls) -> dict[str, str]:
        """Get the code-level data version for this feature.

        This returns a static hash based on code versions and dependencies,
        not sample-level data versions.

        Returns:
            Dictionary mapping field keys to their data version hashes.
        """
        return cls.graph.get_feature_version_by_field(cls.spec().key)

    @classmethod
    def load_input(
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
        """
        from metaxy.models.feature_spec import FeatureDep

        # Extract columns and renames from deps
        upstream_columns: dict[str, tuple[str, ...] | None] = {}
        upstream_renames: dict[str, dict[str, str] | None] = {}

        for dep in cls.spec().deps or []:
            if isinstance(dep, FeatureDep):
                dep_key_str = dep.key.to_string()
                upstream_columns[dep_key_str] = dep.columns
                upstream_renames[dep_key_str] = dep.rename

        # Type cast needed due to TypeVar invariance - IDColumnsT is bound to IDColumns
        # so this is safe at runtime
        from typing import cast

        feature_spec = cast("BaseFeatureSpec[IDColumns]", cls.spec())

        return joiner.join_upstream(
            upstream_refs=upstream_refs,
            feature_spec=feature_spec,
            feature_plan=cls.graph.get_feature_plan(cls.spec().key),
            upstream_columns=upstream_columns,
            upstream_renames=upstream_renames,
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
            ...         result = diff_resolver.find_changes(target_versions, current_metadata, cls.id_columns()())
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
            id_columns=cls.id_columns(),  # Pass ID columns from feature spec
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


# TODO: move this to tests, stop using it in docs and examples
class Feature(BaseFeature[DefaultFeatureCols], spec=None):
    sample_uid: str | None = None


############## testing ######################


# TODO: move this to tests, stop using it in docs and examples
class TestingFeature(BaseFeature[TestingUIDCols], spec=None):
    sample_uid: str | None = None
