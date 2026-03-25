"""Lock file generation and loading utilities."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import narwhals as nw
import tomli
import tomlkit
from pydantic import BaseModel, ConfigDict

from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.models.feature_definition import FeatureDefinition
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.config import MetaxyConfig
    from metaxy.metadata_store import MetadataStore
    from metaxy.metadata_store.system.storage import SystemTableStorage
    from metaxy.models.feature import FeatureGraph
    from metaxy.models.feature_selection import FeatureSelection


class LockedFeatureInfo(BaseModel):
    """Version metadata for a locked feature."""

    model_config = ConfigDict(extra="forbid")

    version: str
    version_by_field: dict[str, str]
    definition_version: str


class LockedFeature(BaseModel):
    """A single locked feature entry."""

    model_config = ConfigDict(extra="forbid")

    info: LockedFeatureInfo
    data: FeatureDefinition


class LockFile(BaseModel):
    """Pydantic model for metaxy.lock file."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, LockedFeature]

    def to_toml(self) -> str:
        """Serialize to TOML string."""
        return tomlkit.dumps(self.model_dump(mode="json", exclude_none=True), sort_keys=True)

    @classmethod
    def from_toml(cls, content: str, *, path: Path | None = None) -> LockFile:
        """Parse from TOML string."""
        from pydantic import ValidationError

        try:
            data = tomli.loads(content)
            return cls.model_validate(data)
        except ValidationError as e:
            location = f" ({path})" if path else ""
            raise ValidationError(f"Invalid lock file{location}. Try regenerating it with `metaxy lock`.") from e


def load_lock_file(config: MetaxyConfig) -> list[FeatureDefinition]:
    """Load external feature definitions from a lock file.

    Looks for `metaxy.lock` next to the config file.

    Args:
        config: Metaxy configuration (used to locate the lock file).

    Returns:
        List of loaded FeatureDefinition objects. Empty list if no lock file exists.
    """
    from metaxy.models.feature import FeatureGraph

    lock_path = config.lock_file
    if lock_path is None:
        return []

    if not lock_path.exists():
        if config.metaxy_lock_path is not None:
            warnings.warn(
                f"Lock file not found at {lock_path}. "
                "External features from the lock file will not be loaded. "
                "Run `metaxy lock` to generate it.",
                stacklevel=2,
            )
        return []

    lock_file = LockFile.from_toml(lock_path.read_text())

    if not lock_file.features:
        return []

    graph = FeatureGraph.get_active()
    definitions: list[FeatureDefinition] = []

    for locked_feature in lock_file.features.values():
        # Mark as external and add to graph
        locked_feature.data._is_external = True
        locked_feature.data._source = "metaxy.lock"
        graph.add_feature_definition(locked_feature.data)
        definitions.append(locked_feature.data)

    return definitions


class FeatureChange(BaseModel):
    """A single feature that was added or removed."""

    model_config = ConfigDict(extra="forbid")

    project: str
    key: str
    version: str


class FeatureUpdate(BaseModel):
    """A single feature whose definition changed."""

    model_config = ConfigDict(extra="forbid")

    project: str
    key: str
    old_version: str
    new_version: str
    metadata_only: bool


class LockFileResult(BaseModel):
    """Result of a lock file generation.

    Attributes:
        added: Features present in the new lock file but not the old one.
        updated: Features whose definition changed between old and new lock files.
        removed: Features present in the old lock file but not the new one.
        unchanged_count: Number of features that are identical between old and new lock files.
    """

    model_config = ConfigDict(extra="forbid")

    added: list[FeatureChange]
    updated: list[FeatureUpdate]
    removed: list[FeatureChange]
    unchanged_count: int = 0

    @property
    def count(self) -> int:
        """Total number of features in the new lock file."""
        return len(self.added) + len(self.updated) + self.unchanged_count

    @property
    def changed(self) -> bool:
        """Whether any features were added, updated, or removed."""
        return bool(self.added or self.updated or self.removed)


def generate_lock_file(
    store: MetadataStore,
    output_path: Path,
    *,
    exclude_project: str | None = None,
    selection: FeatureSelection | None = None,
) -> LockFileResult:
    """Generate a lock file with feature definitions from a metadata store.

    Automatically discovers external dependencies by analyzing the current FeatureGraph.
    Recursively resolves transitive dependencies.

    Args:
        store: Metadata store to fetch features from.
        output_path: Path to write the lock file.
        exclude_project: Optional project name to exclude from the lock file.
            Features belonging to this project will be skipped.
        selection: Optional feature selection to include additional features
            beyond those discovered from the graph's dependencies.

    Returns:
        Result with the number of features and whether the lock file changed.

    Raises:
        FeatureNotFoundError: If any required features are not found in the store.
    """
    from contextlib import nullcontext

    from metaxy.metadata_store.system.storage import SystemTableStorage
    from metaxy.models.feature import FeatureGraph

    graph = FeatureGraph.get_active()

    # Find all dependency keys referenced by non-external features in the graph
    local_keys: set[FeatureKey] = set()
    referenced_dep_keys: set[FeatureKey] = set()

    for key, defn in graph.feature_definitions_by_key.items():
        if not defn.is_external:
            local_keys.add(key)
            for dep in defn.spec.deps:
                referenced_dep_keys.add(dep.feature)

    # External deps are those referenced but not defined locally
    external_keys_needed = referenced_dep_keys - local_keys

    if not external_keys_needed and not selection:
        return _write_lock_file(output_path, LockFile(features={}))

    # Use nullcontext if store is already open, otherwise open it
    cm = nullcontext(store) if store._is_open else store
    with cm:
        storage = SystemTableStorage(store)
        # Single query: load all features except current project
        all_store_definitions, db_versions = _load_all_features_from_store(storage, exclude_project)

    # Resolve selection against loaded features
    if selection:
        _apply_selection(selection, all_store_definitions, local_keys, external_keys_needed)

    if not external_keys_needed:
        return _write_lock_file(output_path, LockFile(features={}))

    # Resolve transitive dependencies in memory
    definitions, missing = _resolve_transitive_deps(external_keys_needed, all_store_definitions, local_keys, graph)

    # Check for missing features (includes transitive dependencies)
    if missing:
        sorted_missing = sorted(missing)
        raise FeatureNotFoundError(
            f"Features not found in metadata store: {', '.join(k.to_string() for k in sorted_missing)}",
            keys=sorted_missing,
        )

    # Build and write lock file (validates versions match)
    lock_file = _build_lock_file(definitions, db_versions, graph)
    return _write_lock_file(output_path, lock_file)


def _load_all_features_from_store(
    storage: SystemTableStorage,
    exclude_project: str | None,
) -> tuple[dict[FeatureKey, FeatureDefinition], dict[FeatureKey, str]]:
    """Load all features from store, optionally excluding a project.

    Single query to minimize store access.

    Returns:
        Tuple of (definitions by key, DB versions by key).
    """
    from metaxy.models.feature_definition import FeatureDefinition

    # Build filter to exclude current project
    filters = []
    if exclude_project:
        filters.append(nw.col("project") != exclude_project)

    features_df = storage._read_latest_features_by_project(filters=filters)

    definitions: dict[FeatureKey, FeatureDefinition] = {}
    db_versions: dict[FeatureKey, str] = {}

    for row in features_df.iter_rows(named=True):
        try:
            defn = FeatureDefinition.from_stored_data(
                feature_spec=row["feature_spec"],
                feature_schema=row["feature_schema"],
                feature_class_path=row["feature_class_path"],
                project=row["project"],
            )
        except Exception as e:
            from metaxy._warnings import InvalidStoredFeatureWarning

            feature_key = row.get("feature_key", "<unknown>")
            warnings.warn(
                f"Skipping feature '{feature_key}': failed to load from store: {e}",
                InvalidStoredFeatureWarning,
            )
            continue
        definitions[defn.key] = defn
        db_versions[defn.key] = row["metaxy_feature_version"]

    return definitions, db_versions


def _apply_selection(
    selection: FeatureSelection,
    all_store_definitions: dict[FeatureKey, FeatureDefinition],
    local_keys: set[FeatureKey],
    external_keys_needed: set[FeatureKey],
) -> None:
    """Add feature keys matching the selection to ``external_keys_needed`` (mutates in place)."""
    if selection.all:
        external_keys_needed.update(k for k in all_store_definitions if k not in local_keys)
        return

    if selection.projects:
        project_set = set(selection.projects)
        for key, defn in all_store_definitions.items():
            if defn.project in project_set and key not in local_keys:
                external_keys_needed.add(key)

    if selection.keys:
        external_keys_needed.update(k for k in selection.keys if k not in local_keys)


def _resolve_transitive_deps(
    initial_keys: set[FeatureKey],
    all_definitions: dict[FeatureKey, FeatureDefinition],
    local_keys: set[FeatureKey],
    graph: FeatureGraph,
) -> tuple[list[FeatureDefinition], set[FeatureKey]]:
    """Resolve transitive dependencies in memory.

    Adds loaded definitions to the graph for version computation.

    Returns:
        Tuple of (needed definitions, missing keys).
    """
    needed: dict[FeatureKey, FeatureDefinition] = {}
    keys_to_process = set(initial_keys)
    all_requested_keys: set[FeatureKey] = set(initial_keys)

    while keys_to_process:
        new_dep_keys: set[FeatureKey] = set()

        for key in keys_to_process:
            if key in needed or key in local_keys:
                continue

            if key not in all_definitions:
                # Will be reported as missing
                continue

            defn = all_definitions[key]
            needed[key] = defn
            graph.add_feature_definition(defn, on_conflict="ignore")

            for dep in defn.spec.deps:
                if dep.feature not in needed and dep.feature not in local_keys:
                    new_dep_keys.add(dep.feature)

        all_requested_keys.update(new_dep_keys)
        keys_to_process = new_dep_keys - set(needed.keys())

    missing_keys = all_requested_keys - set(needed.keys()) - local_keys
    return list(needed.values()), missing_keys


def _build_lock_file(
    definitions: list[FeatureDefinition],
    db_versions: dict[FeatureKey, str],
    graph: FeatureGraph,
) -> LockFile:
    """Build LockFile model.

    Validates that computed versions match DB versions.
    """
    from metaxy.utils.exceptions import MetaxyInvariantViolationError

    features: dict[str, LockedFeature] = {}

    for defn in sorted(definitions, key=lambda d: d.key.to_string()):
        feature_key = defn.key.to_string()

        # Compute version from graph
        computed_version = graph.get_feature_version(defn.key)
        db_version = db_versions.get(defn.key)

        # Verify computed version matches DB version
        if db_version is not None and computed_version != db_version:
            raise MetaxyInvariantViolationError(
                f"Version mismatch for feature '{feature_key}': "
                f"computed={computed_version}, db={db_version}. "
                f"This may indicate the feature graph is out of sync with the metadata store."
            )

        features[feature_key] = LockedFeature(
            info=LockedFeatureInfo(
                version=computed_version,
                version_by_field=graph.get_feature_version_by_field(defn.key),
                definition_version=defn.feature_definition_version,
            ),
            data=defn,
        )

    return LockFile(features=features)


def _diff_lock_files(output_path: Path, new_lock: LockFile) -> LockFileResult:
    """Compare new lock file against existing one."""
    from pydantic import ValidationError

    def _all_added() -> LockFileResult:
        return LockFileResult(
            added=[
                FeatureChange(
                    project=new_lock.features[k].data.project, key=k, version=new_lock.features[k].info.version
                )
                for k in sorted(new_lock.features)
            ],
            updated=[],
            removed=[],
        )

    if not output_path.exists():
        return _all_added()

    try:
        old_lock = LockFile.from_toml(output_path.read_text(), path=output_path)
    except (tomli.TOMLDecodeError, ValidationError):
        warnings.warn(
            f"Could not parse existing lock file at {output_path}; treating as fresh.",
            stacklevel=2,
        )
        return _all_added()

    old_keys = set(old_lock.features.keys())
    new_keys = set(new_lock.features.keys())
    common = old_keys & new_keys

    added = [
        FeatureChange(project=new_lock.features[k].data.project, key=k, version=new_lock.features[k].info.version)
        for k in sorted(new_keys - old_keys)
    ]
    removed = [
        FeatureChange(project=old_lock.features[k].data.project, key=k, version=old_lock.features[k].info.version)
        for k in sorted(old_keys - new_keys)
    ]
    updated: list[FeatureUpdate] = []
    for k in sorted(common):
        old_info = old_lock.features[k].info
        new_info = new_lock.features[k].info
        if old_info.definition_version != new_info.definition_version:
            updated.append(
                FeatureUpdate(
                    project=new_lock.features[k].data.project,
                    key=k,
                    old_version=old_info.version,
                    new_version=new_info.version,
                    metadata_only=old_info.version == new_info.version,
                )
            )

    return LockFileResult(added=added, updated=updated, removed=removed, unchanged_count=len(common) - len(updated))


def _write_lock_file(output_path: Path, lock_file: LockFile) -> LockFileResult:
    """Write lock file with header comment. Skips writing when nothing changed."""
    result = _diff_lock_files(output_path, lock_file)

    if result.changed or not output_path.exists():
        output_path.write_text("# Generated by `metaxy lock`.\n\n" + lock_file.to_toml())

    return result
