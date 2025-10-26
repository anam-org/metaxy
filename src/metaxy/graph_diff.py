"""Core graph diff engine for comparing feature graph snapshots."""

import json
from typing import Any

from pydantic import Field

from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY, MetadataStore
from metaxy.models.bases import FrozenBaseModel
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey, FieldKey


class FieldChange(FrozenBaseModel):
    """Represents a change in a field between two snapshots."""

    field_key: FieldKey
    old_version: str | None = None  # None if field was added
    new_version: str | None = None  # None if field was removed

    @property
    def is_added(self) -> bool:
        """Check if field was added."""
        return self.old_version is None

    @property
    def is_removed(self) -> bool:
        """Check if field was removed."""
        return self.new_version is None

    @property
    def is_changed(self) -> bool:
        """Check if field version changed."""
        return (
            self.old_version is not None
            and self.new_version is not None
            and self.old_version != self.new_version
        )


class FeatureChange(FrozenBaseModel):
    """Represents a change in a feature between two snapshots."""

    feature_key: FeatureKey
    old_version: str | None = None  # None if feature was added
    new_version: str | None = None  # None if feature was removed
    field_changes: list[FieldChange] = Field(default_factory=list)

    @property
    def is_added(self) -> bool:
        """Check if feature was added."""
        return self.old_version is None

    @property
    def is_removed(self) -> bool:
        """Check if feature was removed."""
        return self.new_version is None

    @property
    def has_field_changes(self) -> bool:
        """Check if feature has any field changes."""
        return len(self.field_changes) > 0


class GraphDiff(FrozenBaseModel):
    """Result of comparing two graph snapshots."""

    snapshot1: str
    snapshot2: str
    added_features: list[FeatureKey] = Field(default_factory=list)
    removed_features: list[FeatureKey] = Field(default_factory=list)
    changed_features: list[FeatureChange] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if diff contains any changes."""
        return bool(
            self.added_features or self.removed_features or self.changed_features
        )


class SnapshotResolver:
    """Resolves snapshot version literals to actual snapshot hashes."""

    def resolve_snapshot(
        self, literal: str, store: MetadataStore | None, graph: FeatureGraph | None
    ) -> str:
        """Resolve a snapshot literal to its actual version hash.

        Args:
            literal: Snapshot identifier ("latest", "current", or version hash)
            store: Metadata store to query for snapshots (required for "latest")
            graph: Optional active graph for "current" resolution

        Returns:
            Resolved snapshot version hash

        Raises:
            ValueError: If literal is invalid or cannot be resolved
        """
        if literal == "latest":
            if store is None:
                raise ValueError(
                    "Cannot resolve 'latest': no metadata store provided. "
                    "Provide a store to query for snapshots."
                )
            return self._resolve_latest(store)
        elif literal == "current":
            return self._resolve_current(graph)
        else:
            # Treat as explicit snapshot version
            return literal

    def _resolve_latest(self, store: MetadataStore) -> str:
        """Resolve 'latest' to most recent snapshot in store."""
        snapshots_df = store.read_graph_snapshots()

        if snapshots_df.height == 0:
            raise ValueError(
                "No snapshots found in store. Cannot resolve 'latest'. "
                "Run 'metaxy graph push' to record a snapshot."
            )

        # read_graph_snapshots() returns sorted by recorded_at descending
        latest_snapshot = snapshots_df["snapshot_version"][0]
        return latest_snapshot

    def _resolve_current(self, graph: FeatureGraph | None) -> str:
        """Resolve 'current' to active graph's snapshot version."""
        if graph is None:
            raise ValueError(
                "Cannot resolve 'current': no active graph provided. "
                "Ensure features are loaded before using 'current'."
            )

        if len(graph.features_by_key) == 0:
            raise ValueError(
                "Cannot resolve 'current': active graph is empty. "
                "Ensure features are loaded before using 'current'."
            )

        return graph.snapshot_version


class GraphDiffer:
    """Compares two graph snapshots and produces a diff."""

    def diff(
        self,
        snapshot1_data: dict[str, dict[str, Any]],
        snapshot2_data: dict[str, dict[str, Any]],
    ) -> GraphDiff:
        """Compute diff between two snapshots.

        Args:
            snapshot1_data: First snapshot (feature_key -> {feature_version, fields})
            snapshot2_data: Second snapshot (feature_key -> {feature_version, fields})

        Returns:
            GraphDiff with added, removed, and changed features
        """
        # Extract feature keys
        keys1 = set(snapshot1_data.keys())
        keys2 = set(snapshot2_data.keys())

        # Identify added and removed features
        added_keys = keys2 - keys1
        removed_keys = keys1 - keys2
        common_keys = keys1 & keys2

        added_features = [FeatureKey(k.split("/")) for k in sorted(added_keys)]
        removed_features = [FeatureKey(k.split("/")) for k in sorted(removed_keys)]

        # Identify changed features
        changed_features = []
        for key_str in sorted(common_keys):
            feature1 = snapshot1_data[key_str]
            feature2 = snapshot2_data[key_str]

            version1 = feature1["feature_version"]
            version2 = feature2["feature_version"]
            fields1 = feature1.get("fields", {})
            fields2 = feature2.get("fields", {})

            # Check if feature version changed or fields changed
            if version1 != version2 or fields1 != fields2:
                # Compute field changes
                field_changes = self._compute_field_changes(fields1, fields2)

                changed_features.append(
                    FeatureChange(
                        feature_key=FeatureKey(key_str.split("/")),
                        old_version=version1,
                        new_version=version2,
                        field_changes=field_changes,
                    )
                )

        # Determine snapshot versions from data
        # For now, use placeholder - caller should provide these
        snapshot1_version = "snapshot1"
        snapshot2_version = "snapshot2"

        return GraphDiff(
            snapshot1=snapshot1_version,
            snapshot2=snapshot2_version,
            added_features=added_features,
            removed_features=removed_features,
            changed_features=changed_features,
        )

    def _compute_field_changes(
        self, fields1: dict[str, str], fields2: dict[str, str]
    ) -> list[FieldChange]:
        """Compute changes between two field version mappings.

        Args:
            fields1: Field key (string) -> field version (hash) from snapshot1
            fields2: Field key (string) -> field version (hash) from snapshot2

        Returns:
            List of FieldChange objects
        """
        field_keys1 = set(fields1.keys())
        field_keys2 = set(fields2.keys())

        added_fields = field_keys2 - field_keys1
        removed_fields = field_keys1 - field_keys2
        common_fields = field_keys1 & field_keys2

        changes = []

        # Added fields
        for field_key_str in sorted(added_fields):
            changes.append(
                FieldChange(
                    field_key=FieldKey(field_key_str.split("/")),
                    old_version=None,
                    new_version=fields2[field_key_str],
                )
            )

        # Removed fields
        for field_key_str in sorted(removed_fields):
            changes.append(
                FieldChange(
                    field_key=FieldKey(field_key_str.split("/")),
                    old_version=fields1[field_key_str],
                    new_version=None,
                )
            )

        # Changed fields
        for field_key_str in sorted(common_fields):
            version1 = fields1[field_key_str]
            version2 = fields2[field_key_str]

            if version1 != version2:
                changes.append(
                    FieldChange(
                        field_key=FieldKey(field_key_str.split("/")),
                        old_version=version1,
                        new_version=version2,
                    )
                )

        return changes

    def create_merged_graph_data(
        self,
        snapshot1_data: dict[str, dict[str, Any]],
        snapshot2_data: dict[str, dict[str, Any]],
        diff: GraphDiff,
    ) -> dict[str, Any]:
        """Create merged graph data structure with status annotations.

        This combines features from both snapshots into a single unified view,
        annotating each feature with its status (added/removed/changed/unchanged).

        Args:
            snapshot1_data: First snapshot data (feature_key -> {feature_version, fields})
            snapshot2_data: Second snapshot data (feature_key -> {feature_version, fields})
            diff: Computed diff between snapshots

        Returns:
            Dict with structure:
            {
                'nodes': {
                    feature_key_str: {
                        'status': 'added' | 'removed' | 'changed' | 'unchanged',
                        'old_version': str | None,
                        'new_version': str | None,
                        'fields': {...},  # fields from relevant snapshot
                        'field_changes': [...],  # for changed nodes only
                        'dependencies': [feature_key_str, ...],  # deps from relevant snapshot
                    }
                },
                'edges': [
                    {'from': feature_key_str, 'to': feature_key_str}
                ]
            }
        """
        # Create status mapping for efficient lookup
        added_keys = {fk.to_string() for fk in diff.added_features}
        removed_keys = {fk.to_string() for fk in diff.removed_features}
        changed_keys = {fc.feature_key.to_string(): fc for fc in diff.changed_features}

        # Get all feature keys from both snapshots
        all_keys = set(snapshot1_data.keys()) | set(snapshot2_data.keys())

        nodes = {}
        edges = []

        for feature_key_str in all_keys:
            # Determine status
            if feature_key_str in added_keys:
                status = "added"
                old_version = None
                new_version = snapshot2_data[feature_key_str]["feature_version"]
                fields = snapshot2_data[feature_key_str].get("fields", {})
                field_changes = []
                # Dependencies from snapshot2
                deps = self._extract_dependencies(
                    snapshot2_data[feature_key_str].get("feature_spec", {})
                )
            elif feature_key_str in removed_keys:
                status = "removed"
                old_version = snapshot1_data[feature_key_str]["feature_version"]
                new_version = None
                fields = snapshot1_data[feature_key_str].get("fields", {})
                field_changes = []
                # Dependencies from snapshot1
                deps = self._extract_dependencies(
                    snapshot1_data[feature_key_str].get("feature_spec", {})
                )
            elif feature_key_str in changed_keys:
                status = "changed"
                feature_change = changed_keys[feature_key_str]
                old_version = feature_change.old_version
                new_version = feature_change.new_version
                fields = snapshot2_data[feature_key_str].get("fields", {})
                field_changes = feature_change.field_changes
                # Dependencies from snapshot2 (current version)
                deps = self._extract_dependencies(
                    snapshot2_data[feature_key_str].get("feature_spec", {})
                )
            else:
                # Unchanged
                status = "unchanged"
                old_version = snapshot1_data[feature_key_str]["feature_version"]
                new_version = snapshot2_data[feature_key_str]["feature_version"]
                fields = snapshot2_data[feature_key_str].get("fields", {})
                field_changes = []
                # Dependencies from snapshot2
                deps = self._extract_dependencies(
                    snapshot2_data[feature_key_str].get("feature_spec", {})
                )

            nodes[feature_key_str] = {
                "status": status,
                "old_version": old_version,
                "new_version": new_version,
                "fields": fields,
                "field_changes": field_changes,
                "dependencies": deps,
            }

            # Create edges for dependencies (arrow points from dependency to feature)
            for dep_key in deps:
                edges.append({"from": dep_key, "to": feature_key_str})

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def _extract_dependencies(self, feature_spec: dict[str, Any]) -> list[str]:
        """Extract dependency feature keys from a feature spec.

        Args:
            feature_spec: Parsed feature spec dict

        Returns:
            List of dependency feature keys as strings
        """
        deps = feature_spec.get("deps", [])
        if deps is None:
            return []

        dep_keys = []
        for dep in deps:
            # dep is a dict with 'key' field
            dep_key = dep.get("key", [])
            if isinstance(dep_key, list):
                dep_keys.append("/".join(dep_key))
            else:
                dep_keys.append(dep_key)

        return dep_keys

    def filter_merged_graph(
        self,
        merged_data: dict[str, Any],
        focus_feature: str | None = None,
        up: int | None = None,
        down: int | None = None,
    ) -> dict[str, Any]:
        """Filter merged graph to show only relevant features.

        Args:
            merged_data: Merged graph data with nodes and edges
            focus_feature: Feature key to focus on (string format with / or __)
            up: Number of upstream levels (None = all if focus_feature is set, 0 otherwise)
            down: Number of downstream levels (None = all if focus_feature is set, 0 otherwise)

        Returns:
            Filtered merged graph data with same structure

        Raises:
            ValueError: If focus_feature is specified but not found in graph
        """
        if focus_feature is None:
            # No filtering
            return merged_data

        # Parse feature key (support both / and __ formats)
        if "/" in focus_feature:
            focus_key = focus_feature
        else:
            focus_key = focus_feature.replace("__", "/")

        # Check if focus feature exists
        if focus_key not in merged_data["nodes"]:
            raise ValueError(f"Feature '{focus_feature}' not found in graph")

        # Build dependency graph for traversal
        # Build forward edges (feature -> dependents) and backward edges (feature -> dependencies)
        forward_edges: dict[str, list[str]] = {}  # feature -> list of dependents
        backward_edges: dict[str, list[str]] = {}  # feature -> list of dependencies

        for edge in merged_data["edges"]:
            dep = edge["from"]  # dependency
            feat = edge["to"]  # dependent feature

            if feat not in backward_edges:
                backward_edges[feat] = []
            backward_edges[feat].append(dep)

            if dep not in forward_edges:
                forward_edges[dep] = []
            forward_edges[dep].append(feat)

        # Find features to include
        features_to_include = {focus_key}

        # Add upstream (dependencies)
        # Default behavior: if focus_feature is set but up is not specified, include all upstream
        if up is None:
            # Include all upstream
            upstream = self._get_upstream_features(
                focus_key, backward_edges, max_levels=None
            )
            features_to_include.update(upstream)
        elif up > 0:
            # Include specified number of levels
            upstream = self._get_upstream_features(
                focus_key, backward_edges, max_levels=up
            )
            features_to_include.update(upstream)
        # else: up == 0, don't include upstream

        # Add downstream (dependents)
        # Default behavior: if focus_feature is set but down is not specified, include all downstream
        if down is None:
            # Include all downstream
            downstream = self._get_downstream_features(
                focus_key, forward_edges, max_levels=None
            )
            features_to_include.update(downstream)
        elif down > 0:
            # Include specified number of levels
            downstream = self._get_downstream_features(
                focus_key, forward_edges, max_levels=down
            )
            features_to_include.update(downstream)
        # else: down == 0, don't include downstream

        # Filter nodes and edges
        filtered_nodes = {
            k: v for k, v in merged_data["nodes"].items() if k in features_to_include
        }
        filtered_edges = [
            e
            for e in merged_data["edges"]
            if e["from"] in features_to_include and e["to"] in features_to_include
        ]

        return {
            "nodes": filtered_nodes,
            "edges": filtered_edges,
        }

    def _get_upstream_features(
        self,
        start_key: str,
        backward_edges: dict[str, list[str]],
        max_levels: int | None = None,
        visited: set[str] | None = None,
        level: int = 0,
    ) -> set[str]:
        """Get upstream features (dependencies) recursively."""
        if visited is None:
            visited = set()

        if start_key in visited:
            return set()

        if max_levels is not None and level >= max_levels:
            return set()

        visited.add(start_key)
        upstream: set[str] = set()

        deps = backward_edges.get(start_key, [])
        for dep in deps:
            if dep not in visited:
                upstream.add(dep)
                # Recurse
                upstream.update(
                    self._get_upstream_features(
                        dep, backward_edges, max_levels, visited, level + 1
                    )
                )

        return upstream

    def _get_downstream_features(
        self,
        start_key: str,
        forward_edges: dict[str, list[str]],
        max_levels: int | None = None,
        visited: set[str] | None = None,
        level: int = 0,
    ) -> set[str]:
        """Get downstream features (dependents) recursively."""
        if visited is None:
            visited = set()

        if start_key in visited:
            return set()

        if max_levels is not None and level >= max_levels:
            return set()

        visited.add(start_key)
        downstream: set[str] = set()

        dependents = forward_edges.get(start_key, [])
        for dependent in dependents:
            if dependent not in visited:
                downstream.add(dependent)
                # Recurse
                downstream.update(
                    self._get_downstream_features(
                        dependent, forward_edges, max_levels, visited, level + 1
                    )
                )

        return downstream

    def load_snapshot_data(
        self, store: MetadataStore, snapshot_version: str
    ) -> dict[str, dict[str, Any]]:
        """Load snapshot data from store.

        Args:
            store: Metadata store to query
            snapshot_version: Snapshot version to load

        Returns:
            Dict mapping feature_key (string) -> {feature_version, fields}
            where fields is dict mapping field_key (string) -> field_version (hash)

        Raises:
            ValueError: If snapshot not found in store
        """
        # Query feature_versions table for this snapshot
        try:
            features_lazy = store._read_metadata_native(FEATURE_VERSIONS_KEY)
            if features_lazy is None:
                raise ValueError(
                    f"No feature_versions table found in store. Cannot load snapshot {snapshot_version}."
                )

            # Filter by snapshot_version
            import narwhals as nw

            features_df = (
                features_lazy.filter(nw.col("snapshot_version") == snapshot_version)
                .collect()
                .to_polars()
            )

            if features_df.height == 0:
                raise ValueError(
                    f"Snapshot {snapshot_version} not found in store. "
                    "Run 'metaxy graph push' to record snapshots or check the version hash."
                )

        except Exception as e:
            raise ValueError(f"Failed to load snapshot {snapshot_version}: {e}") from e

        # Build snapshot data structure for FeatureGraph.from_snapshot()
        snapshot_dict = {}
        for row in features_df.iter_rows(named=True):
            feature_key_str = row["feature_key"]
            feature_version = row["feature_version"]
            feature_spec_json = row["feature_spec"]
            feature_class_path = row.get("feature_class_path", "")

            feature_spec_dict = json.loads(feature_spec_json)

            snapshot_dict[feature_key_str] = {
                "feature_version": feature_version,
                "feature_spec": feature_spec_dict,
                "feature_class_path": feature_class_path,
            }

        # Reconstruct FeatureGraph from snapshot to compute field versions
        try:
            graph = FeatureGraph.from_snapshot(snapshot_dict)
        except Exception as e:
            # If we can't reconstruct the graph (e.g., feature classes moved),
            # fall back to using feature_version as field version (best effort)
            import warnings

            warnings.warn(
                f"Could not reconstruct graph from snapshot {snapshot_version}: {e}. "
                "Using feature_version as field_version (field-level diffs may be inaccurate)."
            )
            # Fall back to old behavior
            snapshot_data = {}
            for feature_key_str, data in snapshot_dict.items():
                feature_spec_dict = data["feature_spec"]
                fields_data = {}
                for field_dict in feature_spec_dict.get("fields", []):
                    field_key_list = field_dict.get("key")
                    if isinstance(field_key_list, list):
                        field_key_str_normalized = "/".join(field_key_list)
                    else:
                        field_key_str_normalized = field_key_list
                    fields_data[field_key_str_normalized] = data["feature_version"]

                snapshot_data[feature_key_str] = {
                    "feature_version": data["feature_version"],
                    "fields": fields_data,
                    "feature_spec": feature_spec_dict,
                }
            return snapshot_data

        # Now compute proper field versions using the reconstructed graph
        from metaxy.models.plan import FQFieldKey

        snapshot_data = {}
        for feature_key_str in snapshot_dict.keys():
            feature_version = snapshot_dict[feature_key_str]["feature_version"]
            feature_spec = snapshot_dict[feature_key_str]["feature_spec"]
            feature_key_obj = FeatureKey(feature_key_str.split("/"))

            # Compute field versions using graph
            fields_data = {}
            for field_dict in feature_spec.get("fields", []):
                field_key_list = field_dict.get("key")
                if isinstance(field_key_list, list):
                    field_key = FieldKey(field_key_list)
                    field_key_str_normalized = "/".join(field_key_list)
                else:
                    field_key = FieldKey([field_key_list])
                    field_key_str_normalized = field_key_list

                # Compute field version using the graph
                fq_key = FQFieldKey(feature=feature_key_obj, field=field_key)
                field_version = graph.get_field_version(fq_key)
                fields_data[field_key_str_normalized] = field_version

            snapshot_data[feature_key_str] = {
                "feature_version": feature_version,
                "fields": fields_data,
                "feature_spec": feature_spec,
            }

        return snapshot_data
