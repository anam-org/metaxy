"""Graph diffing logic and snapshot resolution."""

from collections.abc import Mapping
from typing import Any

from metaxy.graph.diff.diff_models import (
    AddedNode,
    FieldChange,
    GraphDiff,
    NodeChange,
    RemovedNode,
)
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey, FieldKey


class SnapshotResolver:
    """Resolves snapshot version literals to actual snapshot hashes."""

    def resolve_snapshot(self, literal: str, store: MetadataStore | None, graph: FeatureGraph | None) -> str:
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
                    "Cannot resolve 'latest': no metadata store provided. Provide a store to query for snapshots."
                )
            return self._resolve_latest(store)
        elif literal == "current":
            return self._resolve_current(graph)
        else:
            # Treat as explicit snapshot version
            return literal

    def _resolve_latest(self, store: MetadataStore) -> str:
        """Resolve 'latest' to most recent snapshot in store."""
        from metaxy.metadata_store.system.storage import SystemTableStorage

        with store:
            storage = SystemTableStorage(store)
            snapshots_df = storage.read_graph_snapshots()

        if snapshots_df.height == 0:
            raise ValueError(
                "No snapshots found in store. Cannot resolve 'latest'. Run 'metaxy graph push' to record a snapshot."
            )

        # read_graph_snapshots() returns sorted by recorded_at descending
        latest_snapshot = snapshots_df["metaxy_snapshot_version"][0]
        return latest_snapshot

    def _resolve_current(self, graph: FeatureGraph | None) -> str:
        """Resolve 'current' to active graph's snapshot version."""
        if graph is None:
            raise ValueError(
                "Cannot resolve 'current': no active graph provided. Ensure features are loaded before using 'current'."
            )

        if len(graph.feature_definitions_by_key) == 0:
            raise ValueError(
                "Cannot resolve 'current': active graph is empty. Ensure features are loaded before using 'current'."
            )

        return graph.snapshot_version


class GraphDiffer:
    """Compares two graph snapshots and produces a diff."""

    def diff(
        self,
        snapshot1_data: Mapping[str, Mapping[str, Any]],
        snapshot2_data: Mapping[str, Mapping[str, Any]],
        from_snapshot_version: str = "unknown",
        to_snapshot_version: str = "unknown",
    ) -> GraphDiff:
        """Compute diff between two snapshots.

        Args:
            snapshot1_data: First snapshot (feature_key -> {feature_version, feature_spec, fields})
            snapshot2_data: Second snapshot (feature_key -> {feature_version, feature_spec, fields})
            from_snapshot_version: Source snapshot version
            to_snapshot_version: Target snapshot version

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

        # Build added nodes
        added_nodes = []
        for key_str in sorted(added_keys):
            feature_data = snapshot2_data[key_str]
            feature_spec = feature_data.get("feature_spec", {})

            # Extract fields
            fields_list = []
            for field_dict in feature_spec.get("fields", []):
                field_key_list = field_dict.get("key", [])
                field_key_str = "/".join(field_key_list) if isinstance(field_key_list, list) else field_key_list
                fields_list.append(
                    {
                        "key": field_key_str,
                        "version": feature_data.get("fields", {}).get(field_key_str, ""),
                        "code_version": field_dict.get("code_version"),
                    }
                )

            # Extract dependencies
            deps = []
            if feature_spec.get("deps"):
                for dep in feature_spec["deps"]:
                    dep_key = dep.get("feature") or dep.get("key", [])
                    if isinstance(dep_key, list):
                        deps.append(FeatureKey(dep_key))
                    else:
                        deps.append(FeatureKey(dep_key.split("/")))

            added_nodes.append(
                AddedNode(
                    feature_key=FeatureKey(key_str.split("/")),
                    version=feature_data["metaxy_feature_version"],
                    code_version=feature_spec.get("code_version"),
                    fields=fields_list,
                    dependencies=deps,
                )
            )

        # Build removed nodes
        removed_nodes = []
        for key_str in sorted(removed_keys):
            feature_data = snapshot1_data[key_str]
            feature_spec = feature_data.get("feature_spec", {})

            # Extract fields
            fields_list = []
            for field_dict in feature_spec.get("fields", []):
                field_key_list = field_dict.get("key", [])
                field_key_str = "/".join(field_key_list) if isinstance(field_key_list, list) else field_key_list
                fields_list.append(
                    {
                        "key": field_key_str,
                        "version": feature_data.get("fields", {}).get(field_key_str, ""),
                        "code_version": field_dict.get("code_version"),
                    }
                )

            # Extract dependencies
            deps = []
            if feature_spec.get("deps"):
                for dep in feature_spec["deps"]:
                    dep_key = dep.get("feature") or dep.get("key", [])
                    if isinstance(dep_key, list):
                        deps.append(FeatureKey(dep_key))
                    else:
                        deps.append(FeatureKey(dep_key.split("/")))

            removed_nodes.append(
                RemovedNode(
                    feature_key=FeatureKey(key_str.split("/")),
                    version=feature_data["metaxy_feature_version"],
                    code_version=feature_spec.get("code_version"),
                    fields=fields_list,
                    dependencies=deps,
                )
            )

        # Identify changed features
        changed_nodes = []
        for key_str in sorted(common_keys):
            feature1 = snapshot1_data[key_str]
            feature2 = snapshot2_data[key_str]

            version1 = feature1["metaxy_feature_version"]
            version2 = feature2["metaxy_feature_version"]

            spec1 = feature1.get("feature_spec", {})
            spec2 = feature2.get("feature_spec", {})

            fields1 = feature1.get("fields", {})
            fields2 = feature2.get("fields", {})

            # Get tracking versions for migration detection
            # Use tracking version if available (new system), otherwise fall back to feature_version
            tracking_version1 = feature1.get("metaxy_definition_version", version1)
            tracking_version2 = feature2.get("metaxy_definition_version", version2)

            # Check if feature tracking version changed (indicates migration needed)
            if tracking_version1 != tracking_version2:
                # Compute field changes
                field_changes = self._compute_field_changes(fields1, fields2)

                changed_nodes.append(
                    NodeChange(
                        feature_key=FeatureKey(key_str.split("/")),
                        old_version=version1,
                        new_version=version2,
                        old_code_version=spec1.get("code_version"),
                        new_code_version=spec2.get("code_version"),
                        added_fields=[fc for fc in field_changes if fc.is_added],
                        removed_fields=[fc for fc in field_changes if fc.is_removed],
                        changed_fields=[fc for fc in field_changes if fc.is_changed],
                    )
                )

        return GraphDiff(
            from_snapshot_version=from_snapshot_version,
            to_snapshot_version=to_snapshot_version,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            changed_nodes=changed_nodes,
        )

    def _compute_field_changes(self, fields1: dict[str, str], fields2: dict[str, str]) -> list[FieldChange]:
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
                    old_code_version=None,
                    new_code_version=None,
                )
            )

        # Removed fields
        for field_key_str in sorted(removed_fields):
            changes.append(
                FieldChange(
                    field_key=FieldKey(field_key_str.split("/")),
                    old_version=fields1[field_key_str],
                    new_version=None,
                    old_code_version=None,
                    new_code_version=None,
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
                        old_code_version=None,
                        new_code_version=None,
                    )
                )

        return changes

    def create_merged_graph_data(
        self,
        snapshot1_data: Mapping[str, Mapping[str, Any]],
        snapshot2_data: Mapping[str, Mapping[str, Any]],
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
                        'field_changes': [...],  # FieldChange objects for changed nodes
                        'dependencies': [feature_key_str, ...],  # deps from relevant snapshot
                    }
                },
                'edges': [
                    {'from': feature_key_str, 'to': feature_key_str}
                ]
            }
        """
        # Create status mapping for efficient lookup
        added_keys = {node.feature_key.to_string() for node in diff.added_nodes}
        removed_keys = {node.feature_key.to_string() for node in diff.removed_nodes}
        changed_keys = {node.feature_key.to_string(): node for node in diff.changed_nodes}

        # Get all feature keys from both snapshots
        all_keys = set(snapshot1_data.keys()) | set(snapshot2_data.keys())

        nodes = {}
        edges = []

        for feature_key_str in all_keys:
            # Determine status
            if feature_key_str in added_keys:
                status = "added"
                old_version = None
                new_version = snapshot2_data[feature_key_str]["metaxy_feature_version"]
                fields = snapshot2_data[feature_key_str].get("fields", {})
                field_changes = []
                # Dependencies from snapshot2
                deps = self._extract_dependencies(snapshot2_data[feature_key_str].get("feature_spec", {}))
            elif feature_key_str in removed_keys:
                status = "removed"
                old_version = snapshot1_data[feature_key_str]["metaxy_feature_version"]
                new_version = None
                fields = snapshot1_data[feature_key_str].get("fields", {})
                field_changes = []
                # Dependencies from snapshot1
                deps = self._extract_dependencies(snapshot1_data[feature_key_str].get("feature_spec", {}))
            elif feature_key_str in changed_keys:
                status = "changed"
                node_change = changed_keys[feature_key_str]
                old_version = node_change.old_version
                new_version = node_change.new_version
                fields = snapshot2_data[feature_key_str].get("fields", {})
                # Combine all field changes from the NodeChange
                field_changes = node_change.added_fields + node_change.removed_fields + node_change.changed_fields
                # Dependencies from snapshot2 (current version)
                deps = self._extract_dependencies(snapshot2_data[feature_key_str].get("feature_spec", {}))
            else:
                # Unchanged
                status = "unchanged"
                old_version = snapshot1_data[feature_key_str]["metaxy_feature_version"]
                new_version = snapshot2_data[feature_key_str]["metaxy_feature_version"]
                fields = snapshot2_data[feature_key_str].get("fields", {})
                field_changes = []
                # Dependencies from snapshot2
                deps = self._extract_dependencies(snapshot2_data[feature_key_str].get("feature_spec", {}))

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
            dep_key = dep.get("feature") or dep.get("key", [])
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
            upstream = self._get_upstream_features(focus_key, backward_edges, max_levels=None)
            features_to_include.update(upstream)
        elif up > 0:
            # Include specified number of levels
            upstream = self._get_upstream_features(focus_key, backward_edges, max_levels=up)
            features_to_include.update(upstream)
        # else: up == 0, don't include upstream

        # Add downstream (dependents)
        # Default behavior: if focus_feature is set but down is not specified, include all downstream
        if down is None:
            # Include all downstream
            downstream = self._get_downstream_features(focus_key, forward_edges, max_levels=None)
            features_to_include.update(downstream)
        elif down > 0:
            # Include specified number of levels
            downstream = self._get_downstream_features(focus_key, forward_edges, max_levels=down)
            features_to_include.update(downstream)
        # else: down == 0, don't include downstream

        # Filter nodes and edges
        filtered_nodes = {k: v for k, v in merged_data["nodes"].items() if k in features_to_include}
        filtered_edges = [
            e for e in merged_data["edges"] if e["from"] in features_to_include and e["to"] in features_to_include
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
                upstream.update(self._get_upstream_features(dep, backward_edges, max_levels, visited, level + 1))

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
                    self._get_downstream_features(dependent, forward_edges, max_levels, visited, level + 1)
                )

        return downstream

    def load_snapshot_data(
        self,
        store: MetadataStore,
        snapshot_version: str,
        project: str | None = None,
    ) -> Mapping[str, Mapping[str, Any]]:
        """Load snapshot data from store.

        Args:
            store: Metadata store to query
            snapshot_version: Snapshot version to load
            project: Optional project name to filter by (None means all projects)

        Returns:
            Dict mapping feature_key (string) -> {feature_version, feature_spec, fields}
            where fields is dict mapping field_key (string) -> field_version (hash)

        Raises:
            ValueError: If snapshot not found in store
        """
        from metaxy.metadata_store.system.storage import SystemTableStorage

        # Auto-open store if not already open
        if not store._is_open:
            with store.open("read"):
                return self.load_snapshot_data(store, snapshot_version, project)

        storage = SystemTableStorage(store)

        # Reconstruct the graph from the snapshot using FeatureDefinition objects
        graph = storage.load_graph_from_snapshot(
            snapshot_version=snapshot_version,
            project=project,
        )

        # Build snapshot data using the reconstructed graph
        snapshot_data: dict[str, dict] = {}

        for feature_key, definition in graph.feature_definitions_by_key.items():
            feature_key_str = feature_key.to_string()
            feature_version = graph.get_feature_version(feature_key)
            field_versions = graph.get_feature_version_by_field(feature_key)

            snapshot_data[feature_key_str] = {
                "metaxy_feature_version": feature_version,
                "fields": field_versions,
                "feature_spec": definition.spec.model_dump(mode="json"),
                "metaxy_definition_version": definition.feature_definition_version,
            }

        return snapshot_data
