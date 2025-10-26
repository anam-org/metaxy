"""Core data models for graph rendering."""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import Field

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.types import FeatureKey, FieldKey

if TYPE_CHECKING:
    from metaxy.models.feature import FeatureGraph


class NodeStatus(str, Enum):
    """Status of a node in a diff view."""

    NORMAL = "normal"  # Normal node (not in diff mode)
    UNCHANGED = "unchanged"  # Unchanged in diff
    ADDED = "added"  # Added in diff
    REMOVED = "removed"  # Removed in diff
    CHANGED = "changed"  # Changed in diff


class FieldNode(FrozenBaseModel):
    """Represents a field within a feature node.

    Attributes:
        key: Field key
        version: Current field version hash
        old_version: Previous field version hash (for diffs)
        code_version: Code version (if available)
        status: Field status (for diff rendering)
    """

    key: FieldKey
    version: str | None = None  # None if field was removed
    old_version: str | None = None  # For diff mode
    code_version: int | None = None
    status: NodeStatus = NodeStatus.NORMAL


class GraphNode(FrozenBaseModel):
    """Represents a feature node in the graph.

    Attributes:
        key: Feature key
        version: Current feature version hash
        old_version: Previous feature version hash (for diffs)
        code_version: Code version (if available)
        fields: List of field nodes
        dependencies: List of feature keys this node depends on
        status: Node status (for diff rendering)
        metadata: Additional custom metadata
    """

    key: FeatureKey
    version: str | None = None  # None if feature was removed
    old_version: str | None = None  # For diff mode
    code_version: int | None = None
    fields: list[FieldNode] = Field(default_factory=list)
    dependencies: list[FeatureKey] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.NORMAL
    metadata: dict[str, Any] = Field(default_factory=dict)


class EdgeData(FrozenBaseModel):
    """Represents an edge between two nodes.

    Attributes:
        from_key: Source feature key (dependency)
        to_key: Target feature key (dependent)
    """

    from_key: FeatureKey
    to_key: FeatureKey


class GraphData(FrozenBaseModel):
    """Container for complete graph structure.

    This is the unified data model used by all renderers.

    Attributes:
        nodes: Map from feature key string to GraphNode
        edges: List of edges
        snapshot_version: Optional snapshot version
        old_snapshot_version: Optional old snapshot version (for diffs)
    """

    nodes: dict[str, GraphNode]  # Key is feature_key.to_string()
    edges: list[EdgeData] = Field(default_factory=list)
    snapshot_version: str | None = None
    old_snapshot_version: str | None = None  # For diff mode

    def get_node(self, key: FeatureKey) -> GraphNode | None:
        """Get node by feature key.

        Args:
            key: Feature key to lookup

        Returns:
            GraphNode if found, None otherwise
        """
        return self.nodes.get(key.to_string())

    def get_nodes_by_status(self, status: NodeStatus) -> list[GraphNode]:
        """Get all nodes with a specific status.

        Args:
            status: Status to filter by

        Returns:
            List of nodes with matching status
        """
        return [node for node in self.nodes.values() if node.status == status]

    @classmethod
    def from_feature_graph(cls, graph: "FeatureGraph") -> "GraphData":
        """Convert a FeatureGraph to GraphData.

        Args:
            graph: FeatureGraph instance

        Returns:
            GraphData with all nodes and edges
        """
        from metaxy.models.plan import FQFieldKey

        nodes: dict[str, GraphNode] = {}
        edges: list[EdgeData] = []

        # Convert each feature to a GraphNode
        for feature_key, feature_cls in graph.features_by_key.items():
            feature_key_str = feature_key.to_string()
            spec = feature_cls.spec

            # Get feature version
            feature_version = graph.get_feature_version(feature_key)

            # Convert fields
            field_nodes: list[FieldNode] = []
            if spec.fields:
                for field_spec in spec.fields:
                    # Compute field version
                    fq_field_key = FQFieldKey(feature=feature_key, field=field_spec.key)
                    field_version = graph.get_field_version(fq_field_key)

                    field_node = FieldNode(
                        key=field_spec.key,
                        version=field_version,
                        code_version=field_spec.code_version,
                        status=NodeStatus.NORMAL,
                    )
                    field_nodes.append(field_node)

            # Extract dependencies
            dependencies: list[FeatureKey] = []
            if spec.deps:
                dependencies = [dep.key for dep in spec.deps]

            # Create node
            node = GraphNode(
                key=feature_key,
                version=feature_version,
                code_version=spec.code_version,
                fields=field_nodes,
                dependencies=dependencies,
                status=NodeStatus.NORMAL,
            )
            nodes[feature_key_str] = node

            # Create edges
            for dep_key in dependencies:
                edges.append(EdgeData(from_key=dep_key, to_key=feature_key))

        return cls(
            nodes=nodes,
            edges=edges,
            snapshot_version=graph.snapshot_version,
        )

    @classmethod
    def from_merged_diff(cls, merged_data: dict[str, Any]) -> "GraphData":
        """Convert merged diff data to GraphData.

        Args:
            merged_data: Merged diff data from GraphDiffer.create_merged_graph_data()

        Returns:
            GraphData with status annotations
        """
        from metaxy.graph_diff import FieldChange

        nodes: dict[str, GraphNode] = {}
        edges: list[EdgeData] = []

        # Convert nodes
        for feature_key_str, node_data in merged_data["nodes"].items():
            # Parse feature key
            feature_key = FeatureKey(feature_key_str.split("/"))

            # Map status strings to NodeStatus enum
            status_str = node_data["status"]
            if status_str == "added":
                status = NodeStatus.ADDED
            elif status_str == "removed":
                status = NodeStatus.REMOVED
            elif status_str == "changed":
                status = NodeStatus.CHANGED
            elif status_str == "unchanged":
                status = NodeStatus.UNCHANGED
            else:
                status = NodeStatus.NORMAL

            # Convert fields
            fields_dict = node_data.get("fields", {})
            field_changes_list = node_data.get("field_changes", [])

            # Build field change map for quick lookup
            field_change_map: dict[str, FieldChange] = {}
            for fc in field_changes_list:
                if isinstance(fc, FieldChange):
                    field_change_map[fc.field_key.to_string()] = fc

            # Get all field keys (from both current fields and removed fields in changes)
            all_field_keys = set(fields_dict.keys())
            all_field_keys.update(field_change_map.keys())

            field_nodes: list[FieldNode] = []
            for field_key_str in all_field_keys:
                # Parse field key
                field_key = FieldKey(field_key_str.split("/"))

                # Determine field status and versions
                if field_key_str in field_change_map:
                    fc = field_change_map[field_key_str]
                    if fc.is_added:
                        field_status = NodeStatus.ADDED
                        field_version = fc.new_version
                        old_field_version = None
                    elif fc.is_removed:
                        field_status = NodeStatus.REMOVED
                        field_version = None
                        old_field_version = fc.old_version
                    elif fc.is_changed:
                        field_status = NodeStatus.CHANGED
                        field_version = fc.new_version
                        old_field_version = fc.old_version
                    else:
                        field_status = NodeStatus.UNCHANGED
                        field_version = fc.new_version or fc.old_version
                        old_field_version = None
                else:
                    # Unchanged field
                    field_status = NodeStatus.UNCHANGED
                    field_version = fields_dict.get(field_key_str)
                    old_field_version = None

                field_node = FieldNode(
                    key=field_key,
                    version=field_version,
                    old_version=old_field_version,
                    status=field_status,
                )
                field_nodes.append(field_node)

            # Parse dependencies
            dependencies = [
                FeatureKey(dep_str.split("/"))
                for dep_str in node_data.get("dependencies", [])
            ]

            # Create node
            node = GraphNode(
                key=feature_key,
                version=node_data.get("new_version"),
                old_version=node_data.get("old_version"),
                fields=field_nodes,
                dependencies=dependencies,
                status=status,
            )
            nodes[feature_key_str] = node

        # Convert edges
        for edge_dict in merged_data["edges"]:
            from_key = FeatureKey(edge_dict["from"].split("/"))
            to_key = FeatureKey(edge_dict["to"].split("/"))
            edges.append(EdgeData(from_key=from_key, to_key=to_key))

        return cls(
            nodes=nodes,
            edges=edges,
        )
