"""Core data models for graph rendering."""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import Field
from typing_extensions import Self

from metaxy.models.bases import FrozenBaseModel
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.utils.constants import DEFAULT_CODE_VERSION
from metaxy.utils.exceptions import MetaxyEmptyCodeVersionError

if TYPE_CHECKING:
    from metaxy.models.feature import FeatureGraph


class NodeStatus(str, Enum):
    """Status of a node in a diff view."""

    NORMAL = "normal"  # Normal node (not in diff mode)
    UNCHANGED = "unchanged"  # Unchanged in diff
    ADDED = "added"  # Added in diff
    REMOVED = "removed"  # Removed in diff
    CHANGED = "changed"  # Changed in diff

    @classmethod
    def from_string(cls, status_str: str) -> "NodeStatus":
        """Convert a status string to NodeStatus enum.

        Args:
            status_str: Status string (added, removed, changed, unchanged)

        Returns:
            Corresponding NodeStatus enum value
        """
        status_map = {
            "added": cls.ADDED,
            "removed": cls.REMOVED,
            "changed": cls.CHANGED,
            "unchanged": cls.UNCHANGED,
        }
        return status_map.get(status_str, cls.NORMAL)


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
    code_version: str | None = None
    status: NodeStatus = NodeStatus.NORMAL

    @classmethod
    def from_field_change(cls, field_key: FieldKey, field_change: Any) -> "FieldNode":
        """Create a FieldNode from a FieldChange object.

        Args:
            field_key: The field key
            field_change: FieldChange instance containing version info

        Returns:
            FieldNode with appropriate status and versions
        """
        from metaxy.graph.diff.diff_models import FieldChange

        if not isinstance(field_change, FieldChange):
            msg = f"Expected FieldChange, got {type(field_change)}"
            raise TypeError(msg)

        if field_change.is_added:
            return cls(
                key=field_key,
                version=field_change.new_version,
                old_version=None,
                status=NodeStatus.ADDED,
            )
        if field_change.is_removed:
            return cls(
                key=field_key,
                version=None,
                old_version=field_change.old_version,
                status=NodeStatus.REMOVED,
            )
        if field_change.is_changed:
            return cls(
                key=field_key,
                version=field_change.new_version,
                old_version=field_change.old_version,
                status=NodeStatus.CHANGED,
            )
        # Unchanged
        return cls(
            key=field_key,
            version=field_change.new_version or field_change.old_version,
            old_version=None,
            status=NodeStatus.UNCHANGED,
        )

    @classmethod
    def from_unchanged(cls, field_key: FieldKey, version: str | None) -> "FieldNode":
        """Create an unchanged FieldNode.

        Args:
            field_key: The field key
            version: The field version

        Returns:
            FieldNode with UNCHANGED status
        """
        return cls(
            key=field_key,
            version=version,
            old_version=None,
            status=NodeStatus.UNCHANGED,
        )


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
        project: Project name this feature belongs to
        metadata: Additional custom metadata
    """

    key: FeatureKey
    version: str | None = None  # None if feature was removed
    old_version: str | None = None  # For diff mode
    code_version: str | None = None
    fields: list[FieldNode] = Field(default_factory=list)
    dependencies: list[FeatureKey] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.NORMAL
    project: str | None = None  # Project name (None for legacy features)
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

    def to_struct(self) -> dict[str, Any]:
        """Serialize to struct (native Python types for storage).

        Note: This uses custom serialization instead of Pydantic's model_dump() because:
        1. Polars struct columns require specific type conversions (e.g., None → "" for strings, None → 0 for ints)
        2. Custom types (FeatureKey, FieldKey) need explicit string conversion for storage
        3. The storage schema is a separate concern from the domain model's Python representation
        4. Different storage backends may need different serialization formats in the future

        Returns:
            Dict with structure compatible with Polars struct type
        """
        nodes_list = []
        for node in self.nodes.values():
            fields_list = []
            for field in node.fields:
                if field.code_version is None:
                    raise MetaxyEmptyCodeVersionError(
                        f"Field {field.key.to_string()} in feature {node.key.to_string()} has empty code_version."
                    )
                fields_list.append(
                    {
                        "key": field.key.to_string(),
                        "version": field.version if field.version is not None else "",
                        "code_version": field.code_version,
                    }
                )

            if node.code_version is None:
                raise MetaxyEmptyCodeVersionError(
                    f"Feature {node.key.to_string()} has empty code_version."
                )
            nodes_list.append(
                {
                    "key": node.key.to_string(),
                    "version": node.version if node.version is not None else "",
                    "code_version": node.code_version,
                    "fields": fields_list,
                    "dependencies": [dep.to_string() for dep in node.dependencies],
                    "project": node.project if node.project is not None else "",
                }
            )

        edges_list = []
        for edge in self.edges:
            edges_list.append(
                {
                    "from_key": edge.from_key.to_string(),
                    "to_key": edge.to_key.to_string(),
                }
            )

        result: dict[str, Any] = {
            "nodes": nodes_list,
            "edges": edges_list,
        }

        # Include snapshot_version if present
        if self.snapshot_version is not None:
            result["metaxy_snapshot_version"] = self.snapshot_version

        # Include old_snapshot_version if present (for diffs)
        if self.old_snapshot_version is not None:
            result["old_snapshot_version"] = self.old_snapshot_version

        return result

    @classmethod
    def from_struct(cls, struct_data: dict[str, Any]) -> Self:
        """Deserialize from struct.

        Args:
            struct_data: Dict with structure from to_struct()

        Returns:
            GraphData instance
        """
        nodes = {}
        for node_data in struct_data["nodes"]:
            fields = []
            for field_data in node_data["fields"]:
                if (
                    field_data["code_version"] == ""
                    or field_data["code_version"] is None
                    or field_data["code_version"] == DEFAULT_CODE_VERSION
                ):
                    raise MetaxyEmptyCodeVersionError(
                        f"Field {field_data['key']} in feature {node_data['key']} has empty code_version."
                    )
                fields.append(
                    FieldNode(
                        key=FieldKey(field_data["key"].split("/")),
                        version=field_data["version"]
                        if field_data["version"]
                        else None,
                        code_version=field_data["code_version"],
                    )
                )

            if (
                node_data["code_version"] == ""
                or node_data["code_version"] is None
                or node_data["code_version"] == DEFAULT_CODE_VERSION
            ):
                raise MetaxyEmptyCodeVersionError(
                    f"Feature {node_data['key']} has empty code_version."
                )
            node = GraphNode(
                key=FeatureKey(node_data["key"].split("/")),
                version=node_data["version"] if node_data["version"] else None,
                code_version=node_data["code_version"],
                fields=fields,
                dependencies=[
                    FeatureKey(dep.split("/")) for dep in node_data["dependencies"]
                ],
                project=node_data.get("project") if node_data.get("project") else None,
            )
            nodes[node_data["key"]] = node

        edges = []
        for edge_data in struct_data["edges"]:
            edges.append(
                EdgeData(
                    from_key=FeatureKey(edge_data["from_key"].split("/")),
                    to_key=FeatureKey(edge_data["to_key"].split("/")),
                )
            )

        # Extract snapshot_version if present
        snapshot_version = struct_data.get("metaxy_snapshot_version")

        # Extract old_snapshot_version if present (for diffs)
        old_snapshot_version = struct_data.get("old_snapshot_version")

        return cls(
            nodes=nodes,
            edges=edges,
            snapshot_version=snapshot_version,
            old_snapshot_version=old_snapshot_version,
        )

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
            spec = feature_cls.spec()

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
                dependencies = [dep.feature for dep in spec.deps]

            # Get project from feature class
            feature_project = feature_cls.project

            # Create node
            node = GraphNode(
                key=feature_key,
                version=feature_version,
                fields=field_nodes,
                dependencies=dependencies,
                status=NodeStatus.NORMAL,
                project=feature_project,
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
    def from_snapshot(cls, snapshot_data: dict[str, Any]) -> "GraphData":
        """Convert snapshot data to GraphData for rendering.

        Args:
            snapshot_data: Snapshot data in format {feature_key_str -> {metaxy_feature_version, fields, feature_spec}}

        Returns:
            GraphData with all nodes and edges
        """
        nodes: dict[str, GraphNode] = {}
        edges: list[EdgeData] = []

        # Convert each feature to a GraphNode
        for feature_key_str, feature_data in snapshot_data.items():
            feature_key = FeatureKey(feature_key_str.split("/"))
            feature_version = feature_data.get("metaxy_feature_version")
            fields_dict = feature_data.get("fields", {})
            feature_spec = feature_data.get("feature_spec", {})

            # Convert fields
            field_nodes: list[FieldNode] = []
            for field_key_str, field_version in fields_dict.items():
                field_key = FieldKey(field_key_str.split("/"))
                field_node = FieldNode(
                    key=field_key,
                    version=field_version,
                    status=NodeStatus.NORMAL,
                )
                field_nodes.append(field_node)

            # Extract dependencies from feature_spec
            dependencies: list[FeatureKey] = []
            deps_list = feature_spec.get("deps", [])
            for dep in deps_list:
                dep_feature = dep.get("feature")
                if dep_feature:
                    dependencies.append(FeatureKey(dep_feature.split("/")))

            # Get project from metadata if available
            metadata = feature_spec.get("metadata", {})
            project = metadata.get("project")

            # Create node
            node = GraphNode(
                key=feature_key,
                version=feature_version,
                fields=field_nodes,
                dependencies=dependencies,
                status=NodeStatus.NORMAL,
                project=project,
            )
            nodes[feature_key_str] = node

            # Create edges
            for dep_key in dependencies:
                edges.append(EdgeData(from_key=dep_key, to_key=feature_key))

        return cls(
            nodes=nodes,
            edges=edges,
        )

    @classmethod
    def from_merged_diff(cls, merged_data: dict[str, Any]) -> "GraphData":
        """Convert merged diff data to GraphData.

        Args:
            merged_data: Merged diff data from GraphDiffer.create_merged_graph_data()

        Returns:
            GraphData with status annotations
        """
        nodes: dict[str, GraphNode] = {}

        # Convert nodes
        for feature_key_str, node_data in merged_data["nodes"].items():
            node = cls._convert_merged_node(feature_key_str, node_data)
            nodes[feature_key_str] = node

        # Convert edges
        edges = cls._convert_merged_edges(merged_data["edges"])

        return cls(nodes=nodes, edges=edges)

    @classmethod
    def _convert_merged_node(
        cls, feature_key_str: str, node_data: dict[str, Any]
    ) -> GraphNode:
        """Convert a merged node dict to a GraphNode.

        Args:
            feature_key_str: Feature key string
            node_data: Node data dict from merged diff

        Returns:
            GraphNode with status annotations
        """
        feature_key = FeatureKey(feature_key_str.split("/"))
        status = NodeStatus.from_string(node_data["status"])

        # Convert fields
        field_nodes = cls._convert_merged_fields(node_data)

        # Parse dependencies
        dependencies = [
            FeatureKey(dep_str.split("/"))
            for dep_str in node_data.get("dependencies", [])
        ]

        return GraphNode(
            key=feature_key,
            version=node_data.get("new_version"),
            old_version=node_data.get("old_version"),
            fields=field_nodes,
            dependencies=dependencies,
            status=status,
        )

    @classmethod
    def _convert_merged_fields(cls, node_data: dict[str, Any]) -> list[FieldNode]:
        """Convert merged field data to FieldNodes.

        Args:
            node_data: Node data dict containing fields and field_changes

        Returns:
            List of FieldNodes with status annotations
        """
        from metaxy.graph.diff.diff_models import FieldChange

        fields_dict = node_data.get("fields", {})
        field_changes_list = node_data.get("field_changes", [])

        # Build field change map for quick lookup
        field_change_map: dict[str, FieldChange] = {
            fc.field_key.to_string(): fc
            for fc in field_changes_list
            if isinstance(fc, FieldChange)
        }

        # Get all field keys (from both current fields and removed fields in changes)
        all_field_keys = set(fields_dict.keys()) | set(field_change_map.keys())

        field_nodes: list[FieldNode] = []
        for field_key_str in all_field_keys:
            field_key = FieldKey(field_key_str.split("/"))

            if field_key_str in field_change_map:
                field_node = FieldNode.from_field_change(
                    field_key, field_change_map[field_key_str]
                )
            else:
                field_node = FieldNode.from_unchanged(
                    field_key, fields_dict.get(field_key_str)
                )
            field_nodes.append(field_node)

        return field_nodes

    @classmethod
    def _convert_merged_edges(cls, edges_data: list[dict[str, str]]) -> list[EdgeData]:
        """Convert merged edge data to EdgeData objects.

        Args:
            edges_data: List of edge dicts with 'from' and 'to' keys

        Returns:
            List of EdgeData objects
        """
        return [
            EdgeData(
                from_key=FeatureKey(edge_dict["from"].split("/")),
                to_key=FeatureKey(edge_dict["to"].split("/")),
            )
            for edge_dict in edges_data
        ]
