"""Graphviz renderer for DOT format generation.

Requires pygraphviz library to be installed.
"""

from metaxy.graph.diff.models import GraphNode, NodeStatus
from metaxy.graph.diff.rendering.base import BaseRenderer


class GraphvizRenderer(BaseRenderer):
    """Renders graph using pygraphviz.

    Creates DOT format output using pygraphviz library.
    Requires pygraphviz to be installed as optional dependency.
    Supports both normal and diff rendering via node status.
    """

    def render(self) -> str:
        """Render graph as Graphviz DOT format.

        Returns:
            DOT format as string
        """
        lines = []

        # Get filtered graph data
        filtered_graph = self._get_filtered_graph_data()

        # Graph header
        rankdir = self.config.direction
        lines.append("strict digraph {")
        lines.append(f"    rankdir={rankdir};")

        # Graph attributes
        if self.config.show_snapshot_version:
            label = f"Graph (snapshot: {self._format_hash(filtered_graph.snapshot_version)})"
        else:
            label = "Graph"
        lines.append(f'    label="{label}";')
        lines.append("    labelloc=t;")
        lines.append("    fontsize=14;")
        lines.append("    fontname=helvetica;")
        lines.append("")

        # Add nodes for features
        from metaxy.graph.diff.traversal import GraphWalker

        walker = GraphWalker(filtered_graph)
        for node in walker.topological_sort():
            node_id = node.key.to_string()
            label = self._build_feature_label(node)

            # Choose shape and color based on status
            shape = self._get_node_shape(node)
            color = self._get_node_color(node)

            lines.append(f'    "{node_id}" [label="{label}", shape={shape}, color="{color}"];')

        lines.append("")

        # Add edges for feature dependencies
        for node in filtered_graph.nodes.values():
            if node.dependencies:
                target_id = node.key.to_string()
                for dep_key in node.dependencies:
                    source_id = dep_key.to_string()
                    lines.append(f'    "{source_id}" -> "{target_id}";')

        lines.append("")

        # Add field nodes if configured
        if self.config.show_fields:
            for node in filtered_graph.nodes.values():
                parent_id = node.key.to_string()

                if not node.fields:
                    continue

                for field_node in node.fields:
                    field_id = f"{parent_id}::{field_node.key.to_string()}"
                    label = self._build_field_label(field_node)

                    # Get field color based on status
                    color = self._get_field_color(field_node)

                    lines.append(f'    "{field_id}" [label="{label}", shape=ellipse, color="{color}", fontsize=10];')

                    # Connect field to feature with dashed line
                    lines.append(f'    "{parent_id}" -> "{field_id}" [style=dashed, arrowhead=none];')

        lines.append("}")

        return "\n".join(lines)

    def _get_node_shape(self, node: GraphNode) -> str:
        """Get Graphviz shape based on node properties.

        Args:
            node: GraphNode

        Returns:
            Graphviz shape name
        """
        # Root features get special shape
        if not node.dependencies:
            return "doubleoctagon"

        # Different shapes for diff mode
        if node.status == NodeStatus.REMOVED:
            return "octagon"
        elif node.status == NodeStatus.ADDED:
            return "oval"
        else:
            return "box"

    def _get_node_color(self, node: GraphNode) -> str:
        """Get border color based on node status.

        Args:
            node: GraphNode

        Returns:
            Graphviz color (hex)
        """
        if node.status == NodeStatus.ADDED:
            return self.theme.added_color
        elif node.status == NodeStatus.REMOVED:
            return self.theme.removed_color
        elif node.status == NodeStatus.CHANGED:
            return self.theme.changed_color
        elif node.status == NodeStatus.UNCHANGED:
            return self.theme.unchanged_color
        else:
            return self.theme.feature_color

    def _get_field_color(self, field_node) -> str:
        """Get border color for field based on status.

        Args:
            field_node: FieldNode

        Returns:
            Graphviz color (hex)
        """
        if field_node.status == NodeStatus.ADDED:
            return self.theme.added_color
        elif field_node.status == NodeStatus.REMOVED:
            return self.theme.removed_color
        elif field_node.status == NodeStatus.CHANGED:
            return self.theme.changed_color
        elif field_node.status == NodeStatus.UNCHANGED:
            return self.theme.unchanged_color
        else:
            return self.theme.field_color

    def _build_feature_label(self, node: GraphNode) -> str:
        """Build label for feature node.

        Args:
            node: GraphNode

        Returns:
            Formatted label with optional version info
        """
        parts = [self._format_feature_key(node.key)]

        # Add status badge
        if node.status != NodeStatus.NORMAL:
            badge = self._get_status_badge(node.status)
            parts.append(f"\\n{badge}")

        if self.config.show_feature_versions:
            if node.status == NodeStatus.CHANGED and node.old_version is not None:
                # Show version transition
                old_v = self._format_hash(node.old_version)
                new_v = self._format_hash(node.version)
                parts.append(f"\\nv: {old_v} → {new_v}")
            else:
                version = self._format_hash(node.version)
                parts.append(f"\\nv: {version}")

        if self.config.show_code_versions and node.code_version is not None:
            parts.append(f"\\ncv: {node.code_version}")

        return "".join(parts)

    def _build_field_label(self, field_node) -> str:
        """Build label for field node.

        Args:
            field_node: FieldNode

        Returns:
            Formatted label with optional version info
        """
        parts = [self._format_field_key(field_node.key)]

        # Add status badge
        if field_node.status != NodeStatus.NORMAL:
            badge = self._get_status_badge(field_node.status)
            parts.append(f"\\n{badge}")

        if self.config.show_field_versions:
            if field_node.status == NodeStatus.CHANGED and field_node.old_version is not None:
                # Show version transition
                old_v = self._format_hash(field_node.old_version)
                new_v = self._format_hash(field_node.version)
                parts.append(f"\\nv: {old_v} → {new_v}")
            else:
                version = self._format_hash(field_node.version)
                parts.append(f"\\nv: {version}")

        if self.config.show_code_versions and field_node.code_version is not None:
            parts.append(f"\\ncv: {field_node.code_version}")

        return "".join(parts)

    def _get_status_badge(self, status: NodeStatus) -> str:
        """Get status badge text.

        Args:
            status: Node status

        Returns:
            Badge string
        """
        if status == NodeStatus.ADDED:
            return "[ADDED]"
        elif status == NodeStatus.REMOVED:
            return "[REMOVED]"
        elif status == NodeStatus.CHANGED:
            return "[CHANGED]"
        elif status == NodeStatus.UNCHANGED:
            return "[UNCHANGED]"
        else:
            return ""
