"""Mermaid renderer for flowchart generation.

Requires mermaid-py library to be installed.
"""

from metaxy.graph.diff.models import GraphNode, NodeStatus
from metaxy.graph.diff.rendering.base import BaseRenderer
from metaxy.graph.utils import sanitize_mermaid_id
from metaxy.models.types import FeatureKey


class MermaidRenderer(BaseRenderer):
    """Generates Mermaid flowchart markup using mermaid-py.

    Creates flowchart with type-safe API.
    Supports both normal and diff rendering via node status.
    """

    def render(self) -> str:
        """Render graph as Mermaid flowchart.

        Returns:
            Mermaid markup as string
        """
        from mermaid.flowchart import FlowChart, Link, Node

        # Get filtered graph data
        filtered_graph = self._get_filtered_graph_data()

        # Create nodes with fields as sub-items in the label
        from metaxy.graph.diff.traversal import GraphWalker

        walker = GraphWalker(filtered_graph)
        nodes = []
        node_map = {}  # feature_key string -> Node

        for graph_node in walker.topological_sort():
            node_id = self._node_id_from_key(graph_node.key)

            # Build label with fields inside
            label = self._build_feature_label_with_fields(graph_node)

            # Choose shape based on status
            shape = self._get_node_shape(graph_node)

            node = Node(id_=node_id, content=label, shape=shape)
            nodes.append(node)
            node_map[graph_node.key.to_string()] = node

        # Create links for dependencies
        links = []
        for graph_node in filtered_graph.nodes.values():
            if graph_node.dependencies:
                target_node = node_map.get(graph_node.key.to_string())
                if target_node:
                    for dep_key in graph_node.dependencies:
                        source_node = node_map.get(dep_key.to_string())
                        if source_node:
                            links.append(Link(origin=source_node, end=target_node))

        # Create flowchart
        title = "Feature Graph"

        chart = FlowChart(
            title=title,
            nodes=nodes,
            links=links,
            orientation=self.config.direction,
        )

        script = chart.script

        # Modify script to add styling and snapshot version
        lines = script.split("\n")

        # Find the flowchart line
        for i, line in enumerate(lines):
            if line.startswith("flowchart "):
                insertions = []

                # Add snapshot version comment if needed
                if self.config.show_snapshot_version:
                    snapshot_hash = self._format_hash(filtered_graph.snapshot_version)
                    insertions.append(f"    %% Snapshot version: {snapshot_hash}")

                # Add styling
                insertions.append(
                    "    %%{init: {'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '14px'}}}%%"
                )

                # Insert all additions after the flowchart line
                for j, insertion in enumerate(insertions):
                    lines.insert(i + 1 + j, insertion)
                break

        script = "\n".join(lines)

        # Add color styling for diff nodes if in diff mode
        if self._is_diff_mode(filtered_graph):
            script = self._add_diff_styling(script, filtered_graph)

        return script

    def _node_id_from_key(self, key: FeatureKey) -> str:
        """Generate valid node ID from feature key.

        Args:
            key: Feature key

        Returns:
            Valid node identifier (lowercase, no special chars)
        """
        return sanitize_mermaid_id(key.to_string()).lower()

    def _get_node_shape(self, node: GraphNode) -> str:
        """Get Mermaid node shape based on status.

        Args:
            node: GraphNode

        Returns:
            Mermaid shape name
        """
        # Use different shapes for diff mode
        if node.status == NodeStatus.REMOVED:
            return "stadium"  # Rounded box for removed
        elif node.status == NodeStatus.ADDED:
            return "round"  # Rounded corners for added
        else:
            return "normal"  # Standard rectangle

    def _is_diff_mode(self, graph_data) -> bool:
        """Check if rendering in diff mode.

        Args:
            graph_data: GraphData

        Returns:
            True if any node has non-NORMAL status
        """
        return any(
            node.status != NodeStatus.NORMAL for node in graph_data.nodes.values()
        )

    def _add_diff_styling(self, script: str, graph_data) -> str:
        """Add color styling for diff nodes.

        Args:
            script: Mermaid script
            graph_data: GraphData with status information

        Returns:
            Modified script with style classes
        """
        lines = script.split("\n")

        # Find position to insert style classes (before closing line)
        insert_idx = len(lines)

        style_lines = []

        # Add style classes for each node based on status
        for node in graph_data.nodes.values():
            if node.status == NodeStatus.NORMAL:
                continue

            node_id = self._node_id_from_key(node.key)

            if node.status == NodeStatus.ADDED:
                # Only color the border, no fill
                style_lines.append(
                    f"    style {node_id} stroke:{self.theme.added_color},stroke-width:2px"
                )
            elif node.status == NodeStatus.REMOVED:
                # Only color the border, no fill
                style_lines.append(
                    f"    style {node_id} stroke:{self.theme.removed_color},stroke-width:2px"
                )
            elif node.status == NodeStatus.CHANGED:
                # Only color the border, no fill
                style_lines.append(
                    f"    style {node_id} stroke:{self.theme.changed_color},stroke-width:2px"
                )
            elif node.status == NodeStatus.UNCHANGED:
                # Only color the border, no fill
                style_lines.append(
                    f"    style {node_id} stroke:{self.theme.unchanged_color}"
                )

        # Insert style lines before the end
        if style_lines:
            lines = lines[:insert_idx] + [""] + style_lines + lines[insert_idx:]

        return "\n".join(lines)

    def _build_feature_label_with_fields(self, node: GraphNode) -> str:
        """Build label for feature node with fields displayed inside.

        Args:
            node: GraphNode

        Returns:
            Formatted label with feature info and fields as sub-items
        """
        lines = []

        # Feature key (bold)
        feature_name = self._format_feature_key(node.key)
        lines.append(f"<b>{feature_name}</b>")

        # Add project info if configured
        if self.config.show_projects and node.project:
            lines.append(
                f'<small><font color="#666">Project: {node.project}</font></small>'
            )

        # Add status badge for diff mode
        if node.status != NodeStatus.NORMAL:
            badge = self._get_status_badge_html(node.status)
            lines.append(badge)

        # Feature version info
        if self.config.show_feature_versions or self.config.show_code_versions:
            version_parts = []

            if self.config.show_feature_versions:
                if node.status == NodeStatus.CHANGED and node.old_version is not None:
                    # Show version transition
                    old_v = self._format_hash(node.old_version)
                    new_v = self._format_hash(node.version)
                    version_parts.append(f"v: {old_v} → {new_v}")
                else:
                    version = self._format_hash(node.version)
                    version_parts.append(f"v: {version}")

            if self.config.show_code_versions and node.code_version is not None:
                version_parts.append(f"cv: {node.code_version}")

            lines.append(f"<small>({', '.join(version_parts)})</small>")

        # Fields (if configured)
        if self.config.show_fields and node.fields:
            # Subtle separator line before fields
            lines.append('<font color="#999">---</font>')
            for field_node in node.fields:
                field_line = self._build_field_line(field_node)
                lines.append(field_line)

        # Wrap content in a div with left alignment
        content = "<br/>".join(lines)
        return f'<div style="text-align:left">{content}</div>'

    def _build_field_line(self, field_node) -> str:
        """Build single line for field display.

        Args:
            field_node: FieldNode

        Returns:
            Formatted field line
        """
        parts = [f"• {self._format_field_key(field_node.key)}"]

        # Add status badge
        if field_node.status != NodeStatus.NORMAL:
            badge = self._get_status_badge_text(field_node.status)
            parts[0] = f"{parts[0]} {badge}"

        if self.config.show_field_versions or self.config.show_code_versions:
            version_parts = []

            if self.config.show_field_versions:
                if (
                    field_node.status == NodeStatus.CHANGED
                    and field_node.old_version is not None
                ):
                    # Show version transition
                    old_v = self._format_hash(field_node.old_version)
                    new_v = self._format_hash(field_node.version)
                    version_parts.append(f"v: {old_v} → {new_v}")
                else:
                    version = self._format_hash(field_node.version)
                    version_parts.append(f"v: {version}")

            if self.config.show_code_versions and field_node.code_version is not None:
                version_parts.append(f"cv: {field_node.code_version}")

            parts.append(f"<small>({', '.join(version_parts)})</small>")

        return " ".join(parts)

    def _get_status_badge_html(self, status: NodeStatus) -> str:
        """Get HTML status badge.

        Args:
            status: Node status

        Returns:
            HTML badge string
        """
        if status == NodeStatus.ADDED:
            return '<small><font color="green">[+]</font></small>'
        elif status == NodeStatus.REMOVED:
            return '<small><font color="red">[-]</font></small>'
        elif status == NodeStatus.CHANGED:
            return '<small><font color="orange">[~]</font></small>'
        else:
            return ""

    def _get_status_badge_text(self, status: NodeStatus) -> str:
        """Get text-only status badge.

        Args:
            status: Node status

        Returns:
            Text badge string
        """
        if status == NodeStatus.ADDED:
            return "[+]"
        elif status == NodeStatus.REMOVED:
            return "[-]"
        elif status == NodeStatus.CHANGED:
            return "[~]"
        else:
            return ""
