"""Cards renderer using Rich panels for graph visualization.

Requires rich library to be installed.
"""

from metaxy.graph.diff.models import NodeStatus
from metaxy.graph.diff.rendering.base import BaseRenderer


class CardsRenderer(BaseRenderer):
    """Renders graph as cards with edges for terminal display.

    Uses Rich panels to show features as cards/boxes with dependency information.
    Supports both normal and diff rendering via node status.
    """

    def render(self) -> str:
        """Render graph as cards.

        Returns:
            Rendered cards as string with ANSI color codes
        """
        from rich.columns import Columns
        from rich.console import Console, Group
        from rich.text import Text

        console = Console()

        # Get filtered graph data based on config
        filtered_graph = self._get_filtered_graph_data()

        # Build feature panels in topological order
        from metaxy.graph.diff.traversal import GraphWalker

        walker = GraphWalker(filtered_graph)
        feature_panels = []

        for node in walker.topological_sort():
            panel = self._build_feature_panel(node)
            feature_panels.append(panel)

        # Build edges representation
        edges_text = Text()
        if self.config.show_snapshot_version:
            snapshot_version = self._format_hash(filtered_graph.snapshot_version)
            edges_text.append(f"📊 Graph (snapshot: {snapshot_version})\n\n", style="bold")
        else:
            edges_text.append("📊 Graph\n\n", style="bold")

        # Show dependency edges
        edges_text.append("Dependencies:\n", style="bold cyan")
        for node in walker.topological_sort():
            if node.dependencies:
                source_label = self._format_feature_key(node.key)
                source_color = self._get_status_color(node.status)
                for dep_key in node.dependencies:
                    dep_node = filtered_graph.get_node(dep_key)
                    target_label = self._format_feature_key(dep_key)
                    target_color = self._get_status_color(dep_node.status) if dep_node else source_color
                    edges_text.append(f"  {target_label} ", style=target_color)
                    edges_text.append("→", style="yellow bold")
                    edges_text.append(f" {source_label}\n", style=source_color)

        # Combine everything
        output_group = Group(
            edges_text,
            Text("\nFeatures:", style="bold"),
            Columns(feature_panels, equal=True, expand=True),
        )

        # Render to string
        with console.capture() as capture:
            console.print(output_group)
        return capture.get()

    def _build_feature_panel(self, node):
        """Build a Rich Panel for a feature.

        Args:
            node: GraphNode

        Returns:
            Rich Panel with feature information
        """
        from rich.panel import Panel
        from rich.text import Text

        content = Text()

        # Get status color
        status_color = self._get_status_color(node.status)
        border_color = status_color

        # Feature name
        content.append(self._format_feature_key(node.key), style=f"bold {status_color}")

        # Add status badge for diff mode
        if node.status != NodeStatus.NORMAL:
            status_badge = self._get_status_badge(node.status)
            content.append(f" {status_badge}")

        content.append("\n")

        # Versions
        if self.config.show_feature_versions:
            if node.status == NodeStatus.CHANGED and node.old_version is not None:
                # Show version transition for changed nodes
                old_v = self._format_hash(node.old_version)
                new_v = self._format_hash(node.version)
                content.append(f"v: {old_v} → {new_v}", style="yellow")
            else:
                version = self._format_hash(node.version)
                content.append(f"v: {version}", style="yellow")
            content.append("\n")

        if self.config.show_code_versions and node.code_version is not None:
            content.append(f"cv: {node.code_version}", style="dim")
            content.append("\n")

        # Fields
        if self.config.show_fields and node.fields:
            content.append("\nFields:\n", style="bold green")
            for field_node in node.fields:
                field_text = self._format_field_info(field_node)
                content.append(f"  • {field_text}\n")

        return Panel(content, border_style=border_color, padding=(0, 1))

    def _format_field_info(self, field_node) -> str:
        """Format field information as a string.

        Args:
            field_node: FieldNode

        Returns:
            Formatted field string
        """
        parts = [self._format_field_key(field_node.key)]

        # Show version info
        if self.config.show_field_versions:
            if field_node.status == NodeStatus.CHANGED and field_node.old_version is not None:
                # Show version transition for changed fields
                old_v = self._format_hash(field_node.old_version)
                new_v = self._format_hash(field_node.version)
                parts.append(f"(v: {old_v} → {new_v})")
            else:
                version = self._format_hash(field_node.version)
                parts.append(f"(v: {version})")

        if self.config.show_code_versions and field_node.code_version is not None:
            parts.append(f"(cv: {field_node.code_version})")

        # Add status badge for diff mode
        if field_node.status != NodeStatus.NORMAL:
            status_badge = self._get_status_badge(field_node.status)
            parts.append(status_badge)

        return " ".join(parts)

    def _get_status_badge(self, status: NodeStatus) -> str:
        """Get status badge text.

        Args:
            status: Node status

        Returns:
            Status badge string
        """
        if status == NodeStatus.ADDED:
            return "[+]"
        elif status == NodeStatus.REMOVED:
            return "[-]"
        elif status == NodeStatus.CHANGED:
            return "[~]"
        elif status == NodeStatus.UNCHANGED:
            return ""
        else:
            return ""
