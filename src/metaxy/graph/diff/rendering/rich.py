"""Terminal renderer using Rich Tree for hierarchical display.

Requires rich library to be installed.
"""

from metaxy.graph.diff.models import NodeStatus
from metaxy.graph.diff.rendering.base import BaseRenderer


class TerminalRenderer(BaseRenderer):
    """Renders graph using Rich Tree for terminal display.

    Creates a hierarchical tree view with colors and icons.
    Supports both normal and diff rendering via node status.
    """

    def render(self) -> str:
        """Render graph as Rich Tree for terminal.

        Returns:
            Rendered tree as string with ANSI color codes
        """
        from rich.console import Console
        from rich.tree import Tree

        console = Console()

        # Get filtered graph data based on config
        filtered_graph = self._get_filtered_graph_data()

        # Create root node
        if self.config.show_snapshot_version:
            snapshot_version = self._format_hash(filtered_graph.snapshot_version)
            root = Tree(f"📊 [bold]Graph[/bold] [dim](snapshot: {snapshot_version})[/dim]")
        else:
            root = Tree("📊 [bold]Graph[/bold]")

        # Create walker for filtered graph and add features in topological order
        from metaxy.graph.diff.traversal import GraphWalker

        walker = GraphWalker(filtered_graph)
        for node in walker.topological_sort():
            self._render_feature_node(root, node)

        # Render to string
        with console.capture() as capture:
            console.print(root)
        return capture.get()

    def _render_feature_node(self, parent, node):
        """Add a feature node to the tree.

        Args:
            parent: Parent tree node
            node: GraphNode
        """
        # Get status color
        status_color = self._get_status_color(node.status)

        # Build feature label
        label_parts = [f"[{status_color}]{self._format_feature_key(node.key)}[/{status_color}]"]

        # Show project if configured
        if self.config.show_projects and node.project:
            label_parts.append(f"[dim](project: {node.project})[/dim]")

        # Show version info
        if self.config.show_feature_versions:
            if node.status == NodeStatus.CHANGED and node.old_version is not None:
                # Show version transition for changed nodes
                version_transition = self._format_version_transition(node.old_version, node.version)
                label_parts.append(version_transition)
            else:
                # Normal version display
                version = self._format_hash(node.version)
                label_parts.append(f"[yellow](v: {version})[/yellow]")

        if self.config.show_code_versions and node.code_version is not None:
            label_parts.append(f"[dim](cv: {node.code_version})[/dim]")

        # Add status badge for diff mode
        if node.status != NodeStatus.NORMAL:
            status_badge = self._get_status_badge(node.status)
            label_parts.append(status_badge)

        label = " ".join(label_parts)
        feature_branch = parent.add(label)

        # Add fields
        if self.config.show_fields and node.fields:
            fields_branch = feature_branch.add("🔧 [green]fields[/green]")
            for field_node in node.fields:
                self._render_field_node(fields_branch, field_node)

        # Add dependencies
        if node.dependencies:
            deps_branch = feature_branch.add("⬅️  [blue]depends on[/blue]")
            for dep_key in node.dependencies:
                dep_color = status_color  # Use same color as parent for simplicity
                deps_branch.add(f"[{dep_color}]{self._format_feature_key(dep_key)}[/{dep_color}]")

    def _render_field_node(self, parent, field_node):
        """Add a field node to the tree.

        Args:
            parent: Parent tree node
            field_node: FieldNode
        """
        # Get status color
        status_color = self._get_status_color(field_node.status)

        label_parts = [f"[{status_color}]{self._format_field_key(field_node.key)}[/{status_color}]"]

        # Show version info
        if self.config.show_field_versions:
            if field_node.status == NodeStatus.CHANGED and field_node.old_version is not None:
                # Show version transition for changed fields
                version_transition = self._format_version_transition(field_node.old_version, field_node.version)
                label_parts.append(version_transition)
            else:
                # Normal version display
                version = self._format_hash(field_node.version)
                label_parts.append(f"[yellow](v: {version})[/yellow]")

        if self.config.show_code_versions and field_node.code_version is not None:
            label_parts.append(f"[dim](cv: {field_node.code_version})[/dim]")

        # Add status badge for diff mode
        if field_node.status != NodeStatus.NORMAL:
            status_badge = self._get_status_badge(field_node.status)
            label_parts.append(status_badge)

        label = " ".join(label_parts)
        parent.add(label)

    def _get_status_badge(self, status: NodeStatus) -> str:
        """Get status badge text with color.

        Args:
            status: Node status

        Returns:
            Rich-formatted status badge
        """
        if status == NodeStatus.ADDED:
            return f"[{self.theme.added_color}][+][/{self.theme.added_color}]"
        elif status == NodeStatus.REMOVED:
            return f"[{self.theme.removed_color}][-][/{self.theme.removed_color}]"
        elif status == NodeStatus.CHANGED:
            return f"[{self.theme.changed_color}][~][/{self.theme.changed_color}]"
        elif status == NodeStatus.UNCHANGED:
            return ""  # No badge for unchanged
        else:
            return ""  # No badge for normal
