"""Terminal renderer using Rich Tree for hierarchical display.

Requires rich library to be installed.
"""

from metaxy.graph.renderers.base import GraphRenderer
from metaxy.models.plan import FQFieldKey


class TerminalRenderer(GraphRenderer):
    """Renders graph using Rich Tree for terminal display.

    Creates a hierarchical tree view with colors and icons.
    """

    def render(self) -> str:
        """Render graph as Rich Tree for terminal.

        Returns:
            Rendered tree as string with ANSI color codes
        """
        from rich.console import Console
        from rich.tree import Tree

        console = Console()

        # Create root node
        if self.config.show_snapshot_id:
            snapshot_id = self._format_hash(self.graph.snapshot_id)
            root = Tree(f"📊 [bold]Graph[/bold] [dim](snapshot: {snapshot_id})[/dim]")
        else:
            root = Tree("📊 [bold]Graph[/bold]")

        # Add features in topological order
        for feature_key in self._get_topological_order():
            feature_cls = self.graph.features_by_key[feature_key]
            self._render_feature_node(root, feature_key, feature_cls)

        # Render to string
        with console.capture() as capture:
            console.print(root)
        return capture.get()

    def _render_feature_node(self, parent, feature_key, feature_cls):
        """Add a feature node to the tree.

        Args:
            parent: Parent tree node
            feature_key: Feature key
            feature_cls: Feature class
        """
        # Build feature label
        label_parts = [f"[cyan]{self._format_feature_key(feature_key)}[/cyan]"]

        if self.config.show_feature_versions:
            version = self._format_hash(feature_cls.feature_version())
            label_parts.append(f"[yellow](v: {version})[/yellow]")

        if self.config.show_code_versions:
            label_parts.append(f"[dim](cv: {feature_cls.spec.code_version})[/dim]")

        label = " ".join(label_parts)
        feature_branch = parent.add(label)

        # Add fields
        if self.config.show_fields and feature_cls.spec.fields:
            fields_branch = feature_branch.add("🔧 [green]fields[/green]")
            for field in feature_cls.spec.fields:
                self._render_field_node(fields_branch, feature_key, field)

        # Add dependencies
        if feature_cls.spec.deps:
            deps_branch = feature_branch.add("⬅️  [blue]depends on[/blue]")
            for dep in feature_cls.spec.deps:
                deps_branch.add(f"[cyan]{self._format_feature_key(dep.key)}[/cyan]")

    def _render_field_node(self, parent, feature_key, field_spec):
        """Add a field node to the tree.

        Args:
            parent: Parent tree node
            feature_key: Feature key (for version lookup)
            field_spec: Field specification
        """
        label_parts = [self._format_field_key(field_spec.key)]

        if self.config.show_field_versions:
            fq_key = FQFieldKey(feature=feature_key, field=field_spec.key)
            version = self._format_hash(self.graph.get_field_version(fq_key))
            label_parts.append(f"[yellow](v: {version})[/yellow]")

        if self.config.show_code_versions:
            label_parts.append(f"[dim](cv: {field_spec.code_version})[/dim]")

        # Add field dependencies if present
        if field_spec.deps:
            from metaxy.models.field import SpecialFieldDep

            if isinstance(field_spec.deps, SpecialFieldDep):
                # Special dependency (e.g., ALL) - skip displaying as it's the default sentinel
                pass
            else:
                # List of specific field dependencies
                dep_strs = []
                for dep in field_spec.deps:
                    dep_feature = self._format_feature_key(dep.feature_key)

                    # Check if this dep uses ALL fields or specific fields
                    if isinstance(dep.fields, SpecialFieldDep):
                        # ALL fields - just show the feature name
                        dep_strs.append(dep_feature)
                    else:
                        # Specific fields
                        for field_key in dep.fields:
                            dep_field = self._format_field_key(field_key)
                            dep_strs.append(f"{dep_feature}.{dep_field}")
                if dep_strs:
                    label_parts.append(f"[dim]← {', '.join(dep_strs)}[/dim]")

        label = " ".join(label_parts)
        parent.add(label)
