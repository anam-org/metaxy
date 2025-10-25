"""Cards renderer using Rich panels for graph visualization.

Requires rich library to be installed.
"""

from metaxy.graph.renderers.base import GraphRenderer
from metaxy.models.plan import FQFieldKey


class TerminalCardsRenderer(GraphRenderer):
    """Renders graph as cards with edges for terminal display.

    Uses Rich panels to show features as cards/boxes with dependency information.
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

        # Build feature panels in topological order
        feature_panels = []
        feature_labels = {}  # key -> display label

        for feature_key in self._get_topological_order():
            feature_cls = self.graph.features_by_key[feature_key]
            panel = self._build_feature_panel(feature_key, feature_cls)
            feature_panels.append(panel)
            feature_labels[feature_key] = self._format_feature_key(feature_key)

        # Build edges representation
        edges_text = Text()
        if self.config.show_snapshot_version:
            snapshot_version = self._format_hash(self.graph.snapshot_version)
            edges_text.append(
                f"📊 Graph (snapshot: {snapshot_version})\n\n", style="bold"
            )
        else:
            edges_text.append("📊 Graph\n\n", style="bold")

        # Show dependency edges
        edges_text.append("Dependencies:\n", style="bold cyan")
        for feature_key in self._get_topological_order():
            feature_cls = self.graph.features_by_key[feature_key]
            if feature_cls.spec.deps:
                source_label = self._format_feature_key(feature_key)
                for dep in feature_cls.spec.deps:
                    if dep.key in self.graph.features_by_key:
                        target_label = self._format_feature_key(dep.key)
                        edges_text.append(f"  {target_label} ", style="cyan")
                        edges_text.append("→", style="yellow bold")
                        edges_text.append(f" {source_label}\n", style="cyan")

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

    def _build_feature_panel(self, feature_key, feature_cls):
        """Build a Rich Panel for a feature.

        Args:
            feature_key: Feature key
            feature_cls: Feature class

        Returns:
            Rich Panel with feature information
        """
        from rich.panel import Panel
        from rich.text import Text

        content = Text()

        # Feature name
        content.append(self._format_feature_key(feature_key), style="bold cyan")
        content.append("\n")

        # Versions
        if self.config.show_feature_versions:
            version = self._format_hash(feature_cls.feature_version())
            content.append(f"v: {version}", style="yellow")
            content.append("\n")

        if self.config.show_code_versions:
            content.append(f"cv: {feature_cls.spec.code_version}", style="dim")
            content.append("\n")

        # Fields
        if self.config.show_fields and feature_cls.spec.fields:
            content.append("\nFields:\n", style="bold green")
            for field in feature_cls.spec.fields:
                field_text = self._format_field_info(feature_key, field)
                content.append(f"  • {field_text}\n")

        return Panel(content, border_style="cyan", padding=(0, 1))

    def _format_field_info(self, feature_key, field_spec) -> str:
        """Format field information as a string.

        Args:
            feature_key: Feature key (for version lookup)
            field_spec: Field specification

        Returns:
            Formatted field string
        """
        parts = [self._format_field_key(field_spec.key)]

        if self.config.show_field_versions:
            fq_key = FQFieldKey(feature=feature_key, field=field_spec.key)
            version = self._format_hash(self.graph.get_field_version(fq_key))
            parts.append(f"(v: {version})")

        if self.config.show_code_versions:
            parts.append(f"(cv: {field_spec.code_version})")

        # Add field dependencies
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
                    parts.append(f"← {', '.join(dep_strs)}")

        return " ".join(parts)
