"""Mermaid renderer for flowchart generation.

Requires mermaid-py library to be installed.
"""

from metaxy.graph import utils
from metaxy.graph.renderers.base import GraphRenderer
from metaxy.models.plan import FQFieldKey
from metaxy.models.types import FeatureKey


class MermaidRenderer(GraphRenderer):
    """Generates Mermaid flowchart markup using mermaid-py.

    Creates flowchart with type-safe API.
    """

    def render(self) -> str:
        """Render graph as Mermaid flowchart.

        Returns:
            Mermaid markup as string
        """
        from mermaid.flowchart import FlowChart, Link, Node

        # Create nodes with fields as sub-items in the label
        nodes = []
        node_map = {}  # feature_key -> Node

        for feature_key in self._get_topological_order():
            feature_cls = self.graph.features_by_key[feature_key]
            node_id = self._node_id_from_key(feature_key)

            # Build label with fields inside
            label = self._build_feature_label_with_fields(feature_key, feature_cls)

            node = Node(id_=node_id, content=label, shape="normal")
            nodes.append(node)
            node_map[feature_key] = node

        # Create links for dependencies
        links = []
        for feature_key, feature_cls in self.graph.features_by_key.items():
            if feature_cls.spec.deps:
                target_node = node_map[feature_key]
                for dep in feature_cls.spec.deps:
                    if dep.key in node_map:
                        source_node = node_map[dep.key]
                        links.append(Link(origin=source_node, end=target_node))

        # Create flowchart
        # Note: We avoid complex titles in YAML frontmatter as GitHub's Mermaid
        # renderer has stricter YAML parsing. Use simple title or comment instead.
        title = "Feature Graph"

        chart = FlowChart(
            title=title,
            nodes=nodes,
            links=links,
            orientation=self.config.direction,
        )

        script = chart.script

        # Modify script to add flexible node width styling and snapshot version
        lines = script.split("\n")

        # Find the flowchart line
        for i, line in enumerate(lines):
            if line.startswith("flowchart "):
                insertions = []

                # Add snapshot version comment if needed
                if self.config.show_snapshot_version:
                    snapshot_hash = self._format_hash(self.graph.snapshot_version)
                    insertions.append(f"    %% Snapshot version: {snapshot_hash}")

                # Add styling for flexible node width
                # This ensures text doesn't get cut off
                insertions.append(
                    "    %%{init: {'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '14px'}}}%%"
                )

                # Insert all additions after the flowchart line
                for j, insertion in enumerate(insertions):
                    lines.insert(i + 1 + j, insertion)
                break

        script = "\n".join(lines)
        return script

    def _node_id_from_key(self, key: FeatureKey) -> str:
        """Generate valid node ID from feature key.

        Args:
            key: Feature key

        Returns:
            Valid node identifier (lowercase, no special chars)
        """
        # Use underscore format for node IDs (Mermaid identifiers)
        # but the display label will use the formatted version with slashes
        return utils.sanitize_mermaid_id(key.to_string()).lower()

    def _build_feature_label_with_fields(
        self, feature_key: FeatureKey, feature_cls
    ) -> str:
        """Build label for feature node with fields displayed inside.

        Args:
            feature_key: Feature key
            feature_cls: Feature class

        Returns:
            Formatted label with feature info and fields as sub-items
        """
        lines = []

        # Feature key (bold)
        feature_name = self._format_feature_key(feature_key)
        lines.append(f"<b>{feature_name}</b>")

        # Feature version info
        if self.config.show_feature_versions or self.config.show_code_versions:
            version_parts = []

            if self.config.show_feature_versions:
                version = self._format_hash(feature_cls.feature_version())
                version_parts.append(f"v: {version}")

            if self.config.show_code_versions:
                version_parts.append(f"cv: {feature_cls.spec.code_version}")

            lines.append(f"<small>({', '.join(version_parts)})</small>")

        # Fields (if configured)
        if self.config.show_fields and feature_cls.spec.fields:
            # Subtle separator line before fields
            lines.append('<font color="#999">---</font>')
            for field in feature_cls.spec.fields:
                field_line = self._build_field_line(feature_key, field)
                lines.append(field_line)

        # Wrap content in a div with left alignment
        content = "<br/>".join(lines)
        return f'<div style="text-align:left">{content}</div>'

    def _build_field_line(self, feature_key: FeatureKey, field_spec) -> str:
        """Build single line for field display.

        Args:
            feature_key: Feature key (for version lookup)
            field_spec: Field specification

        Returns:
            Formatted field line
        """
        parts = [f"• {self._format_field_key(field_spec.key)}"]

        if self.config.show_field_versions or self.config.show_code_versions:
            version_parts = []

            if self.config.show_field_versions:
                fq_key = FQFieldKey(feature=feature_key, field=field_spec.key)
                version = self._format_hash(self.graph.get_field_version(fq_key))
                version_parts.append(f"v: {version}")

            if self.config.show_code_versions:
                version_parts.append(f"cv: {field_spec.code_version}")

            parts.append(f"<small>({', '.join(version_parts)})</small>")

        return " ".join(parts)
