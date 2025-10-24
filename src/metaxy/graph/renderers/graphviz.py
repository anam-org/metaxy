"""Graphviz renderer for DOT format generation.

Requires pygraphviz library to be installed.
"""

from metaxy.graph.renderers.base import GraphRenderer
from metaxy.models.plan import FQFieldKey


class GraphvizRenderer(GraphRenderer):
    """Renders graph using pygraphviz.

    Creates DOT format output using pygraphviz library.
    Requires pygraphviz to be installed as optional dependency.
    """

    def render(self) -> str:
        """Render graph as Graphviz DOT format.

        Returns:
            DOT format as string
        """
        lines = []

        # Graph header
        rankdir = self.config.direction
        lines.append("strict digraph {")
        lines.append(f"    rankdir={rankdir};")

        # Graph attributes
        if self.config.show_snapshot_id:
            label = f"Graph (snapshot: {self._format_hash(self.graph.snapshot_id)})"
        else:
            label = "Graph"
        lines.append(f'    label="{label}";')
        lines.append("    labelloc=t;")
        lines.append("    fontsize=14;")
        lines.append("    fontname=helvetica;")
        lines.append("")

        # Add nodes for features
        features_to_include = self._get_filtered_features()
        for feature_key in features_to_include:
            if feature_key not in self.graph.features_by_key:
                continue

            feature_cls = self.graph.features_by_key[feature_key]
            node_id = feature_key.to_string()
            label = self._build_feature_label(feature_key, feature_cls)

            # Root features get different shape
            shape = "box" if not self._is_root_feature(feature_key) else "doubleoctagon"

            lines.append(
                f'    "{node_id}" [label="{label}", shape={shape}, '
                f"style=filled, fillcolor=lightblue];"
            )

        lines.append("")

        # Add edges for feature dependencies
        for feature_key in features_to_include:
            if feature_key not in self.graph.features_by_key:
                continue

            feature_cls = self.graph.features_by_key[feature_key]
            if feature_cls.spec.deps:
                target_id = feature_key.to_string()
                for dep in feature_cls.spec.deps:
                    if dep.key in features_to_include:
                        source_id = dep.key.to_string()
                        lines.append(f'    "{source_id}" -> "{target_id}";')

        lines.append("")

        # Add field nodes if configured
        if self.config.show_fields:
            for feature_key in features_to_include:
                if feature_key not in self.graph.features_by_key:
                    continue

                feature_cls = self.graph.features_by_key[feature_key]
                parent_id = feature_key.to_string()

                if not feature_cls.spec.fields:
                    continue

                for field in feature_cls.spec.fields:
                    field_id = f"{parent_id}::{field.key.to_string()}"
                    label = self._build_field_label(feature_key, field)

                    lines.append(
                        f'    "{field_id}" [label="{label}", shape=ellipse, '
                        f"style=filled, fillcolor=lightyellow, fontsize=10];"
                    )

                    # Connect field to feature with dashed line
                    lines.append(
                        f'    "{parent_id}" -> "{field_id}" [style=dashed, arrowhead=none];'
                    )

        lines.append("}")

        return "\n".join(lines)

    def _build_feature_label(self, feature_key, feature_cls) -> str:
        """Build label for feature node.

        Args:
            feature_key: Feature key
            feature_cls: Feature class

        Returns:
            Formatted label with optional version info
        """
        parts = [self._format_feature_key(feature_key)]

        if self.config.show_feature_versions:
            version = self._format_hash(feature_cls.feature_version())
            parts.append(f"\\nv: {version}")

        if self.config.show_code_versions:
            parts.append(f"\\ncv: {feature_cls.spec.code_version}")

        return "".join(parts)

    def _build_field_label(self, feature_key, field_spec) -> str:
        """Build label for field node.

        Args:
            feature_key: Feature key (for version lookup)
            field_spec: Field specification

        Returns:
            Formatted label with optional version info
        """
        parts = [self._format_field_key(field_spec.key)]

        if self.config.show_field_versions:
            fq_key = FQFieldKey(feature=feature_key, field=field_spec.key)
            version = self._format_hash(self.graph.get_field_version(fq_key))
            parts.append(f"\\nv: {version}")

        if self.config.show_code_versions:
            parts.append(f"\\ncv: {field_spec.code_version}")

        return "".join(parts)
