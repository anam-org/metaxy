"""Formatter for graph diff output."""

import json
from typing import Any

from rich.console import Console

from metaxy.graph import utils
from metaxy.graph.diff.diff_models import FieldChange, GraphDiff


class DiffFormatter:
    """Formats GraphDiff for display with colored output."""

    def __init__(self, console: Console | None = None):
        """Initialize formatter.

        Args:
            console: Rich console for output (creates new one if None)
        """
        self.console = console or Console()

    def format(
        self,
        diff: GraphDiff | None = None,
        merged_data: dict[str, Any] | None = None,
        format: str = "terminal",
        verbose: bool = False,
        diff_only: bool = False,
        show_all_fields: bool = True,
    ) -> str:
        """Format a GraphDiff or merged graph data in the specified format.

        Args:
            diff: GraphDiff to format (required for diff_only mode)
            merged_data: Merged graph data (required for merged mode)
            format: Output format ("terminal", "json", "yaml", or "mermaid")
            verbose: If True, show more details (dependencies, code versions)
            diff_only: If True, show only diff list; otherwise show merged graph
            show_all_fields: If True, show all fields; if False, show only changed fields

        Returns:
            Formatted string

        Raises:
            ValueError: If format is not recognized or required data is missing
        """
        if diff_only:
            if diff is None:
                raise ValueError("diff is required for diff_only mode")
            return self._format_diff_only(diff, format, verbose)
        else:
            if merged_data is None:
                raise ValueError("merged_data is required for merged mode")
            return self._format_merged(merged_data, format, verbose, show_all_fields)

    def _format_diff_only(self, diff: GraphDiff, format: str, verbose: bool) -> str:
        """Format diff-only output."""
        if format == "terminal":
            return self.format_terminal_diff_only(diff, verbose)
        elif format == "json":
            return self.format_json_diff_only(diff)
        elif format == "yaml":
            return self.format_yaml_diff_only(diff)
        elif format == "mermaid":
            return self.format_mermaid_diff_only(diff, verbose)
        else:
            raise ValueError(
                f"Unknown format: {format}. Must be one of: terminal, json, yaml, mermaid"
            )

    def _format_merged(
        self,
        merged_data: dict[str, Any],
        format: str,
        verbose: bool,
        show_all_fields: bool,
    ) -> str:
        """Format merged graph output."""
        if format == "terminal":
            return self.format_terminal_merged(merged_data, verbose, show_all_fields)
        elif format == "json":
            return self.format_json_merged(merged_data)
        elif format == "yaml":
            return self.format_yaml_merged(merged_data)
        elif format == "mermaid":
            return self.format_mermaid_merged(merged_data, verbose, show_all_fields)
        else:
            raise ValueError(
                f"Unknown format: {format}. Must be one of: terminal, json, yaml, mermaid"
            )

    def _format_field_change_line(self, field_change: FieldChange) -> str:
        """Format a single field change line with appropriate color and version info.

        Args:
            field_change: The field change to format

        Returns:
            Formatted line string with Rich markup
        """
        field_key_str = utils.format_field_key(field_change.field_key)

        if field_change.is_added:
            new_ver = (
                utils.format_hash(field_change.new_version)
                if field_change.new_version
                else "none"
            )
            return f"      [green]+[/green] {field_key_str} ({new_ver}...)"
        elif field_change.is_removed:
            old_ver = (
                utils.format_hash(field_change.old_version)
                if field_change.old_version
                else "none"
            )
            return f"      [red]-[/red] {field_key_str} ({old_ver}...)"
        else:
            # Changed
            old_ver = (
                utils.format_hash(field_change.old_version)
                if field_change.old_version
                else "none"
            )
            new_ver = (
                utils.format_hash(field_change.new_version)
                if field_change.new_version
                else "none"
            )
            return f"      [yellow]~[/yellow] {field_key_str} ({old_ver}... → {new_ver}...)"

    def format_terminal_diff_only(self, diff: GraphDiff, verbose: bool = False) -> str:
        """Format a GraphDiff as a human-readable string with colored markup.

        Args:
            diff: GraphDiff to format
            verbose: If True, show more details (dependencies, code versions)

        Returns:
            Formatted string with Rich markup
        """
        if not diff.has_changes:
            return self._format_no_changes(diff)

        lines = []

        # Header
        lines.append(
            f"Graph Diff: {utils.format_hash(diff.from_snapshot_version)}... → {utils.format_hash(diff.to_snapshot_version)}..."
        )
        lines.append("")

        # Added nodes
        if diff.added_nodes:
            lines.append(f"[bold green]Added ({len(diff.added_nodes)}):[/bold green]")
            for node in diff.added_nodes:
                lines.append(
                    f"  [green]+[/green] {utils.format_feature_key(node.feature_key)}"
                )
            lines.append("")

        # Removed nodes
        if diff.removed_nodes:
            lines.append(f"[bold red]Removed ({len(diff.removed_nodes)}):[/bold red]")
            for node in diff.removed_nodes:
                lines.append(
                    f"  [red]-[/red] {utils.format_feature_key(node.feature_key)}"
                )
            lines.append("")

        # Changed nodes
        if diff.changed_nodes:
            lines.append(
                f"[bold yellow]Changed ({len(diff.changed_nodes)}):[/bold yellow]"
            )
            for node_change in diff.changed_nodes:
                # Show feature-level change
                old_ver = (
                    utils.format_hash(node_change.old_version)
                    if node_change.old_version
                    else "none"
                )
                new_ver = (
                    utils.format_hash(node_change.new_version)
                    if node_change.new_version
                    else "none"
                )
                lines.append(
                    f"  [yellow]~[/yellow] {utils.format_feature_key(node_change.feature_key)} "
                    f"({old_ver}... → {new_ver}...)"
                )

                # Show field changes if any
                all_field_changes = (
                    node_change.added_fields
                    + node_change.removed_fields
                    + node_change.changed_fields
                )
                if all_field_changes:
                    lines.append("    fields:")
                    for field_change in all_field_changes:
                        lines.append(self._format_field_change_line(field_change))
            lines.append("")

        # Summary
        total_changes = (
            len(diff.added_nodes) + len(diff.removed_nodes) + len(diff.changed_nodes)
        )
        lines.append(f"[dim]Total changes: {total_changes}[/dim]")

        return "\n".join(lines)

    def _format_no_changes(self, diff: GraphDiff) -> str:
        """Format message when there are no changes."""
        return (
            f"[green]No changes between snapshots[/green]\n"
            f"  {utils.format_hash(diff.from_snapshot_version)}... → {utils.format_hash(diff.to_snapshot_version)}..."
        )

    def print(self, diff: GraphDiff, verbose: bool = False) -> None:
        """Print formatted diff to console.

        Args:
            diff: GraphDiff to print
            verbose: If True, show more details
        """
        formatted = self.format_terminal_diff_only(diff, verbose=verbose)
        self.console.print(formatted)

    def format_json_diff_only(self, diff: GraphDiff) -> str:
        """Format GraphDiff as JSON.

        Args:
            diff: GraphDiff to format

        Returns:
            JSON string representation of the diff
        """
        data = {
            "from_snapshot_version": diff.from_snapshot_version,
            "to_snapshot_version": diff.to_snapshot_version,
            "added_nodes": [
                utils.format_feature_key(node.feature_key) for node in diff.added_nodes
            ],
            "removed_nodes": [
                utils.format_feature_key(node.feature_key)
                for node in diff.removed_nodes
            ],
            "changed_nodes": [
                {
                    "feature_key": utils.format_feature_key(nc.feature_key),
                    "old_version": nc.old_version,
                    "new_version": nc.new_version,
                    "field_changes": [
                        {
                            "field_key": utils.format_field_key(field.field_key),
                            "old_version": field.old_version,
                            "new_version": field.new_version,
                            "is_added": field.is_added,
                            "is_removed": field.is_removed,
                            "is_changed": field.is_changed,
                        }
                        for field in (
                            nc.added_fields + nc.removed_fields + nc.changed_fields
                        )
                    ],
                }
                for nc in diff.changed_nodes
            ],
        }
        return json.dumps(data, indent=2)

    def format_yaml_diff_only(self, diff: GraphDiff) -> str:
        """Format GraphDiff as YAML.

        Args:
            diff: GraphDiff to format

        Returns:
            YAML string representation of the diff
        """
        import yaml

        data = {
            "from_snapshot_version": diff.from_snapshot_version,
            "to_snapshot_version": diff.to_snapshot_version,
            "added_nodes": [node.feature_key.to_string() for node in diff.added_nodes],
            "removed_nodes": [
                node.feature_key.to_string() for node in diff.removed_nodes
            ],
            "changed_nodes": [
                {
                    "feature_key": nc.feature_key.to_string(),
                    "old_version": nc.old_version,
                    "new_version": nc.new_version,
                    "field_changes": [
                        {
                            "field_key": field.field_key.to_string(),
                            "old_version": field.old_version,
                            "new_version": field.new_version,
                            "is_added": field.is_added,
                            "is_removed": field.is_removed,
                            "is_changed": field.is_changed,
                        }
                        for field in (
                            nc.added_fields + nc.removed_fields + nc.changed_fields
                        )
                    ],
                }
                for nc in diff.changed_nodes
            ],
        }
        # Use width=999999 to prevent line wrapping for long hashes
        return yaml.safe_dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            width=999999,
            allow_unicode=True,
        )

    def format_mermaid_diff_only(self, diff: GraphDiff, verbose: bool = False) -> str:
        """Format GraphDiff as Mermaid flowchart.

        Args:
            diff: GraphDiff to format
            verbose: If True, show more details

        Returns:
            Mermaid flowchart markup showing the diff
        """
        lines = []
        lines.append("---")
        lines.append("title: Feature Graph Changes")
        lines.append("---")
        lines.append("flowchart TB")
        lines.append(
            "    %%{init: {'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '14px'}}}%%"
        )
        lines.append("")

        # Collect all features
        all_features = set()
        for node in diff.added_nodes:
            all_features.add(node.feature_key.to_string())
        for node in diff.removed_nodes:
            all_features.add(node.feature_key.to_string())
        for nc in diff.changed_nodes:
            all_features.add(nc.feature_key.to_string())

        if not all_features:
            lines.append("    Empty[No changes]")
            lines.append("")
            return "\n".join(lines)

        # Generate node IDs (sanitized for Mermaid)
        def sanitize_id(s: str) -> str:
            return s.replace("/", "_").replace("-", "_")

        # Define nodes with styling (border only, no fill)
        for node in diff.added_nodes:
            node_id = sanitize_id(node.feature_key.to_string())
            feature_str = node.feature_key.to_string()
            lines.append(f'    {node_id}["{feature_str}"]')
            lines.append(f"    style {node_id} stroke:#00FF00,stroke-width:2px")

        for node in diff.removed_nodes:
            node_id = sanitize_id(node.feature_key.to_string())
            feature_str = node.feature_key.to_string()
            lines.append(f'    {node_id}["{feature_str}"]')
            lines.append(f"    style {node_id} stroke:#FF0000,stroke-width:2px")

        for nc in diff.changed_nodes:
            node_id = sanitize_id(nc.feature_key.to_string())
            feature_str = nc.feature_key.to_string()

            all_field_changes = nc.added_fields + nc.removed_fields + nc.changed_fields
            if verbose and all_field_changes:
                # Show field changes in verbose mode
                field_changes_str = "<br/>".join(
                    [
                        f"{'+ ' if field.is_added else '- ' if field.is_removed else '~ '}{field.field_key.to_string()}"
                        for field in all_field_changes
                    ]
                )
                lines.append(f'    {node_id}["{feature_str}<br/>{field_changes_str}"]')
            else:
                lines.append(f'    {node_id}["{feature_str}"]')

            lines.append(f"    style {node_id} stroke:#FFAA00,stroke-width:2px")

        lines.append("")

        return "\n".join(lines)

    # Merged graph format methods

    def _get_status_symbol_and_text(self, status: str) -> tuple[str, str]:
        """Get the symbol and text for a given status.

        Args:
            status: The node status (added, removed, changed, unchanged)

        Returns:
            Tuple of (symbol, status_text) with Rich markup
        """
        status_map = {
            "added": ("[green]+[/green]", "[green](added)[/green]"),
            "removed": ("[red]-[/red]", "[red](removed)[/red]"),
            "changed": ("[yellow]~[/yellow]", "[yellow](changed)[/yellow]"),
        }
        return status_map.get(status, (" ", ""))

    def _format_version_line(
        self, status: str, old_version: str | None, new_version: str | None
    ) -> str:
        """Format the version line based on status.

        Args:
            status: The node status
            old_version: The old version hash
            new_version: The new version hash

        Returns:
            Formatted version line
        """
        if status == "added":
            ver = utils.format_hash(new_version) if new_version else "none"
            return f"  version: {ver}..."
        elif status == "removed":
            ver = utils.format_hash(old_version) if old_version else "none"
            return f"  version: {ver}..."
        elif status == "changed":
            old_ver_str = utils.format_hash(old_version) if old_version else "none"
            new_ver_str = utils.format_hash(new_version) if new_version else "none"
            return f"  version: {old_ver_str}... → {new_ver_str}..."
        else:
            # Unchanged
            ver = utils.format_hash(new_version) if new_version else "none"
            return f"  version: {ver}..."

    def _format_merged_field_line(
        self,
        field_key_str: str,
        field_change: FieldChange | None,
        field_version: str | None,
    ) -> str:
        """Format a single field line for merged output.

        Args:
            field_key_str: The field key string
            field_change: The field change if any, None for unchanged fields
            field_version: The field version for unchanged fields

        Returns:
            Formatted field line with Rich markup
        """
        if field_change is None:
            # Unchanged field
            ver = utils.format_hash(field_version) if field_version else "none"
            return f"      {field_key_str} ({ver}...)"

        if field_change.is_added:
            new_ver = (
                utils.format_hash(field_change.new_version)
                if field_change.new_version
                else "none"
            )
            return f"    [green]+[/green] {field_key_str} ({new_ver}...)"
        elif field_change.is_removed:
            old_ver = (
                utils.format_hash(field_change.old_version)
                if field_change.old_version
                else "none"
            )
            return f"    [red]-[/red] {field_key_str} ({old_ver}...)"
        else:
            # Changed
            old_ver = (
                utils.format_hash(field_change.old_version)
                if field_change.old_version
                else "none"
            )
            new_ver = (
                utils.format_hash(field_change.new_version)
                if field_change.new_version
                else "none"
            )
            return (
                f"    [yellow]~[/yellow] {field_key_str} "
                f"([red]{old_ver}[/red]... → [green]{new_ver}[/green]...)"
            )

    def _format_merged_node_fields(
        self,
        fields: dict[str, str | None],
        field_changes: list[FieldChange],
        show_all_fields: bool,
    ) -> list[str]:
        """Format the fields section for a merged node.

        Args:
            fields: Dictionary of field key to version
            field_changes: List of field changes
            show_all_fields: Whether to show all fields or only changed ones

        Returns:
            List of formatted field lines
        """
        lines: list[str] = []
        lines.append("  fields:")

        # Build a map of field changes for quick lookup
        field_change_map = {
            fc.field_key.to_string(): fc
            for fc in field_changes
            if isinstance(fc, FieldChange)
        }

        # Collect field keys based on show_all_fields setting
        if show_all_fields:
            all_field_keys = set(fields.keys())
            all_field_keys.update(field_change_map.keys())
        else:
            all_field_keys = set(field_change_map.keys())

        # Show fields
        for field_key_str in sorted(all_field_keys):
            field_change = field_change_map.get(field_key_str)
            field_version = fields.get(field_key_str)
            lines.append(
                self._format_merged_field_line(
                    field_key_str, field_change, field_version
                )
            )

        return lines

    def format_terminal_merged(
        self,
        merged_data: dict[str, Any],
        verbose: bool = False,
        show_all_fields: bool = True,
    ) -> str:
        """Format merged graph as terminal tree view with status annotations.

        Args:
            merged_data: Merged graph data with nodes and edges
            verbose: If True, show more details
            show_all_fields: If True, show all fields; if False, show only changed fields

        Returns:
            Formatted string with Rich markup
        """
        nodes = merged_data["nodes"]

        if not nodes:
            return "[yellow]Empty graph (no features)[/yellow]"

        lines: list[str] = []
        lines.append("[bold]Feature Graph (merged view):[/bold]")
        lines.append("")

        # Sort features by status priority: unchanged, changed, added, removed
        status_order = {"unchanged": 0, "changed": 1, "added": 2, "removed": 3}
        sorted_features = sorted(
            nodes.items(), key=lambda x: (status_order[x[1]["status"]], x[0])
        )

        for feature_key_str, node_data in sorted_features:
            status = node_data["status"]
            symbol, status_text = self._get_status_symbol_and_text(status)

            # Feature line
            lines.append(f"{symbol} [bold]{feature_key_str}[/bold] {status_text}")

            # Version information
            lines.append(
                self._format_version_line(
                    status, node_data["old_version"], node_data["new_version"]
                )
            )

            # Dependencies
            dependencies = node_data["dependencies"]
            if dependencies:
                lines.append(f"  depends on: {', '.join(dependencies)}")

            # Fields
            fields = node_data["fields"]
            if fields:
                lines.extend(
                    self._format_merged_node_fields(
                        fields, node_data["field_changes"], show_all_fields
                    )
                )

            lines.append("")

        return "\n".join(lines)

    def format_json_merged(self, merged_data: dict[str, Any]) -> str:
        """Format merged graph as JSON.

        Args:
            merged_data: Merged graph data with nodes and edges

        Returns:
            JSON string representation of merged graph
        """
        # Convert to JSON-serializable format
        nodes_json = {}
        for feature_key, node_data in merged_data["nodes"].items():
            field_changes_json = []
            for field_change in node_data["field_changes"]:
                if isinstance(field_change, FieldChange):
                    field_changes_json.append(
                        {
                            "field_key": field_change.field_key.to_string(),
                            "old_version": field_change.old_version,
                            "new_version": field_change.new_version,
                            "is_added": field_change.is_added,
                            "is_removed": field_change.is_removed,
                            "is_changed": field_change.is_changed,
                        }
                    )

            nodes_json[feature_key] = {
                "status": node_data["status"],
                "old_version": node_data["old_version"],
                "new_version": node_data["new_version"],
                "dependencies": node_data["dependencies"],
                "field_changes": field_changes_json,
            }

        data = {
            "nodes": nodes_json,
            "edges": merged_data["edges"],
        }
        return json.dumps(data, indent=2)

    def format_yaml_merged(self, merged_data: dict[str, Any]) -> str:
        """Format merged graph as YAML.

        Args:
            merged_data: Merged graph data with nodes and edges

        Returns:
            YAML string representation of merged graph
        """
        import yaml

        # Convert to YAML-serializable format
        nodes_yaml = {}
        for feature_key, node_data in merged_data["nodes"].items():
            field_changes_yaml = []
            for field_change in node_data["field_changes"]:
                if isinstance(field_change, FieldChange):
                    field_changes_yaml.append(
                        {
                            "field_key": field_change.field_key.to_string(),
                            "old_version": field_change.old_version,
                            "new_version": field_change.new_version,
                            "is_added": field_change.is_added,
                            "is_removed": field_change.is_removed,
                            "is_changed": field_change.is_changed,
                        }
                    )

            nodes_yaml[feature_key] = {
                "status": node_data["status"],
                "old_version": node_data["old_version"],
                "new_version": node_data["new_version"],
                "dependencies": node_data["dependencies"],
                "field_changes": field_changes_yaml,
            }

        data = {
            "nodes": nodes_yaml,
            "edges": merged_data["edges"],
        }
        # Use width=999999 to prevent line wrapping for long hashes
        return yaml.safe_dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            width=999999,
            allow_unicode=True,
        )

    def _format_mermaid_version_part(
        self, status: str, old_version: str | None, new_version: str | None
    ) -> str:
        """Format version part for Mermaid node label.

        Args:
            status: The node status
            old_version: The old version hash
            new_version: The new version hash

        Returns:
            Formatted version string for Mermaid label
        """
        if status == "changed":
            old_ver = (
                utils.format_hash(old_version, length=6) if old_version else "none"
            )
            new_ver = (
                utils.format_hash(new_version, length=6) if new_version else "none"
            )
            return (
                f'<font color="#CC0000">{old_ver}</font> → '
                f'<font color="#00AA00">{new_ver}</font>'
            )
        elif status == "removed":
            ver = utils.format_hash(old_version, length=6) if old_version else "none"
            return ver
        else:
            # Added or unchanged
            ver = utils.format_hash(new_version, length=6) if new_version else "none"
            return ver

    def _format_mermaid_field_label(
        self,
        field_key_str: str,
        field_change: FieldChange | None,
        field_version: str | None,
    ) -> str | None:
        """Format a single field for Mermaid node label.

        Args:
            field_key_str: The field key string
            field_change: The field change if any
            field_version: The field version for unchanged fields

        Returns:
            Formatted field label string, or None if field should be skipped
        """
        if field_change is None:
            # Unchanged field
            if field_version:
                ver = utils.format_hash(field_version, length=6)
                return f"- {field_key_str} ({ver})"
            return None

        if field_change.is_added:
            new_ver = (
                utils.format_hash(field_change.new_version, length=6)
                if field_change.new_version
                else "none"
            )
            return f'<font color="#00AA00">- {field_key_str} ({new_ver})</font>'
        elif field_change.is_removed:
            old_ver = (
                utils.format_hash(field_change.old_version, length=6)
                if field_change.old_version
                else "none"
            )
            return f'<font color="#CC0000">- {field_key_str} ({old_ver})</font>'
        else:
            # Changed
            old_ver = (
                utils.format_hash(field_change.old_version, length=6)
                if field_change.old_version
                else "none"
            )
            new_ver = (
                utils.format_hash(field_change.new_version, length=6)
                if field_change.new_version
                else "none"
            )
            return (
                f'- <font color="#FFAA00">{field_key_str}</font> '
                f'(<font color="#CC0000">{old_ver}</font> → '
                f'<font color="#00AA00">{new_ver}</font>)'
            )

    def _get_mermaid_node_style(self, status: str, node_id: str) -> str | None:
        """Get Mermaid style line for a node based on status.

        Args:
            status: The node status
            node_id: The node ID

        Returns:
            Style line string, or None for unchanged nodes
        """
        style_map = {
            "added": f"    style {node_id} stroke:#00FF00,stroke-width:2px",
            "removed": f"    style {node_id} stroke:#FF0000,stroke-width:2px",
            "changed": f"    style {node_id} stroke:#FFAA00,stroke-width:2px",
        }
        return style_map.get(status)

    def _build_mermaid_field_labels(
        self,
        fields: dict[str, str | None],
        field_changes: list[FieldChange],
        status: str,
        show_all_fields: bool,
    ) -> list[str]:
        """Build field label parts for Mermaid node.

        Args:
            fields: Dictionary of field key to version
            field_changes: List of field changes
            status: The node status
            show_all_fields: Whether to show all fields or only changed ones

        Returns:
            List of field label strings
        """
        # Build field change map
        field_change_map = {
            fc.field_key.to_string(): fc
            for fc in field_changes
            if isinstance(fc, FieldChange)
        }

        # Collect field keys based on show_all_fields setting
        if show_all_fields:
            all_field_keys = set(fields.keys())
            all_field_keys.update(field_change_map.keys())
        elif status == "changed" and field_change_map:
            all_field_keys = set(field_change_map.keys())
        else:
            all_field_keys = set()

        label_parts: list[str] = []
        for field_key_str in sorted(all_field_keys):
            field_change = field_change_map.get(field_key_str)
            field_version = fields.get(field_key_str)
            label = self._format_mermaid_field_label(
                field_key_str, field_change, field_version
            )
            if label:
                label_parts.append(label)

        return label_parts

    def format_mermaid_merged(
        self,
        merged_data: dict[str, Any],
        verbose: bool = False,
        show_all_fields: bool = True,
    ) -> str:
        """Format merged graph as Mermaid flowchart with status colors.

        Args:
            merged_data: Merged graph data with nodes and edges
            verbose: If True, show field changes on changed nodes
            show_all_fields: If True, show all fields; if False, show only changed fields

        Returns:
            Mermaid flowchart markup
        """
        nodes = merged_data["nodes"]
        edges = merged_data["edges"]

        lines: list[str] = []
        lines.append("---")
        lines.append("title: Feature Graph Changes")
        lines.append("---")
        lines.append("flowchart TB")
        lines.append(
            "    %%{init: {'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '14px'}}}%%"
        )
        lines.append("")

        if not nodes:
            lines.append("    Empty[No features]")
            return "\n".join(lines)

        def sanitize_id(s: str) -> str:
            return s.replace("/", "_").replace("-", "_")

        # Define nodes with styling based on status
        for feature_key_str, node_data in nodes.items():
            node_id = sanitize_id(feature_key_str)
            status = node_data["status"]
            fields = node_data["fields"]

            # Build node label parts
            label_parts = [f"<b>{feature_key_str}</b>"]

            # Add version info
            label_parts.append(
                self._format_mermaid_version_part(
                    status, node_data["old_version"], node_data["new_version"]
                )
            )

            # Add separator and fields
            if fields:
                label_parts.append('<font color="#999">---</font>')
                label_parts.extend(
                    self._build_mermaid_field_labels(
                        fields, node_data["field_changes"], status, show_all_fields
                    )
                )

            # Wrap content in left-aligned div
            label = "<br/>".join(label_parts)
            label = f'<div style="text-align:left">{label}</div>'
            lines.append(f'    {node_id}["{label}"]')

            # Apply styling based on status
            style_line = self._get_mermaid_node_style(status, node_id)
            if style_line:
                lines.append(style_line)

        lines.append("")

        # Add edges
        for edge in edges:
            from_id = sanitize_id(edge["from"])
            to_id = sanitize_id(edge["to"])
            lines.append(f"    {from_id} --> {to_id}")

        lines.append("")

        return "\n".join(lines)
