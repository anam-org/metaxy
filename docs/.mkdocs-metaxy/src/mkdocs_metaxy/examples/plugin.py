"""MkDocs plugin for Metaxy examples.

This plugin provides markdown directives for displaying example scenarios, files,
and patches from the Metaxy examples directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from mkdocs.config import Config, config_options
from mkdocs.plugins import BasePlugin

from mkdocs_metaxy.examples.core import RunbookLoader
from mkdocs_metaxy.examples.renderer import ExampleRenderer


def _write_if_changed(path: Path, content: str) -> bool:
    """Write content to file only if it differs from existing content.

    This prevents unnecessary file modifications that would trigger
    the MkDocs file watcher and cause rebuild loops.

    Returns True if file was written, False if unchanged.
    """
    if path.exists():
        existing = path.read_text()
        if existing == content:
            return False
    path.write_text(content)
    return True


class MetaxyExamplesPluginConfig(Config):
    """Configuration for MetaxyExamplesPlugin."""

    examples_dir = config_options.Type(str, default="../examples")


class MetaxyExamplesPlugin(BasePlugin[MetaxyExamplesPluginConfig]):
    """MkDocs plugin for rendering Metaxy example content.

    This plugin registers a markdown extension that provides directives for
    displaying example scenarios, source files at different stages, and patches.

    Configuration:
        examples_dir: Path to examples directory relative to docs/ (default: "../examples")

    Example mkdocs.yml:
        plugins:
          - metaxy-examples:
              examples_dir: "../examples"
    """

    def __init__(self) -> None:
        """Initialize the plugin."""
        super().__init__()
        self.loader: RunbookLoader | None = None
        self.renderer: ExampleRenderer | None = None
        self._patch_enumerations: dict[str, dict[str, int]] = {}
        self.generated_dir: Path | None = None

    def on_config(self, config: Any) -> Any:
        """Hook to modify MkDocs configuration.

        Args:
            config: MkDocs configuration object.

        Returns:
            Modified configuration object.
        """
        # Resolve examples_dir relative to docs_dir
        docs_dir = Path(config["docs_dir"])
        examples_dir = docs_dir / self.config["examples_dir"]
        examples_dir = examples_dir.resolve()

        if not examples_dir.exists():
            raise ValueError(
                f"Examples directory not found: {examples_dir} "
                f"(resolved from docs_dir={docs_dir}, "
                f"examples_dir={self.config['examples_dir']})"
            )

        # Store resolved path for use by hooks
        self._examples_dir = examples_dir

        # Initialize loader and renderer
        self.loader = RunbookLoader(examples_dir)
        self.renderer = ExampleRenderer()

        # Create directory for generated files (patched versions)
        self.generated_dir = examples_dir / ".generated"
        assert self.generated_dir is not None  # For type checker
        self.generated_dir.mkdir(exist_ok=True)

        # Remove fenced_code if present (conflicts with pymdownx.superfences)
        extensions = config.setdefault("markdown_extensions", [])
        if "fenced_code" in extensions:
            extensions.remove("fenced_code")

        return config

    def on_page_markdown(self, markdown: str, **kwargs: Any) -> str:
        """Process markdown before it's converted to HTML.

        Args:
            markdown: Page markdown content.
            **kwargs: Additional arguments (page, config, files).

        Returns:
            Modified markdown content.
        """
        if self.loader is None or self.renderer is None or self.generated_dir is None:
            return markdown

        # Pattern to match directive opening line (allow hyphens in directive type)
        directive_pattern = re.compile(
            r"^:::\s+metaxy-example\s+([\w-]+)\s*$",
            re.MULTILINE,
        )

        result_parts: list[str] = []
        pos = 0

        for match in directive_pattern.finditer(markdown):
            # These are guaranteed to be non-None due to the check above
            assert self.loader is not None
            assert self.renderer is not None

            result_parts.append(markdown[pos : match.start()])

            directive_type = match.group(1)

            # Collect indented content lines after the opening directive
            remaining = markdown[match.end() :]
            content_lines: list[str] = []
            end_pos = match.end()
            for line in remaining.split("\n")[1:]:  # skip first empty split
                if line.strip() == "":
                    content_lines.append(line)
                elif line.startswith("    "):
                    content_lines.append(line)
                else:
                    break
                end_pos += len(line) + 1  # +1 for newline
            end_pos += 1  # account for newline after opening line

            # Strip trailing empty lines and un-consume them so they remain
            # as separators between the replacement and subsequent content
            while content_lines and not content_lines[-1].strip():
                end_pos -= len(content_lines[-1]) + 1
                content_lines.pop()

            # Dedent the content
            non_empty = [line for line in content_lines if line.strip()]
            if non_empty:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty)
                directive_content = "\n".join(
                    line[min_indent:] if len(line) >= min_indent else line for line in content_lines
                ).strip()
            else:
                directive_content = ""

            params = yaml.safe_load(directive_content) or {}
            example_name = params.get("example")
            if not example_name:
                raise ValueError(f"Missing required 'example' parameter in {directive_type} directive")

            # Process based on directive type
            if directive_type == "scenarios":
                scenarios = self.loader.get_scenarios(example_name)
                html = self.renderer.render_scenarios(scenarios, example_name)
            elif directive_type == "source-link" or directive_type == "github":
                button_style = params.get("button", True)
                text = params.get("text", None)
                html = self.renderer.render_source_link(example_name, button_style=button_style, text=text)
            elif directive_type == "file":
                html = self._render_file(example_name, params)
            elif directive_type == "patch":
                html = self._render_patch(example_name, params)
            elif directive_type == "patch-with-diff":
                html = self._render_patch_with_diff(example_name, params)
            elif directive_type == "output":
                html = self._render_output(example_name, params)
            elif directive_type == "graph":
                html = self._render_graph(example_name, params)
            elif directive_type == "graph-diff":
                html = self._render_graph_diff(example_name, params)
            else:
                raise ValueError(f"Unknown directive type: {directive_type}")

            result_parts.append(html)
            pos = end_pos

        result_parts.append(markdown[pos:])
        return "".join(result_parts)

    def _get_patch_enumeration(self, example_name: str) -> dict[str, int]:
        """Get patch enumeration for an example."""
        if example_name in self._patch_enumerations:
            return self._patch_enumerations[example_name]

        if self.loader is None:
            return {}

        scenarios = self.loader.get_scenarios(example_name)
        patch_paths = []

        for scenario in scenarios:
            steps = scenario.get("steps", [])
            for step in steps:
                if step.get("type") == "apply_patch":
                    patch_path = step.get("patch_path")
                    if patch_path and patch_path not in patch_paths:
                        patch_paths.append(patch_path)

        enumeration = {patch: idx + 2 for idx, patch in enumerate(patch_paths)}
        self._patch_enumerations[example_name] = enumeration
        return enumeration

    def _extract_changed_lines_from_patches(
        self, example_name: str, patch_paths: list[str], patched_content: str
    ) -> list[int]:
        """Extract line numbers of changed lines from patch files.

        This method parses the patches to extract added/modified lines, then
        finds those lines in the actual patched file content to get accurate
        line numbers (since patch header line numbers may not match the actual file).

        Args:
            example_name: Name of the example.
            patch_paths: List of patch file paths to parse.
            patched_content: The actual patched file content.

        Returns:
            List of line numbers (1-indexed) that were added or modified.
        """
        if self.loader is None:
            return []

        # Extract the content of added/changed lines from patches
        added_line_contents = []

        for patch_path in patch_paths:
            patch_content = self.loader.read_patch(example_name, patch_path)
            lines = patch_content.split("\n")

            for line in lines:
                # Lines starting with '+' (but not '+++') are additions/changes
                if line.startswith("+") and not line.startswith("+++"):
                    # Remove the '+' prefix to get the actual line content
                    added_line_contents.append(line[1:])

        # Now find these lines in the actual patched content
        patched_lines = patched_content.split("\n")
        changed_lines = []

        for line_num, patched_line in enumerate(patched_lines, start=1):
            for added_content in added_line_contents:
                if patched_line == added_content:
                    changed_lines.append(line_num)
                    break  # Don't count the same line multiple times

        return sorted(set(changed_lines))  # Remove duplicates and sort

    def _render_file(self, example_name: str, params: dict[str, Any]) -> str:
        """Render a source file."""
        if self.loader is None or self.renderer is None or self.generated_dir is None:
            return ""

        file_path = params.get("path")
        if not file_path:
            return self.renderer.render_error("Missing required parameter: path")

        patches = params.get("patches", [])
        show_linenos = params.get("linenos", True)

        example_name_full = example_name if example_name.startswith("example-") else f"example-{example_name}"

        hl_lines = None
        snippets_path = f"{example_name_full}/{file_path}"

        if patches:
            patch_enum = self._get_patch_enumeration(example_name)
            version_num = max(patch_enum.get(p, 1) for p in patches)

            # Apply patches to get patched content
            content = self.loader.read_file(example_name, file_path, patches)

            # Extract changed line numbers by comparing patch additions with actual content
            hl_lines = self._extract_changed_lines_from_patches(example_name, patches, content)

            # Write patched file to generated directory, preserving directory structure
            file_path_obj = Path(file_path)
            # Create versioned filename: dir/file_v2.ext
            if file_path_obj.parent != Path("."):
                # Preserve directory structure
                versioned_name = f"{file_path_obj.stem}_v{version_num}{file_path_obj.suffix}"
                versioned_path = file_path_obj.parent / versioned_name
                generated_file = self.generated_dir / versioned_path
                generated_file.parent.mkdir(parents=True, exist_ok=True)
                snippets_path = f".generated/{versioned_path}"
            else:
                # No directory structure
                versioned_name = f"{file_path_obj.stem}_v{version_num}{file_path_obj.suffix}"
                generated_file = self.generated_dir / versioned_name
                snippets_path = f".generated/{versioned_name}"

            _write_if_changed(generated_file, content)

        return self.renderer.render_snippet(
            path=snippets_path,
            show_line_numbers=show_linenos,
            hl_lines=hl_lines,
        )

    def _render_patch(self, example_name: str, params: dict[str, Any]) -> str:
        """Render a patch file, optionally with graph diff in tabs."""
        if self.loader is None or self.renderer is None or self.generated_dir is None:
            return ""

        patch_path = params.get("path")
        if not patch_path:
            return self.renderer.render_error("Missing required parameter: path")

        # Load patch snapshots cache if available
        patch_snapshots = self.loader.get_patch_snapshots(example_name)
        snapshots = patch_snapshots.get(patch_path)

        content = self.loader.read_patch(example_name, patch_path)

        # Preserve the full path structure for patches
        generated_file = self.generated_dir / patch_path
        generated_file.parent.mkdir(parents=True, exist_ok=True)
        _write_if_changed(generated_file, content)

        snippets_path = f".generated/{patch_path}"

        # Check if we have graph diff snapshots and they are different
        if snapshots and snapshots[0] and snapshots[1] and snapshots[0] != snapshots[1]:
            # Create tabbed output with patch and graph diff
            example_dir = self.loader.get_example_dir(example_name)
            graph_diff = self.renderer.render_graph_diff(snapshots[0], snapshots[1], example_name, example_dir)

            if graph_diff:
                # For tabs, render the patch without collapsible wrapper
                patch_render = self.renderer.render_snippet(
                    path=snippets_path,
                    show_line_numbers=True,
                    hl_lines=None,
                    collapsible=False,
                )
                # Indent the content for proper tab rendering
                patch_lines = patch_render.strip().split("\n")
                patch_indented = "\n".join(f"    {line}" if line else "" for line in patch_lines)

                graph_lines = graph_diff.strip().split("\n")
                graph_indented = "\n".join(f"    {line}" if line else "" for line in graph_lines)

                return f"""
=== "Patch"

{patch_indented}

=== "Feature Graph Changes"

{graph_indented}
"""

        # No graph diff available, return the patch with collapsible wrapper
        return self.renderer.render_snippet(
            path=snippets_path,
            show_line_numbers=True,
            hl_lines=None,
            collapsible=True,
        )

    def _render_patch_with_diff(self, example_name: str, params: dict[str, Any]) -> str:
        """Render patch and graph diff side-by-side in content tabs.

        Args:
            example_name: Name of the example.
            params: Directive parameters (path, scenario, step).

        Returns:
            Markdown string with tabbed content.
        """
        from metaxy_testing import PatchApplied

        if self.loader is None or self.renderer is None or self.generated_dir is None:
            return ""

        patch_path = params.get("path")
        if not patch_path:
            raise ValueError("Missing required parameter for patch-with-diff directive: path")

        scenario_name = params.get("scenario")
        step_name = params.get("step")

        # Write the patch file to generated directory
        content = self.loader.read_patch(example_name, patch_path)
        generated_file = self.generated_dir / patch_path
        generated_file.parent.mkdir(parents=True, exist_ok=True)
        _write_if_changed(generated_file, content)
        snippets_path = f".generated/{patch_path}"

        # Render the patch content (without collapsible wrapper for tabs)
        patch_md = self.renderer.render_snippet(
            path=snippets_path,
            show_line_numbers=True,
            hl_lines=None,
            collapsible=False,
        )

        # Get graph diff from execution result
        result = self.loader.load_execution_result(example_name)

        # Find the PatchApplied event
        matching_event = None
        for event in result.execution_state.events:
            if not isinstance(event, PatchApplied):
                continue
            if scenario_name and event.scenario_name != scenario_name:
                continue
            if step_name and event.step_name != step_name:
                continue
            if event.patch_path == patch_path:
                matching_event = event
                break

        if matching_event is None:
            raise ValueError(
                f"No PatchApplied event found for patch '{patch_path}' (scenario={scenario_name}, step={step_name})"
            )

        if not matching_event.before_graph or not matching_event.after_graph:
            raise ValueError(
                f"PatchApplied event for '{patch_path}' is missing graph data. "
                f"Re-run the example tests to regenerate the execution result."
            )

        graph_diff_md = self._render_graph_diff_from_snapshots(matching_event.before_graph, matching_event.after_graph)

        # Build tabbed content using pymdownx.tabbed syntax
        patch_lines = patch_md.strip().split("\n")
        patch_indented = "\n".join(f"    {line}" if line else "" for line in patch_lines)

        graph_lines = graph_diff_md.strip().split("\n")
        graph_indented = "\n".join(f"    {line}" if line else "" for line in graph_lines)

        return f"""
=== "Patch"

{patch_indented}

=== "Feature Graph Changes"

{graph_indented}
"""

    def _render_output(self, example_name: str, params: dict[str, Any]) -> str:
        """Render execution output from saved result.

        Args:
            example_name: Name of the example.
            params: Directive parameters (scenario, step, etc.).

        Returns:
            Markdown string with execution output.
        """
        from metaxy_testing import CommandExecuted

        if self.loader is None or self.renderer is None:
            return ""

        # Load execution result (raises FileNotFoundError if missing)
        result = self.loader.load_execution_result(example_name)

        # Filter events by scenario if specified
        scenario_name = params.get("scenario")
        events = list(result.execution_state.events)
        if scenario_name:
            events = [e for e in events if e.scenario_name == scenario_name]

        # Filter by step name if specified
        step_name = params.get("step")
        if step_name:
            events = [e for e in events if e.step_name == step_name]

        # Render only command output events
        # Use graph/graph-diff/patch/patch-with-diff directives for other content
        md_parts = []
        for event in events:
            if isinstance(event, CommandExecuted):
                show_command = params.get("show_command", True)
                md_parts.append(self.renderer.render_command_output(event, show_command))

        if not md_parts:
            raise ValueError(f"No events matched the specified criteria: scenario={scenario_name}, step={step_name}")

        return "\n".join(md_parts)

    def _render_graph_diff_from_snapshots(self, before_graph: dict, after_graph: dict, direction: str = "TB") -> str:
        """Render graph diff as mermaid from saved graph snapshots.

        Args:
            before_graph: Serialized graph before patch.
            after_graph: Serialized graph after patch.
            direction: Graph layout direction ("TB" for top-bottom, "LR" for left-right).

        Returns:
            Markdown string with mermaid code block.
        """
        from metaxy.graph import MermaidRenderer
        from metaxy.graph.diff import GraphDiffer
        from metaxy.graph.diff.models import GraphData
        from metaxy.graph.diff.rendering.base import RenderConfig

        # Compute diff using snapshot data format
        differ = GraphDiffer()
        diff = differ.diff(before_graph, after_graph)

        # Create merged graph data for rendering
        merged_data = differ.create_merged_graph_data(before_graph, after_graph, diff)
        graph_data = GraphData.from_merged_diff(merged_data)

        # Create config with direction
        config = RenderConfig(direction=direction)

        # Render as mermaid
        renderer = MermaidRenderer(graph_data=graph_data, config=config)
        mermaid = renderer.render()

        return f"```mermaid\n{mermaid}\n```"

    def _render_graph(self, example_name: str, params: dict[str, Any]) -> str:
        """Render feature graph from GraphPushed event as mermaid diagram.

        Args:
            example_name: Name of the example.
            params: Directive parameters (scenario, step, direction, show_field_deps, features).

        Returns:
            Mermaid diagram markdown string.
        """
        from metaxy_testing import GraphPushed

        if self.loader is None:
            return ""

        result = self.loader.load_execution_result(example_name)

        scenario_name = params.get("scenario")
        step_name = params.get("step")
        direction = params.get("direction", "TB")
        show_field_deps = params.get("show_field_deps", False)
        features = params.get("features")

        # Find the matching GraphPushed event
        for event in result.execution_state.events:
            if not isinstance(event, GraphPushed):
                continue
            if scenario_name and event.scenario_name != scenario_name:
                continue
            if step_name and event.step_name != step_name:
                continue

            # Found matching event - render graph from stored data
            if not event.graph:
                raise ValueError(
                    "GraphPushed event is missing graph data. "
                    "Re-run the example tests to regenerate the execution result."
                )

            if show_field_deps:
                return self._render_field_deps_from_snapshot(event.graph, features=features, direction=direction)

            return self._render_graph_from_snapshot(event.graph, direction=direction)

        raise ValueError(f"No GraphPushed event found for scenario={scenario_name}, step={step_name}")

    def _render_graph_from_snapshot(self, graph_data: dict, direction: str = "TB") -> str:
        """Render graph as mermaid from saved graph snapshot.

        Args:
            graph_data: Serialized graph data.
            direction: Graph layout direction ("TB" for top-bottom, "LR" for left-right).

        Returns:
            Markdown string with mermaid code block.
        """
        from metaxy.graph import MermaidRenderer
        from metaxy.graph.diff.models import GraphData
        from metaxy.graph.diff.rendering.base import RenderConfig

        # Create graph data for rendering (no diff, just the graph)
        gd = GraphData.from_snapshot(graph_data)

        # Create config with direction
        config = RenderConfig(direction=direction)

        # Render as mermaid
        renderer = MermaidRenderer(graph_data=gd, config=config)
        mermaid = renderer.render()

        return f"```mermaid\n{mermaid}\n```"

    def _render_field_deps_from_snapshot(
        self,
        graph_data: dict,
        features: list[str] | None = None,
        direction: str = "TB",
    ) -> str:
        """Render field-level dependency graph as mermaid from saved graph snapshot.

        Args:
            graph_data: Serialized graph data.
            features: Optional list of feature key strings to filter to.
            direction: Graph layout direction.

        Returns:
            Markdown string with mermaid code block.
        """
        from metaxy.graph.diff.rendering.field_deps import render_field_deps_mermaid

        mermaid = render_field_deps_mermaid(graph_data, features=features, direction=direction)
        return f"```mermaid\n{mermaid}\n```"

    def _render_graph_diff(self, example_name: str, params: dict[str, Any]) -> str:
        """Render graph diff from PatchApplied event as mermaid diagram.

        Args:
            example_name: Name of the example.
            params: Directive parameters (scenario, step, direction).

        Returns:
            Mermaid diagram markdown string.
        """
        from metaxy_testing import PatchApplied

        if self.loader is None:
            return ""

        result = self.loader.load_execution_result(example_name)

        scenario_name = params.get("scenario")
        step_name = params.get("step")
        direction = params.get("direction", "TB")

        # Find the matching PatchApplied event
        for event in result.execution_state.events:
            if not isinstance(event, PatchApplied):
                continue
            if scenario_name and event.scenario_name != scenario_name:
                continue
            if step_name and event.step_name != step_name:
                continue

            # Found matching event - render graph diff
            if not event.before_graph or not event.after_graph:
                raise ValueError(
                    "PatchApplied event is missing graph data. "
                    "Re-run the example tests to regenerate the execution result."
                )

            return self._render_graph_diff_from_snapshots(event.before_graph, event.after_graph, direction=direction)

        raise ValueError(f"No PatchApplied event found for scenario={scenario_name}, step={step_name}")
