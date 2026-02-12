"""Markdown extension for Metaxy examples.

This module provides a markdown preprocessor that handles directives like:
    ::: metaxy-example scenarios
        example: recompute

    ::: metaxy-example file
        example: recompute
        path: src/example_recompute/features.py
        stage: initial

    ::: metaxy-example file
        example: recompute
        path: src/example_recompute/features.py
        patches: ["patches/01_update_parent_algorithm.patch"]

    ::: metaxy-example patch
        example: recompute
        path: patches/01_update_parent_algorithm.patch

    ::: metaxy-example output
        example: recompute
        scenario: "Initial pipeline run"
        step: "run_pipeline"

    ::: metaxy-example patch-with-diff
        example: one-to-many
        path: patches/01_update_video_code_version.patch
        scenario: "Code change - audio field only"
        step: "update_audio_version"

Content is determined by indentation (4 spaces), like MkDocs admonitions.
No closing ::: is needed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from markdown import Markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor

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


class MetaxyExamplesPreprocessor(Preprocessor):
    """Preprocessor for Metaxy example directives."""

    def __init__(
        self,
        md: Markdown | None,
        examples_dir: str,
        docs_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            md: Markdown instance.
            examples_dir: Path to examples directory.
            docs_dir: Path to docs directory (for calculating relative paths).
            **kwargs: Additional arguments.
        """
        super().__init__(md)
        self.examples_dir = Path(examples_dir)
        self.docs_dir = Path(docs_dir) if docs_dir else None
        self.loader = RunbookLoader(self.examples_dir)
        self.renderer = ExampleRenderer()

        # Create directory for generated files (patched versions)
        self.generated_dir = self.examples_dir / ".generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        # Cache for patch enumerations per example
        self._patch_enumerations: dict[str, dict[str, int]] = {}

        # Pattern to match directive opening line:
        # ::: metaxy-example <type>
        #     key: value
        #     ...
        # Content determined by indentation (like MkDocs admonitions)
        self.directive_pattern = re.compile(r"^:::\s+metaxy-example\s+(\w+)\s*$", re.MULTILINE)

    def run(self, lines: list[str]) -> list[str]:
        """Process markdown lines.

        Args:
            lines: List of markdown lines.

        Returns:
            Processed lines with directives replaced by HTML.
        """
        text = "\n".join(lines)
        result_lines: list[str] = []
        pos = 0

        for match in self.directive_pattern.finditer(text):
            directive_type = match.group(1)
            directive_start = match.end()

            # Add text before directive
            result_lines.append(text[pos : match.start()])

            # Collect indented content lines (4+ spaces or empty lines)
            remaining = text[directive_start:]
            content_lines: list[str] = []
            end_pos = directive_start
            for line in remaining.split("\n")[1:]:  # skip the first (empty) split after opening line
                if line.strip() == "":
                    content_lines.append(line)
                elif line.startswith("    "):
                    content_lines.append(line)
                else:
                    break
                end_pos += len(line) + 1  # +1 for the newline

            # Also account for the newline after the opening directive line
            end_pos += 1

            # Strip trailing empty lines and un-consume them so they remain
            # as separators between the replacement and subsequent content
            while content_lines and not content_lines[-1].strip():
                end_pos -= len(content_lines[-1]) + 1
                content_lines.pop()

            # Dedent the content
            directive_content = "\n".join(content_lines)
            non_empty_lines = [line for line in content_lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                directive_content = "\n".join(
                    line[min_indent:] if len(line) >= min_indent else line for line in content_lines
                ).strip()
            else:
                directive_content = ""

            pos = end_pos

            # Process the directive - let exceptions propagate to fail the build
            html = self._process_directive(directive_type, directive_content)
            result_lines.append(html)

        # Add remaining text
        result_lines.append(text[pos:])

        final_text = "".join(result_lines)
        final_lines = final_text.split("\n")
        return final_lines

    def _get_patch_enumeration(self, example_name: str) -> dict[str, int]:
        """Get patch enumeration for an example.

        Args:
            example_name: Name of the example.

        Returns:
            Dictionary mapping patch path to version number (starting at 2 for v2).
        """
        if example_name in self._patch_enumerations:
            return self._patch_enumerations[example_name]

        # Load scenarios and extract all patches in order
        scenarios = self.loader.get_scenarios(example_name)
        patch_paths = []

        for scenario in scenarios:
            steps = scenario.get("steps", [])
            for step in steps:
                if step.get("type") == "apply_patch":
                    patch_path = step.get("patch_path")
                    if patch_path and patch_path not in patch_paths:
                        patch_paths.append(patch_path)

        # Create enumeration: patch -> version number
        # Version starts at 2 (v2) since v1 is the original
        enumeration = {patch: idx + 2 for idx, patch in enumerate(patch_paths)}

        self._patch_enumerations[example_name] = enumeration
        return enumeration

    def _process_directive(self, directive_type: str, content: str) -> str:
        """Process a single directive.

        Args:
            directive_type: Type of directive (scenarios, file, patch).
            content: YAML content of the directive.

        Returns:
            HTML string.

        Raises:
            ValueError: If directive type is unknown or required parameters missing.
        """
        # Parse YAML content
        try:
            params = yaml.safe_load(content) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in directive: {e}") from e

        if not isinstance(params, dict):
            raise ValueError(f"Directive content must be a YAML dictionary, got {type(params)}")

        example_name = params.get("example")
        if not example_name:
            raise ValueError("Missing required parameter: example")

        # Dispatch based on directive type
        if directive_type == "scenarios":
            return self._render_scenarios(example_name)
        if directive_type == "file":
            return self._render_file(example_name, params)
        if directive_type == "patch":
            return self._render_patch(example_name, params)
        if directive_type == "output":
            return self._render_output(example_name, params)
        if directive_type == "patch-with-diff":
            return self._render_patch_with_diff(example_name, params)
        if directive_type == "graph":
            return self._render_graph(example_name, params)
        if directive_type == "graph-diff":
            return self._render_graph_diff(example_name, params)
        raise ValueError(f"Unknown directive type: {directive_type}")

    def _render_scenarios(self, example_name: str) -> str:
        """Render scenarios list.

        Args:
            example_name: Name of the example.

        Returns:
            HTML string.
        """
        scenarios = self.loader.get_scenarios(example_name)
        return self.renderer.render_scenarios(scenarios, example_name)

    def _render_file(self, example_name: str, params: dict[str, Any]) -> str:
        """Render a source file.

        Args:
            example_name: Name of the example.
            params: Directive parameters.

        Returns:
            Markdown string with snippet directive.

        Raises:
            ValueError: If required parameters are missing.
        """
        file_path = params.get("path")
        if not file_path:
            raise ValueError("Missing required parameter for file directive: path")

        # Get optional parameters
        patches = params.get("patches", [])
        show_linenos = params.get("linenos", True)

        # Construct example name with prefix
        example_name_full = example_name if example_name.startswith("example-") else f"example-{example_name}"

        # Calculate highlight lines and snippet path
        hl_lines = None
        snippets_path = f"{example_name_full}/{file_path}"

        if patches:
            # Get patch enumeration for this example
            patch_enum = self._get_patch_enumeration(example_name)

            # Determine version based on the last patch applied
            # Use the highest version number from the patches list
            version_num = max(patch_enum.get(p, 1) for p in patches)

            # Read patched content
            content = self.loader.read_file(example_name, file_path, patches)

            # Read original file to calculate changed lines
            original_content = self.loader.read_file(example_name, file_path, patches=None)

            # Compare line by line to find changes
            original_lines = original_content.split("\n")
            patched_lines = content.split("\n")
            hl_lines = []
            for i, (orig, patched) in enumerate(zip(original_lines, patched_lines), start=1):
                if orig != patched:
                    hl_lines.append(i)

            # Write patched file to .generated directory
            file_base = Path(file_path).stem
            file_ext = Path(file_path).suffix
            filename = f"{file_base}_v{version_num}{file_ext}"
            generated_file = self.generated_dir / filename
            _write_if_changed(generated_file, content)

            # Update snippet path to generated file
            snippets_path = f".generated/{filename}"

        # Use pymdownx.snippets for all files
        return self.renderer.render_snippet(
            path=snippets_path,
            show_line_numbers=show_linenos,
            hl_lines=hl_lines,
        )

    def _render_patch(self, example_name: str, params: dict[str, Any]) -> str:
        """Render a patch file.

        Args:
            example_name: Name of the example.
            params: Directive parameters.

        Returns:
            Markdown string with snippet directive.

        Raises:
            ValueError: If required parameters are missing.
        """
        patch_path = params.get("path")
        if not patch_path:
            raise ValueError("Missing required parameter for patch directive: path")

        # Read patch content
        content = self.loader.read_patch(example_name, patch_path)

        # Write patch to .generated directory
        patch_filename = Path(patch_path).name
        generated_file = self.generated_dir / patch_filename
        _write_if_changed(generated_file, content)

        # Use pymdownx.snippets for patch file
        snippets_path = f".generated/{patch_filename}"
        return self.renderer.render_snippet(
            path=snippets_path,
            show_line_numbers=True,
            hl_lines=None,
        )

    def _render_output(self, example_name: str, params: dict[str, Any]) -> str:
        """Render execution output from saved result.

        Args:
            example_name: Name of the example.
            params: Directive parameters (scenario, step, etc.).

        Returns:
            Markdown string with execution output.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        from metaxy_testing import CommandExecuted, GraphPushed, PatchApplied

        # Load execution result (raises FileNotFoundError if missing)
        result = self.loader.load_execution_result(example_name)

        # Filter events by scenario if specified
        scenario_name = params.get("scenario")
        events = result.execution_state.events
        if scenario_name:
            events = [e for e in events if e.scenario_name == scenario_name]

        # Filter by step name if specified
        step_name = params.get("step")
        if step_name:
            events = [e for e in events if e.step_name == step_name]

        # Render all matching events
        md_parts = []
        for event in events:
            if isinstance(event, CommandExecuted):
                show_command = params.get("show_command", True)
                md_parts.append(self.renderer.render_command_output(event, show_command))
            elif isinstance(event, PatchApplied):
                # Render graph diff from saved graph snapshots
                graph_diff_md = None
                if event.before_graph and event.after_graph:
                    graph_diff_md = self._render_graph_diff_from_snapshots(event.before_graph, event.after_graph)
                md_parts.append(self.renderer.render_patch_applied(event, graph_diff_md))
            elif isinstance(event, GraphPushed):
                md_parts.append(self.renderer.render_graph_pushed(event))

        if not md_parts:
            return self.renderer.render_error(
                "No events matched the specified criteria",
                details=f"scenario={scenario_name}, step={step_name}",
            )

        return "\n".join(md_parts)

    def _render_patch_with_diff(self, example_name: str, params: dict[str, Any]) -> str:
        """Render patch and graph diff side-by-side in content tabs.

        Args:
            example_name: Name of the example.
            params: Directive parameters (path, scenario, step).

        Returns:
            Markdown string with tabbed content.
        """
        from metaxy_testing import PatchApplied

        patch_path = params.get("path")
        if not patch_path:
            raise ValueError("Missing required parameter for patch-with-diff directive: path")

        scenario_name = params.get("scenario")
        step_name = params.get("step")

        # Render the patch content
        patch_md = self._render_patch(example_name, {"path": patch_path})

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
        md_parts = []
        md_parts.append('=== "Patch"')
        md_parts.append("")
        # Indent the patch content for the tab
        for line in patch_md.split("\n"):
            md_parts.append(f"    {line}" if line else "")
        md_parts.append("")

        md_parts.append('=== "Feature Graph Changes"')
        md_parts.append("")
        # Indent the graph diff content for the tab
        for line in graph_diff_md.split("\n"):
            md_parts.append(f"    {line}" if line else "")
        md_parts.append("")

        return "\n".join(md_parts)

    def _render_graph_diff_from_snapshots(self, before_graph: dict, after_graph: dict) -> str:
        """Render graph diff from saved graph snapshots.

        Args:
            before_graph: Serialized graph before the change.
            after_graph: Serialized graph after the change.

        Returns:
            Mermaid diagram string.
        """
        from metaxy.graph import MermaidRenderer
        from metaxy.graph.diff import GraphDiffer
        from metaxy.graph.diff.models import GraphData

        differ = GraphDiffer()
        diff = differ.diff(before_graph, after_graph)

        # Create merged graph data for rendering
        merged_data = differ.create_merged_graph_data(before_graph, after_graph, diff)
        graph_data = GraphData.from_merged_diff(merged_data)

        renderer = MermaidRenderer(graph_data=graph_data)
        mermaid = renderer.render()

        return f"```mermaid\n{mermaid}\n```"

    def _render_graph(self, example_name: str, params: dict[str, Any]) -> str:
        """Render feature graph from command output as mermaid diagram.

        Args:
            example_name: Name of the example.
            params: Directive parameters (scenario, step).

        Returns:
            Mermaid diagram markdown string.

        Raises:
            ValueError: If required parameters are missing.
        """
        from metaxy_testing import CommandExecuted

        result = self.loader.load_execution_result(example_name)

        scenario_name = params.get("scenario")
        step_name = params.get("step")

        # Find the matching command output
        for event in result.execution_state.events:
            if not isinstance(event, CommandExecuted):
                continue
            if scenario_name and event.scenario_name != scenario_name:
                continue
            if step_name and event.step_name != step_name:
                continue

            # Found matching event - return stdout wrapped in mermaid fence
            if event.stdout:
                return f"```mermaid\n{event.stdout.strip()}\n```"

        raise ValueError(f"No CommandExecuted event found for scenario={scenario_name}, step={step_name}")

    def _render_graph_diff(self, example_name: str, params: dict[str, Any]) -> str:
        """Render graph diff from PatchApplied event as mermaid diagram.

        Args:
            example_name: Name of the example.
            params: Directive parameters (scenario, step).

        Returns:
            Mermaid diagram markdown string.

        Raises:
            ValueError: If required parameters are missing or no matching event found.
        """
        from metaxy_testing import PatchApplied

        result = self.loader.load_execution_result(example_name)

        scenario_name = params.get("scenario")
        step_name = params.get("step")

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

            return self._render_graph_diff_from_snapshots(event.before_graph, event.after_graph)

        raise ValueError(f"No PatchApplied event found for scenario={scenario_name}, step={step_name}")


class MetaxyExamplesExtension(Extension):
    """Markdown extension for Metaxy examples."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the extension.

        Args:
            **kwargs: Configuration including 'examples_dir' and 'docs_dir'.
        """
        # Extract docs_dir before calling super().__init__
        # to avoid it being parsed by the config system
        self._docs_dir = kwargs.pop("docs_dir", None)

        self.config = {
            "examples_dir": ["../examples", "Path to examples directory"],
        }
        super().__init__(**kwargs)

    def extendMarkdown(self, md: Markdown) -> None:
        """Register the preprocessor with markdown.

        Args:
            md: Markdown instance.
        """
        examples_dir = self.getConfig("examples_dir")
        preprocessor = MetaxyExamplesPreprocessor(md, examples_dir=examples_dir, docs_dir=self._docs_dir)
        # Register with priority < 32 to run BEFORE pymdownx.snippets (priority 32)
        # This allows snippets to process our generated --8<-- directives
        md.preprocessors.register(preprocessor, "metaxy_examples", 20)


def makeExtension(**kwargs: Any) -> MetaxyExamplesExtension:
    """Create the extension.

    Args:
        **kwargs: Configuration options.

    Returns:
        Extension instance.
    """
    return MetaxyExamplesExtension(**kwargs)
