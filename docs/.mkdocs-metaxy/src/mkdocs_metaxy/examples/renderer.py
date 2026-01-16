"""Markdown rendering for Metaxy example content.

This module provides rendering utilities for:
- Scenario lists with descriptions
- Source code in markdown code blocks
- Diff patches in markdown code blocks
- Execution events (commands, patches, graph pushes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from metaxy._testing import CommandExecuted, GraphPushed, PatchApplied


class ExampleRenderer:
    """Renderer for Metaxy example content."""

    def __init__(self) -> None:
        """Initialize the renderer."""
        pass

    def render_scenarios(
        self, scenarios: list[dict[str, Any]], example_name: str
    ) -> str:
        """Render a list of scenarios as markdown.

        Args:
            scenarios: List of scenario dictionaries from runbook.
            example_name: Name of the example for reference.

        Returns:
            Markdown string.
        """
        md_parts = []

        for i, scenario in enumerate(scenarios, 1):
            name = scenario.get("name", f"Scenario {i}")
            description = scenario.get("description", "")

            md_parts.append(f"**{i}. {name}**")
            md_parts.append("")

            if description:
                md_parts.append(f"*{description}*")
                md_parts.append("")

            # List steps (skip assert_output steps - they're for tests only)
            steps = scenario.get("steps", [])
            if steps:
                for step in steps:
                    # Skip assert_output steps in documentation
                    if step.get("type") == "assert_output":
                        continue
                    step_desc = self._format_step(step)
                    md_parts.append(f"- {step_desc}")
                md_parts.append("")

        return "\n".join(md_parts)

    def _format_step(self, step: dict[str, Any]) -> str:
        """Format a step for display.

        Args:
            step: Step dictionary from runbook.

        Returns:
            Markdown string describing the step.
        """
        step_type = step.get("type", "unknown")
        description = step.get("description")

        if step_type == "run_command":
            command = step.get("command", "")
            if description:
                return f"{description}\n```shell\n{command}\n```"
            return f"```shell\n{command}\n```"

        if step_type == "apply_patch":
            patch_path = step.get("patch_path", "")
            patch_command = f"patch -p1 -i {patch_path}"
            if description:
                return f"{description}:\n```shell\n{patch_command}\n```"
            return f"Apply patch:\n```shell\n{patch_command}\n```"

        if step_type == "assert_output":
            if description:
                return f"Assert: {description}"
            return "Validate output"

        return f"Step: {step_type}"

    def render_snippet(
        self,
        path: str,
        show_line_numbers: bool = True,
        hl_lines: list[int] | None = None,
        collapsible: bool = True,
    ) -> str:
        """Render a code snippet using pymdownx.snippets.

        Args:
            path: Path to the file (relative to snippets base_path).
            show_line_numbers: Whether to show line numbers.
            hl_lines: List of line numbers to highlight.
            collapsible: Whether to make the code block collapsible.

        Returns:
            Markdown string with snippet directive.
        """
        lang = self._get_language_from_path(path)
        display_path = self._get_display_path(path)
        fence_attrs = self._build_fence_attrs(
            lang, display_path, show_line_numbers, hl_lines
        )
        return self._build_snippet_result(path, display_path, fence_attrs, collapsible)

    def _get_language_from_path(self, path: str) -> str:
        """Determine syntax highlighting language from file path."""
        if "." not in path:
            return ""
        _, extension = path.rsplit(".", 1)
        if not extension:
            return ""
        return "diff" if extension == "patch" else extension

    def _get_display_path(self, path: str) -> str:
        """Clean up path for display in title."""
        if path.startswith(".generated/"):
            return path[11:]  # Remove .generated/ prefix
        if "/" in path:
            parts_split = path.split("/", 1)
            if parts_split[0].startswith("example-"):
                return parts_split[1]
        return path

    def _build_fence_attrs(
        self,
        lang: str,
        display_path: str,
        show_line_numbers: bool,
        hl_lines: list[int] | None,
    ) -> str:
        """Build the code fence attributes string."""
        parts = []
        if lang:
            parts.append(lang)
        if display_path:
            parts.append(f'title="{display_path}"')
        if show_line_numbers:
            parts.append('linenums="1"')
        if hl_lines:
            hl_lines_str = " ".join(str(line) for line in hl_lines)
            parts.append(f'hl_lines="{hl_lines_str}"')
        return " ".join(parts)

    def _build_snippet_result(
        self, path: str, display_path: str, fence_attrs: str, collapsible: bool
    ) -> str:
        """Build the final snippet result with optional collapsible wrapper."""
        result = "\n\n"
        if collapsible:
            summary_text = f"`{display_path}`"
            result += f'???+ example "{summary_text}"\n\n'
            result += f"    ``` {fence_attrs}\n"
            result += f'    --8<-- "{path}"\n'
            result += "    ```\n"
        else:
            result += f"``` {fence_attrs}\n"
            result += f'--8<-- "{path}"\n'
            result += "```\n"
        result += "\n"
        return result

    def render_source_link(
        self, example_name: str, button_style: bool = True, text: str | None = None
    ) -> str:
        """Render a GitHub source link for an example.

        Args:
            example_name: Name of the example.
            button_style: Whether to render as a button (True) or inline link (False).
            text: Custom text for the link. Defaults to "View Example Source on GitHub".

        Returns:
            Markdown string with GitHub link.
        """
        example_dir = (
            f"example-{example_name}"
            if not example_name.startswith("example-")
            else example_name
        )
        github_url = (
            f"https://github.com/anam-org/metaxy/tree/main/examples/{example_dir}"
        )

        if text is None:
            text = "View Example Source on GitHub"

        if button_style:
            return f"[:octicons-mark-github-16: {text}]({github_url}){{.md-button target=_blank}}\n\n"
        else:
            return (
                f"[:octicons-mark-github-16: {text}]({github_url}){{target=_blank}}\n\n"
            )

    def render_error(self, message: str, details: str | None = None) -> str:
        """Render an error message as markdown.

        Args:
            message: Error message.
            details: Optional detailed error information.

        Returns:
            Markdown string.
        """
        md_parts = [
            "!!! error",
            f"    {message}",
        ]

        if details:
            md_parts.append("")
            md_parts.append("    ```")
            for line in details.split("\n"):
                md_parts.append(f"    {line}")
            md_parts.append("    ```")

        md_parts.append("")
        return "\n".join(md_parts)

    def render_graph_diff(
        self,
        from_snapshot: str,
        to_snapshot: str,
        example_name: str,
        example_dir: Path,
    ) -> str | None:
        """Render a graph diff as Mermaid diagram using the CLI.

        Args:
            from_snapshot: Before snapshot version hash.
            to_snapshot: After snapshot version hash.
            example_name: Name of the example for context.
            example_dir: Path to the example directory.

        Returns:
            Mermaid diagram string or None if rendering fails.
        """
        import os
        import subprocess
        import sys

        try:
            # Run metaxy graph-diff render command from the example directory
            metaxy_path = os.path.join(os.path.dirname(sys.executable), "metaxy")

            result = subprocess.run(
                [
                    metaxy_path,
                    "graph-diff",
                    "render",
                    "--format",
                    "mermaid",
                    from_snapshot,
                    to_snapshot,
                ],
                capture_output=True,
                text=True,
                cwd=example_dir,
            )

            if result.returncode != 0:
                return f"```\nError rendering graph diff: {result.stderr}\n```"

            # Wrap the mermaid output in a code block
            return f"```mermaid\n{result.stdout}\n```"

        except Exception as e:
            return f"```\nError rendering graph diff: {e}\n```"

    def render_command_output(
        self, event: CommandExecuted, show_command: bool = True
    ) -> str:
        """Render command execution output as markdown.

        Args:
            event: CommandExecuted event from execution state.
            show_command: Whether to show the command that was executed.

        Returns:
            Markdown string with command output.
        """
        md_parts = []

        if show_command:
            md_parts.append(f"```shell\n$ {event.command}\n```")
            md_parts.append("")

        # Show stdout if present
        if event.stdout:
            md_parts.append("```")
            md_parts.append(event.stdout.rstrip())
            md_parts.append("```")
            md_parts.append("")

        # Skip stderr - warnings/errors are noisy in documentation

        return "\n".join(md_parts)

    def render_patch_applied(
        self, event: PatchApplied, graph_diff_md: str | None = None
    ) -> str:
        """Render patch application event as markdown.

        Args:
            event: PatchApplied event from execution state.
            graph_diff_md: Optional pre-rendered graph diff markdown.

        Returns:
            Markdown string describing the patch application.
        """
        md_parts = []

        md_parts.append(f"**Applied patch:** `{event.patch_path}`")
        md_parts.append("")

        # Only show snapshot changes if they actually differ
        if (
            event.before_snapshot
            and event.after_snapshot
            and event.before_snapshot != event.after_snapshot
        ):
            before_short = event.before_snapshot[:8]
            after_short = event.after_snapshot[:8]
            md_parts.append(
                f"Graph snapshot changed: `{before_short}...` → `{after_short}...`"
            )
            md_parts.append("")

            # Include graph diff if provided
            if graph_diff_md:
                md_parts.append(graph_diff_md)
                md_parts.append("")

        return "\n".join(md_parts)

    def render_graph_pushed(self, event: GraphPushed) -> str:
        """Render graph push event as markdown.

        Args:
            event: GraphPushed event from execution state.

        Returns:
            Markdown string describing the graph push.
        """
        snapshot_short = event.snapshot_version[:8]
        return f"**Graph snapshot recorded:** `{snapshot_short}...`\n\n"
