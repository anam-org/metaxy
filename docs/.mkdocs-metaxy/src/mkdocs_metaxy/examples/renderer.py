"""Markdown rendering for Metaxy example content.

This module provides rendering utilities for:
- Scenario lists with descriptions
- Source code in markdown code blocks
- Diff patches in markdown code blocks
"""

from __future__ import annotations

from typing import Any


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
    ) -> str:
        """Render a code snippet using pymdownx.snippets.

        Args:
            path: Path to the file (relative to snippets base_path).
            show_line_numbers: Whether to show line numbers.
            hl_lines: List of line numbers to highlight.

        Returns:
            Markdown string with snippet directive.
        """
        # Build attributes list
        attrs = []

        # Determine file extension for syntax highlighting
        base, extension = path.rsplit(".", 1)
        if extension:
            # Map patch extension to diff language
            if extension == "patch":
                attrs.append("diff")
            else:
                attrs.append(extension)

        # Note: Line numbers are controlled by pymdownx.highlight config, not fence attributes
        # Note: show_line_numbers parameter is ignored - use pymdownx.highlight configuration instead

        # Add highlighted lines if provided (this IS supported by superfences)
        if hl_lines:
            hl_lines_str = " ".join(str(line) for line in hl_lines)
            attrs.append(f'hl_lines="{hl_lines_str}"')

        # Build the snippet directive for pymdownx.snippets
        # Format: ``` <attrs>
        #         --8<-- "path"
        #         ```
        # Pymdownx.snippets will replace the --8<-- line with actual file content
        attr_str = " ".join(attrs) if attrs else ""

        # Standard code block with snippet directive
        # Need blank line before for block-level recognition
        result = f"\n\n``` {attr_str}\n"
        result += f'--8<-- "{path}"\n'
        result += "```\n\n"

        return result

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
