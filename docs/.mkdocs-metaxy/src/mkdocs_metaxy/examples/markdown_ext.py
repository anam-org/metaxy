"""Markdown extension for Metaxy examples.

This module provides a markdown preprocessor that handles directives like:
    ::: metaxy-example scenarios
        example: recompute
    :::

    ::: metaxy-example file
        example: recompute
        path: src/example_recompute/features.py
        stage: initial
    :::

    ::: metaxy-example file
        example: recompute
        path: src/example_recompute/features.py
        patches: ["patches/01_update_parent_algorithm.patch"]
    :::

    ::: metaxy-example patch
        example: recompute
        path: patches/01_update_parent_algorithm.patch
    :::
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
        self.generated_dir.mkdir(exist_ok=True)

        # Cache for patch enumerations per example
        self._patch_enumerations: dict[str, dict[str, int]] = {}

        # Pattern to match directive blocks:
        # ::: metaxy-example <type>
        #     key: value
        #     ...
        # :::
        self.directive_pattern = re.compile(
            r"^:::\s+metaxy-example\s+(\w+)\s*$", re.MULTILINE
        )
        self.end_pattern = re.compile(r"^:::\s*$", re.MULTILINE)

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

            # Find the closing :::
            end_match = self.end_pattern.search(text, directive_start)
            if not end_match:
                # No closing tag, skip this directive
                result_lines.append(text[match.start() : match.end()])
                pos = directive_start
                continue

            # Extract directive content (YAML between ::: lines)
            directive_content = text[directive_start : end_match.start()]

            # Dedent the content - find minimum indentation and remove it
            lines_content = directive_content.split("\n")
            # Filter out empty lines for indentation calculation
            non_empty_lines = [line for line in lines_content if line.strip()]
            if non_empty_lines:
                min_indent = min(
                    len(line) - len(line.lstrip()) for line in non_empty_lines
                )
                directive_content = "\n".join(
                    line[min_indent:] if len(line) >= min_indent else line
                    for line in lines_content
                ).strip()
            else:
                directive_content = ""

            pos = end_match.end()

            # Process the directive
            try:
                html = self._process_directive(directive_type, directive_content)
                result_lines.append(html)
            except Exception as e:
                # Render error
                error_html = self.renderer.render_error(
                    f"Failed to process metaxy-example {directive_type}",
                    details=str(e),
                )
                result_lines.append(error_html)

        # Add remaining text
        result_lines.append(text[pos:])

        final_text = "".join(result_lines)
        final_lines = final_text.split("\n")

        # DEBUG: Print a section of the output
        for i, line in enumerate(final_lines):
            if "Initial Code" in line and i < len(final_lines) - 10:
                print(
                    f"\nDEBUG: Lines around 'Initial Code' ({i} to {i + 10}):"
                )  # DEBUG
                for j in range(i, min(i + 10, len(final_lines))):
                    print(f"  {j}: {repr(final_lines[j])}")  # DEBUG
                break

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
            raise ValueError(
                f"Directive content must be a YAML dictionary, got {type(params)}"
            )

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
        example_name_full = (
            example_name
            if example_name.startswith("example-")
            else f"example-{example_name}"
        )

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
            original_content = self.loader.read_file(
                example_name, file_path, patches=None
            )

            # Compare line by line to find changes
            original_lines = original_content.split("\n")
            patched_lines = content.split("\n")
            hl_lines = []
            for i, (orig, patched) in enumerate(
                zip(original_lines, patched_lines), start=1
            ):
                if orig != patched:
                    hl_lines.append(i)

            # Write patched file to .generated directory
            file_base = Path(file_path).stem
            file_ext = Path(file_path).suffix
            filename = f"{file_base}_v{version_num}{file_ext}"
            generated_file = self.generated_dir / filename
            generated_file.write_text(content)

            # Update snippet path to generated file
            snippets_path = f".generated/{filename}"

        # Use pymdownx.snippets for all files
        result = self.renderer.render_snippet(
            path=snippets_path,
            show_line_numbers=show_linenos,
            hl_lines=hl_lines,
        )
        print(f"DEBUG: render_snippet for {file_path} returned:")  # DEBUG
        print(repr(result))  # DEBUG
        return result

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
        generated_file.write_text(content)

        # Use pymdownx.snippets for patch file
        snippets_path = f".generated/{patch_filename}"
        return self.renderer.render_snippet(
            path=snippets_path,
            show_line_numbers=True,
            hl_lines=None,
        )


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
        preprocessor = MetaxyExamplesPreprocessor(
            md, examples_dir=examples_dir, docs_dir=self._docs_dir
        )
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
