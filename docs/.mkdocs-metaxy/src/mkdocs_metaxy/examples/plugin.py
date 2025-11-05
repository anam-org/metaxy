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
        print("DEBUG: MetaxyExamplesPlugin.on_config() called")  # DEBUG

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

        # Pattern to match directive blocks (allow hyphens in directive type)
        directive_pattern = re.compile(
            r"^:::\s+metaxy-example\s+([\w-]+)\s*\n(.*?)\n:::\s*$",
            re.MULTILINE | re.DOTALL,
        )

        def replace_directive(match: re.Match[str]) -> str:
            # These are guaranteed to be non-None due to the check above
            assert self.loader is not None
            assert self.renderer is not None

            directive_type = match.group(1)
            directive_content = match.group(2).strip()

            try:
                # Parse YAML content
                # Dedent the content
                lines = directive_content.split("\n")
                non_empty = [line for line in lines if line.strip()]
                if non_empty:
                    min_indent = min(
                        len(line) - len(line.lstrip()) for line in non_empty
                    )
                    directive_content = "\n".join(
                        line[min_indent:] if len(line) >= min_indent else line
                        for line in lines
                    ).strip()

                params = yaml.safe_load(directive_content) or {}
                example_name = params.get("example")
                if not example_name:
                    return match.group(0)  # Return original if no example name

                # Process based on directive type
                if directive_type == "scenarios":
                    scenarios = self.loader.get_scenarios(example_name)
                    return self.renderer.render_scenarios(scenarios, example_name)
                elif directive_type == "source-link" or directive_type == "github":
                    # Render GitHub source link
                    button_style = params.get("button", True)
                    text = params.get("text", None)
                    return self.renderer.render_source_link(
                        example_name, button_style=button_style, text=text
                    )
                elif directive_type == "file":
                    return self._render_file(example_name, params)
                elif directive_type == "patch":
                    return self._render_patch(example_name, params)
                else:
                    return match.group(0)  # Unknown type, return original

            except Exception as e:
                # Return error message
                return self.renderer.render_error(
                    f"Failed to process metaxy-example {directive_type}", details=str(e)
                )

        return directive_pattern.sub(replace_directive, markdown)

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

        example_name_full = (
            example_name
            if example_name.startswith("example-")
            else f"example-{example_name}"
        )

        hl_lines = None
        snippets_path = f"{example_name_full}/{file_path}"

        if patches:
            patch_enum = self._get_patch_enumeration(example_name)
            version_num = max(patch_enum.get(p, 1) for p in patches)

            # Apply patches to get patched content
            content = self.loader.read_file(example_name, file_path, patches)

            # Extract changed line numbers by comparing patch additions with actual content
            hl_lines = self._extract_changed_lines_from_patches(
                example_name, patches, content
            )

            # Write patched file to generated directory, preserving directory structure
            file_path_obj = Path(file_path)
            # Create versioned filename: dir/file_v2.ext
            if file_path_obj.parent != Path("."):
                # Preserve directory structure
                versioned_name = (
                    f"{file_path_obj.stem}_v{version_num}{file_path_obj.suffix}"
                )
                versioned_path = file_path_obj.parent / versioned_name
                generated_file = self.generated_dir / versioned_path
                generated_file.parent.mkdir(parents=True, exist_ok=True)
                snippets_path = f".generated/{versioned_path}"
            else:
                # No directory structure
                versioned_name = (
                    f"{file_path_obj.stem}_v{version_num}{file_path_obj.suffix}"
                )
                generated_file = self.generated_dir / versioned_name
                snippets_path = f".generated/{versioned_name}"

            generated_file.write_text(content)

        return self.renderer.render_snippet(
            path=snippets_path,
            show_line_numbers=show_linenos,
            hl_lines=hl_lines,
        )

    def _render_patch(self, example_name: str, params: dict[str, Any]) -> str:
        """Render a patch file."""
        if self.loader is None or self.renderer is None or self.generated_dir is None:
            return ""

        patch_path = params.get("path")
        if not patch_path:
            return self.renderer.render_error("Missing required parameter: path")

        content = self.loader.read_patch(example_name, patch_path)

        # Preserve the full path structure for patches
        generated_file = self.generated_dir / patch_path
        generated_file.parent.mkdir(parents=True, exist_ok=True)
        generated_file.write_text(content)

        snippets_path = f".generated/{patch_path}"
        return self.renderer.render_snippet(
            path=snippets_path,
            show_line_numbers=True,
            hl_lines=None,
        )
