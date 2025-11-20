"""Markdown extension for Metaxy configuration documentation.

This module provides a markdown preprocessor that handles directives like:
    ::: metaxy-config
        class: metaxy.ext.sqlmodel.SQLModelPluginConfig
    :::

The directive will dynamically import the specified class, extract field information,
and generate comprehensive documentation for each field.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

import yaml
from markdown import Markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor

from mkdocs_metaxy.config_generator import (
    extract_field_info,
    generate_individual_field_doc,
)


class MetaxyConfigPreprocessor(Preprocessor):
    """Preprocessor for Metaxy config directives."""

    def __init__(self, md: Markdown | None, **kwargs: Any) -> None:
        """Initialize the preprocessor.

        Args:
            md: Markdown instance.
            **kwargs: Additional arguments.
        """
        super().__init__(md)

        # Pattern to match directive blocks:
        # ::: metaxy-config
        #     key: value
        #     ...
        # :::
        self.directive_pattern = re.compile(r"^:::\s+metaxy-config\s*$", re.MULTILINE)
        self.end_pattern = re.compile(r"^:::\s*$", re.MULTILINE)

    def run(self, lines: list[str]) -> list[str]:
        """Process markdown lines.

        Args:
            lines: List of markdown lines.

        Returns:
            Processed lines with directives replaced by generated documentation.
        """
        text = "\n".join(lines)
        result_lines: list[str] = []
        pos = 0

        for match in self.directive_pattern.finditer(text):
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
                markdown = self._process_directive(directive_content)
                result_lines.append(markdown)
            except Exception as e:
                # Render error as admonition
                error_msg = (
                    f'!!! error "Failed to process metaxy-config directive"\n\n'
                    f"    {str(e)}\n\n"
                )
                result_lines.append(error_msg)

        # Add remaining text
        result_lines.append(text[pos:])

        final_text = "".join(result_lines)
        final_lines = final_text.split("\n")
        return final_lines

    def _process_directive(self, content: str) -> str:
        """Process a single directive.

        Args:
            content: YAML content of the directive.

        Returns:
            Markdown string with generated documentation.

        Raises:
            ValueError: If required parameters are missing or class cannot be imported.
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

        class_path = params.get("class")
        if not class_path:
            raise ValueError("Missing required parameter: class")

        # Optional path prefix for nested configs (e.g., ["ext", "sqlmodel"])
        path_prefix = params.get("path_prefix", [])
        if isinstance(path_prefix, str):
            # Allow comma-separated or dot-separated strings
            if "," in path_prefix:
                path_prefix = [p.strip() for p in path_prefix.split(",")]
            elif "." in path_prefix:
                path_prefix = [p.strip() for p in path_prefix.split(".")]
            else:
                path_prefix = [path_prefix]

        # Optional header level (default 3 = h3)
        header_level = params.get("header_level", 3)

        # Import the class dynamically
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            config_class = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import class '{class_path}': {e}") from e

        # Extract env_prefix from model_config if available
        env_prefix = "METAXY_"
        env_nested_delimiter = "__"
        if hasattr(config_class, "model_config"):
            model_config = config_class.model_config
            if isinstance(model_config, dict):
                env_prefix = model_config.get("env_prefix", "METAXY_")
                env_nested_delimiter = model_config.get("env_nested_delimiter", "__")
            else:
                # SettingsConfigDict or similar
                env_prefix = getattr(model_config, "env_prefix", "METAXY_")
                env_nested_delimiter = getattr(
                    model_config, "env_nested_delimiter", "__"
                )

        # Extract field information
        try:
            fields = extract_field_info(config_class)
        except Exception as e:
            raise ValueError(
                f"Failed to extract field info from {class_path}: {e}"
            ) from e

        # Filter out nested model headers - only include actual fields
        actual_fields = [f for f in fields if not f["is_nested"]]

        # # Exclude 'ext' field for MetaxyConfig (plugins documented separately)
        # if class_path == "metaxy.config.MetaxyConfig":
        #     actual_fields = [f for f in actual_fields if f["path"][0] != "ext"]

        # Generate documentation for each field
        field_docs = []
        for field in actual_fields:
            # Skip 'ext' field to avoid infinite recursion
            if field["name"] == "ext":
                continue

            # Store original path for env var generation
            original_path = field["path"].copy()

            # Apply path prefix for TOML display if provided
            if path_prefix:
                field["path"] = path_prefix + field["path"]

            # Generate doc with original path for env vars
            field_doc = generate_individual_field_doc(
                field,
                env_prefix=env_prefix,
                env_nested_delimiter=env_nested_delimiter,
                include_tool_prefix=False,
                header_level=header_level,
                env_var_path=original_path,
            )
            field_docs.append(field_doc)

        # Concatenate all field docs
        markdown = "\n".join(field_docs)

        return markdown


class MetaxyConfigExtension(Extension):
    """Markdown extension for Metaxy configuration documentation."""

    def extendMarkdown(self, md: Markdown) -> None:
        """Register the preprocessor with markdown.

        Args:
            md: Markdown instance.
        """
        preprocessor = MetaxyConfigPreprocessor(md)
        # Register with priority 25 to run before snippets (32)
        md.preprocessors.register(preprocessor, "metaxy_config", 25)


def makeExtension(**kwargs: Any) -> MetaxyConfigExtension:
    """Create the extension.

    Args:
        **kwargs: Configuration options.

    Returns:
        Extension instance.
    """
    return MetaxyConfigExtension(**kwargs)
