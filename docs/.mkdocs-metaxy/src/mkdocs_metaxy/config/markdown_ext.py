"""Markdown extension for Metaxy configuration documentation.

This module provides a markdown preprocessor that handles directives like:
    ::: metaxy-config
        class: metaxy.ext.sqlmodel.SQLModelPluginConfig

The directive will dynamically import the specified class, extract field information,
and generate comprehensive documentation for each field.
"""

from __future__ import annotations

import importlib
import logging
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

# Use MkDocs logger directly for --strict mode to catch warnings
log = logging.getLogger("mkdocs")


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
        # (no closing ::: required - captures all indented lines after directive)
        self.directive_pattern = re.compile(r"^:::\s+metaxy-config\s*\n((?:[ \t]+.+\n)*)", re.MULTILINE)

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
            # Add text before directive
            result_lines.append(text[pos : match.start()])

            # Extract directive content from capture group (indented lines after directive)
            directive_content = match.group(1)

            # Dedent the content - find minimum indentation and remove it
            lines_content = directive_content.split("\n")
            # Filter out empty lines for indentation calculation
            non_empty_lines = [line for line in lines_content if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                directive_content = "\n".join(
                    line[min_indent:] if len(line) >= min_indent else line for line in lines_content
                ).strip()
            else:
                directive_content = ""

            # Update position to end of match
            pos = match.end()

            # Process the directive
            try:
                markdown = self._process_directive(directive_content)
                result_lines.append(markdown)
            except Exception as e:
                # Log warning so strict mode fails
                log.warning(f"Failed to process metaxy-config directive: {e}")
                # Render error as admonition
                error_msg = f'!!! error "Failed to process metaxy-config directive"\n\n    {str(e)}\n\n'
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
            raise ValueError(f"Directive content must be a YAML dictionary, got {type(params)}")

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

        # Optional list of field names to exclude from documentation
        exclude_fields = params.get("exclude_fields", [])
        if isinstance(exclude_fields, str):
            exclude_fields = [f.strip() for f in exclude_fields.split(",")]

        # Import the class dynamically
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            config_class = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import class '{class_path}': {e}") from e

        # Extract env_prefix from model_config
        # For pydantic-settings BaseSettings, model_config is a SettingsConfigDict (TypedDict)
        model_config = getattr(config_class, "model_config", {})
        # Handle both dict-like and object-like access
        if isinstance(model_config, dict):
            env_prefix = model_config.get("env_prefix") or "METAXY_"
            env_nested_delimiter = model_config.get("env_nested_delimiter") or "__"
        else:
            env_prefix = getattr(model_config, "env_prefix", None) or "METAXY_"
            env_nested_delimiter = getattr(model_config, "env_nested_delimiter", None) or "__"

        # Extract field information
        try:
            fields = extract_field_info(config_class)
        except Exception as e:
            raise ValueError(f"Failed to extract field info from {class_path}: {e}") from e

        # Helper to check if a nested field has children
        def has_children(field: dict[str, Any], all_fields: list[dict[str, Any]]) -> bool:
            """Check if a nested model field has child fields."""
            field_path = field["path"]
            for other in all_fields:
                if other is field:
                    continue
                # Check if other field is a child (starts with this field's path)
                if len(other["path"]) > len(field_path) and other["path"][: len(field_path)] == field_path:
                    return True
            return False

        # Filter fields: include leaf fields and nested models with children
        filtered_fields = []
        for field in fields:
            # Skip 'ext' field to avoid infinite recursion
            if field["name"] == "ext":
                continue

            # Skip explicitly excluded fields (from directive or field metadata)
            if field["name"] in exclude_fields:
                continue

            # Skip fields marked with mkdocs_metaxy_hide
            if field.get("hide_from_docs", False):
                continue

            if field["is_nested"]:
                # Only include nested models that have children
                if has_children(field, fields):
                    filtered_fields.append(field)
            else:
                # Include all leaf fields
                filtered_fields.append(field)

        # Build a map of nested model paths for depth calculation
        nested_model_paths = {tuple(field["path"]): field for field in filtered_fields if field["is_nested"]}

        # Generate documentation for each field
        field_docs = []
        for field in filtered_fields:
            # Store original path for depth calculation
            original_path = field["path"].copy()

            # Calculate depth: how many nested models are parents of this field
            depth = 0
            for parent_path in nested_model_paths:
                if len(parent_path) < len(original_path):
                    if original_path[: len(parent_path)] == list(parent_path):
                        depth += 1

            # Apply path prefix for TOML display if provided
            if path_prefix:
                field["path"] = path_prefix + field["path"]

            # Calculate header level based on depth
            field_header_level = header_level + depth

            # For nested models with children, just generate a header
            if field["is_nested"]:
                # Use just field name for header, not full path with prefix
                header_prefix = "#" * field_header_level
                field_doc = f"{header_prefix} `{field['name']}`\n\n"
                if field["description"]:
                    field_doc += f"{field['description']}\n\n"
            else:
                # Generate full doc with examples for leaf fields
                # Use full path (with prefix) for env var generation
                field_doc = generate_individual_field_doc(
                    field,
                    env_prefix=env_prefix,
                    env_nested_delimiter=env_nested_delimiter,
                    include_tool_prefix=False,
                    header_level=field_header_level,
                    env_var_path=field["path"],  # Use full path with prefix
                    header_name=field["name"],  # Use just field name for header
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

        # CRITICAL: Force our preprocessor to run BEFORE superfences
        # Since we inject fenced code blocks dynamically, superfences must process
        # them AFTER we generate them. The extension load order in mkdocs.yml isn't
        # sufficient - we need to explicitly control preprocessor execution order.
        try:
            superfences_prep = md.preprocessors.deregister("fenced_code_block")
            # Register our preprocessor first (lower priority = runs earlier)
            md.preprocessors.register(preprocessor, "metaxy_config", 25)
            # Re-register superfences after us so it processes our generated fences
            if superfences_prep is not None:
                md.preprocessors.register(superfences_prep, "fenced_code_block", 26)
        except (ValueError, KeyError):
            # superfences not registered (edge case), just register ours normally
            md.preprocessors.register(preprocessor, "metaxy_config", 25)


def makeExtension(**kwargs: Any) -> MetaxyConfigExtension:
    """Create the extension.

    Args:
        **kwargs: Configuration options.

    Returns:
        Extension instance.
    """
    return MetaxyConfigExtension(**kwargs)
