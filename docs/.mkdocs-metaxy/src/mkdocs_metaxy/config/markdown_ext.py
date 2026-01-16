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

# Use MkDocs logging pattern for proper integration with --strict mode
log = logging.getLogger("mkdocs.plugins.metaxy_config")


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
        self.directive_pattern = re.compile(
            r"^:::\s+metaxy-config\s*\n((?:[ \t]+.+\n)*)", re.MULTILINE
        )

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
                min_indent = min(
                    len(line) - len(line.lstrip()) for line in non_empty_lines
                )
                directive_content = "\n".join(
                    line[min_indent:] if len(line) >= min_indent else line
                    for line in lines_content
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
        params = self._parse_directive_params(content)
        config_class = self._import_config_class(params["class"])
        env_prefix, env_nested_delimiter = self._extract_env_config(config_class)
        fields = self._extract_fields(config_class, params["class"])
        filtered_fields = self._filter_fields(fields, params["exclude_fields"])
        return self._generate_field_docs(
            filtered_fields,
            fields,
            params["path_prefix"],
            params["header_level"],
            env_prefix,
            env_nested_delimiter,
        )

    def _parse_directive_params(self, content: str) -> dict[str, Any]:
        """Parse and validate directive parameters from YAML content."""
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

        path_prefix = params.get("path_prefix", [])
        if isinstance(path_prefix, str):
            if "," in path_prefix:
                path_prefix = [p.strip() for p in path_prefix.split(",")]
            elif "." in path_prefix:
                path_prefix = [p.strip() for p in path_prefix.split(".")]
            else:
                path_prefix = [path_prefix]

        exclude_fields = params.get("exclude_fields", [])
        if isinstance(exclude_fields, str):
            exclude_fields = [f.strip() for f in exclude_fields.split(",")]

        return {
            "class": class_path,
            "path_prefix": path_prefix,
            "header_level": params.get("header_level", 3),
            "exclude_fields": exclude_fields,
        }

    def _import_config_class(self, class_path: str) -> type:
        """Import a config class from its dotted path."""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import class '{class_path}': {e}") from e

    def _extract_env_config(self, config_class: type) -> tuple[str, str]:
        """Extract environment variable configuration from a config class."""
        model_config = getattr(config_class, "model_config", {})
        if isinstance(model_config, dict):
            env_prefix = model_config.get("env_prefix") or "METAXY_"
            env_nested_delimiter = model_config.get("env_nested_delimiter") or "__"
        else:
            env_prefix = getattr(model_config, "env_prefix", None) or "METAXY_"
            env_nested_delimiter = (
                getattr(model_config, "env_nested_delimiter", None) or "__"
            )
        return env_prefix, env_nested_delimiter

    def _extract_fields(
        self, config_class: type, class_path: str
    ) -> list[dict[str, Any]]:
        """Extract field information from a config class."""
        try:
            return extract_field_info(config_class)
        except Exception as e:
            raise ValueError(
                f"Failed to extract field info from {class_path}: {e}"
            ) from e

    def _filter_fields(
        self, fields: list[dict[str, Any]], exclude_fields: list[str]
    ) -> list[dict[str, Any]]:
        """Filter fields based on exclusion rules."""
        filtered = []
        for field in fields:
            if field["name"] == "ext":
                continue
            if field["name"] in exclude_fields:
                continue
            if field.get("hide_from_docs", False):
                continue
            if field["is_nested"]:
                if self._has_children(field, fields):
                    filtered.append(field)
            else:
                filtered.append(field)
        return filtered

    def _has_children(
        self, field: dict[str, Any], all_fields: list[dict[str, Any]]
    ) -> bool:
        """Check if a nested model field has child fields."""
        field_path = field["path"]
        for other in all_fields:
            if other is field:
                continue
            if (
                len(other["path"]) > len(field_path)
                and other["path"][: len(field_path)] == field_path
            ):
                return True
        return False

    def _generate_field_docs(
        self,
        filtered_fields: list[dict[str, Any]],
        all_fields: list[dict[str, Any]],
        path_prefix: list[str],
        header_level: int,
        env_prefix: str,
        env_nested_delimiter: str,
    ) -> str:
        """Generate documentation for filtered fields."""
        nested_model_paths = {
            tuple(field["path"]): field
            for field in filtered_fields
            if field["is_nested"]
        }

        field_docs = []
        for field in filtered_fields:
            original_path = field["path"].copy()
            depth = self._calculate_field_depth(original_path, nested_model_paths)

            if path_prefix:
                field["path"] = path_prefix + field["path"]

            field_header_level = header_level + depth
            field_doc = self._generate_single_field_doc(
                field, field_header_level, env_prefix, env_nested_delimiter
            )
            field_docs.append(field_doc)

        return "\n".join(field_docs)

    def _calculate_field_depth(
        self, original_path: list[str], nested_model_paths: dict[tuple, Any]
    ) -> int:
        """Calculate nesting depth for a field."""
        depth = 0
        for parent_path in nested_model_paths:
            if len(parent_path) < len(original_path):
                if original_path[: len(parent_path)] == list(parent_path):
                    depth += 1
        return depth

    def _generate_single_field_doc(
        self,
        field: dict[str, Any],
        header_level: int,
        env_prefix: str,
        env_nested_delimiter: str,
    ) -> str:
        """Generate documentation for a single field."""
        if field["is_nested"]:
            header_prefix = "#" * header_level
            field_doc = f"{header_prefix} `{field['name']}`\n\n"
            if field["description"]:
                field_doc += f"{field['description']}\n\n"
            return field_doc

        return generate_individual_field_doc(
            field,
            env_prefix=env_prefix,
            env_nested_delimiter=env_nested_delimiter,
            include_tool_prefix=False,
            header_level=header_level,
            env_var_path=field["path"],
            header_name=field["name"],
        )


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
