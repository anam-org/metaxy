"""MkDocs plugin for Metaxy configuration documentation.

Runs in on_page_markdown (before mkdocstrings) to replace ``metaxy-config``
directives with a mkdocstrings `:::` block (rendered by griffe-pydantic)
followed by per-field TOML/env var config example tabs.
"""

from __future__ import annotations

import importlib
import logging
import re
from typing import Any

import yaml
from mkdocs.config import Config
from mkdocs.plugins import BasePlugin

from mkdocs_metaxy.config_generator import extract_field_info

log = logging.getLogger("mkdocs")

DIRECTIVE_PATTERN = re.compile(
    r"^:::\s+metaxy-config\s*\n((?:[ \t]+.+\n)*)",
    re.MULTILINE,
)


class MetaxyConfigPluginConfig(Config):
    """Configuration for MetaxyConfigPlugin (none required)."""


class MetaxyConfigPlugin(BasePlugin[MetaxyConfigPluginConfig]):
    """MkDocs plugin that expands ``metaxy-config`` directives.

    Each directive is replaced with a mkdocstrings `:::` block
    followed by per-field TOML/env var config example tabs.
    """

    def on_page_markdown(self, markdown: str, **kwargs: Any) -> str:
        result_parts: list[str] = []
        pos = 0

        for match in DIRECTIVE_PATTERN.finditer(markdown):
            result_parts.append(markdown[pos : match.start()])
            pos = match.end()

            directive_content = _dedent_block(match.group(1))
            replacement = _process_directive(directive_content)
            result_parts.append(replacement)

        result_parts.append(markdown[pos:])
        return "".join(result_parts)


def _dedent_block(text: str) -> str:
    """Remove common leading whitespace from indented block lines."""
    lines = text.split("\n")
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return ""
    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty)
    return "\n".join(line[min_indent:] if len(line) >= min_indent else line for line in lines).strip()


def _process_directive(content: str) -> str:
    """Process a single metaxy-config directive and return replacement markdown."""
    params = yaml.safe_load(content) or {}
    if not isinstance(params, dict):
        msg = f"Directive content must be a YAML dictionary, got {type(params)}"
        raise TypeError(msg)

    class_path = params.get("class")
    if not class_path:
        msg = "Missing required parameter: class"
        raise ValueError(msg)

    header_level = params.get("header_level", 3)
    exclude_fields = _parse_list(params.get("exclude_fields", []))

    # Import the config class
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    config_class = getattr(module, class_name)

    # Extract and filter pydantic fields (top-level only)
    fields = extract_field_info(config_class)
    top_level = [f for f in _filter_fields(fields, exclude_fields) if len(f["path"]) == 1]
    field_names = [f["name"] for f in top_level]

    return _build_mkdocstrings_block(class_path, header_level, field_names)


def _build_mkdocstrings_block(class_path: str, header_level: int, field_names: list[str]) -> str:
    """Build a mkdocstrings `:::` block showing the class and selected fields."""
    members_list = "[" + ", ".join(field_names) + "]" if field_names else "[]"
    return (
        f"::: {class_path}\n"
        f"    options:\n"
        f"        heading_level: {header_level}\n"
        f"        show_source: false\n"
        f"        show_bases: false\n"
        f"        show_signature: false\n"
        f"        show_docstring_attributes: false\n"
        f"        inherited_members: true\n"
        f"        members_order: source\n"
        f"        members: {members_list}\n"
        f"\n"
    )


def _filter_fields(
    fields: list[dict[str, Any]],
    exclude_fields: list[str],
) -> list[dict[str, Any]]:
    """Filter out excluded and internal fields."""
    result: list[dict[str, Any]] = []
    for field in fields:
        if field["name"] == "ext":
            continue
        if field["name"] in exclude_fields:
            continue
        if field.get("hide_from_docs", False):
            continue
        result.append(field)
    return result


def _parse_list(value: Any) -> list[str]:
    """Parse a list parameter from string or list input."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [f.strip() for f in value.split(",")]
    return []
