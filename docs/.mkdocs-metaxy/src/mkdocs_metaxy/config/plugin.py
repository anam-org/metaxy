"""MkDocs plugin for Metaxy configuration documentation.

Runs in on_page_markdown (before mkdocstrings) to replace ``metaxy-config``
directives with per-attribute mkdocstrings blocks (griffe-pydantic renders
types, defaults, descriptions) followed by TOML / pyproject.toml / env var
example tabs.
"""

from __future__ import annotations

import importlib
import logging
import re
from typing import Any

import yaml
from mkdocs.config import Config
from mkdocs.plugins import BasePlugin

from mkdocs_metaxy.config_generator import (
    extract_field_info,
    generate_field_tabs,
)

log = logging.getLogger("mkdocs")

DIRECTIVE_PATTERN = re.compile(
    r"^:::\s+metaxy-config\s*\n((?:[ \t]+.+\n)*)",
    re.MULTILINE,
)


class MetaxyConfigPluginConfig(Config):
    """Configuration for MetaxyConfigPlugin (none required)."""


class MetaxyConfigPlugin(BasePlugin[MetaxyConfigPluginConfig]):
    """MkDocs plugin that expands ``metaxy-config`` directives.

    Each directive is replaced with a class docstring block (via mkdocstrings)
    plus per-attribute mkdocstrings blocks with TOML/env var tabs appended.
    """

    def on_page_markdown(self, markdown: str, **kwargs: Any) -> str:
        result_parts: list[str] = []
        pos = 0

        for match in DIRECTIVE_PATTERN.finditer(markdown):
            result_parts.append(markdown[pos : match.start()])
            pos = match.end()

            directive_content = _dedent_block(match.group(1))
            result_parts.append(_process_directive(directive_content))

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
    path_prefix = _parse_path_prefix(params.get("path_prefix", []))

    # Import the config class
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    config_class = getattr(module, class_name)

    # Extract env_prefix from model_config
    model_config = getattr(config_class, "model_config", {})
    if isinstance(model_config, dict):
        env_prefix = model_config.get("env_prefix") or "METAXY_"
        env_nested_delimiter = model_config.get("env_nested_delimiter") or "__"
    else:
        env_prefix = getattr(model_config, "env_prefix", None) or "METAXY_"
        env_nested_delimiter = getattr(model_config, "env_nested_delimiter", None) or "__"

    # Extract and filter fields (top-level only)
    fields = extract_field_info(config_class)
    top_level = [f for f in _filter_fields(fields, exclude_fields) if len(f["path"]) == 1]

    parts: list[str] = []

    # Class docstring via mkdocstrings (no members — fields rendered individually below)
    parts.append(
        f"::: {class_path}\n"
        f"    options:\n"
        f"        heading_level: {header_level}\n"
        f"        show_root_heading: false\n"
        f"        show_source: false\n"
        f"        show_bases: false\n"
        f"        show_signature: false\n"
        f"        members: false\n"
        f"\n"
    )

    # Per-attribute: mkdocstrings block + TOML tabs
    field_heading_level = header_level + 1
    for field in top_level:
        if field["is_discriminator"]:
            continue

        attr_path = f"{class_path}.{field['name']}"

        # mkdocstrings block for this attribute (griffe-pydantic handles rendering)
        parts.append(
            f"::: {attr_path}\n"
            f"    options:\n"
            f"        heading_level: {field_heading_level}\n"
            f"        show_root_heading: true\n"
            f"        show_source: false\n"
            f"\n"
        )

        # Append TOML/env tabs (skip for nested model fields)
        if not field["is_nested"]:
            tabs = generate_field_tabs(field, path_prefix, env_prefix, env_nested_delimiter)
            if tabs:
                parts.append(tabs)
                parts.append("")

    return "\n".join(parts)


def _parse_path_prefix(value: Any) -> list[str]:
    """Parse path_prefix from directive params."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if "," in value:
            return [p.strip() for p in value.split(",")]
        if "." in value:
            return [p.strip() for p in value.split(".")]
        return [value]
    return []


def _filter_fields(
    fields: list[dict[str, Any]],
    exclude_fields: list[str],
) -> list[dict[str, Any]]:
    """Filter out excluded and internal fields."""
    return [
        f
        for f in fields
        if f["name"] != "ext" and f["name"] not in exclude_fields and not f.get("hide_from_docs", False)
    ]


def _parse_list(value: Any) -> list[str]:
    """Parse a list parameter from string or list input."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [f.strip() for f in value.split(",")]
    return []
