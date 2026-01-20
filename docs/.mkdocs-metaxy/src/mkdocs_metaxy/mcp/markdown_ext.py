"""Markdown extension for Metaxy MCP tools documentation.

This module provides a markdown preprocessor that handles directives like:
    ::: metaxy-mcp-tools

The directive will dynamically import the MCP server, extract tool information,
and generate comprehensive documentation for each tool.

Note: This custom extension is used instead of mkdocstrings because MCP tools
use JSON Schema for parameter definitions rather than Python type annotations.
mkdocstrings' Python handler expects Python signatures and docstrings, which
don't capture the JSON Schema metadata that MCP tools expose. This extension
directly parses the JSON Schema from FastMCP tool objects.
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

# Use MkDocs logging pattern for proper integration with --strict mode
log = logging.getLogger("mkdocs.plugins.metaxy_mcp")


def _json_type_to_display(schema: dict[str, Any]) -> str:
    """Convert JSON Schema type to a display string."""
    if "anyOf" in schema:
        types = []
        for option in schema["anyOf"]:
            if option.get("type") == "null":
                continue
            types.append(_json_type_to_display(option))
        if any(opt.get("type") == "null" for opt in schema["anyOf"]):
            return f"{' | '.join(types)} | None"
        return " | ".join(types)

    json_type = schema.get("type", "any")

    if json_type == "array":
        items = schema.get("items", {})
        item_type = _json_type_to_display(items)
        return f"list[{item_type}]"

    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "object": "dict",
        "null": "None",
    }

    return type_map.get(json_type, json_type)


def _generate_tool_doc(tool: Any, header_level: int = 3) -> str:
    """Generate markdown documentation for a single MCP tool.

    Args:
        tool: FastMCP tool object.
        header_level: Header level to use (default: 3 = ###).

    Returns:
        Markdown documentation string.
    """
    header_prefix = "#" * header_level
    lines = [f"{header_prefix} `{tool.name}`", ""]

    # Parse docstring to separate description from Args/Returns sections
    description = tool.description or ""
    desc_parts = description.split("\n\nArgs:")
    main_desc = desc_parts[0].strip()

    # Also handle case where Returns comes before Args or no Args
    if "\n\nReturns:" in main_desc:
        main_desc = main_desc.split("\n\nReturns:")[0].strip()

    if main_desc:
        lines.append(main_desc)
        lines.append("")

    # Generate parameters table if there are any
    params = tool.parameters or {}
    properties = params.get("properties", {})
    required = set(params.get("required", []))

    if properties:
        lines.append("**Parameters:**")
        lines.append("")
        lines.append("| Name | Type | Required | Default | Description |")
        lines.append("|------|------|----------|---------|-------------|")

        # Extract parameter descriptions from docstring
        param_descriptions = {}
        if "Args:" in description:
            args_section = description.split("Args:")[1]
            if "Returns:" in args_section:
                args_section = args_section.split("Returns:")[0]

            # Parse parameter descriptions
            current_param = None
            current_desc_lines: list[str] = []

            for line in args_section.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Check if this is a new parameter (name: description format)
                param_match = re.match(r"^(\w+):\s*(.*)$", line)
                if param_match:
                    # Save previous parameter
                    if current_param:
                        param_descriptions[current_param] = " ".join(
                            current_desc_lines
                        ).strip()

                    current_param = param_match.group(1)
                    current_desc_lines = [param_match.group(2)]
                elif current_param:
                    # Continuation of previous parameter description
                    current_desc_lines.append(line)

            # Save last parameter
            if current_param:
                param_descriptions[current_param] = " ".join(current_desc_lines).strip()

        for param_name, param_schema in properties.items():
            param_type = _json_type_to_display(param_schema)
            is_required = param_name in required

            if "default" not in param_schema:
                default = "—"
            else:
                default = param_schema["default"]
                if default is None:
                    default = "`None`"
                elif default is True:
                    default = "`True`"
                elif default is False:
                    default = "`False`"
                elif isinstance(default, str):
                    default = f'`"{default}"`'
                else:
                    default = f"`{default}`"

            param_desc = param_descriptions.get(param_name, "")
            # Remove default info from description if present (already in table)
            param_desc = re.sub(r"\s*\(default:.*?\)\s*$", "", param_desc)

            lines.append(
                f"| `{param_name}` | `{param_type}` | "
                f"{'Yes' if is_required else 'No'} | {default} | {param_desc} |"
            )

        lines.append("")

    # Generate returns section
    if "Returns:" in description:
        returns_section = description.split("Returns:")[1].strip()
        lines.append("**Returns:**")
        lines.append("")

        # Parse returns section - handle indented lists from Google-style docstrings
        returns_lines = returns_section.split("\n")
        in_list = False
        for line in returns_lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("- "):
                # List item - extract key: value format
                item = stripped[2:]
                lines.append(f"- {item}")
                in_list = True
            elif in_list and ":" in stripped and not stripped.endswith(":"):
                # Continuation that looks like a key: value (part of previous context)
                lines.append(f"- {stripped}")
            elif stripped.endswith(":"):
                # Header line like "Dictionary containing:"
                lines.append(stripped)
                lines.append("")
            else:
                lines.append(stripped)

        lines.append("")

    return "\n".join(lines)


def _generate_mcp_tools_doc(
    module_path: str = "metaxy.ext.mcp.server",
    server_attr: str = "mcp",
    header_level: int = 3,
) -> str:
    """Generate documentation for all MCP tools.

    Args:
        module_path: Python module path to import.
        server_attr: Attribute name of the FastMCP server.
        header_level: Header level for each tool (default: 3).

    Returns:
        Markdown documentation string for all tools.
    """
    # Import the module
    module = importlib.import_module(module_path)
    mcp = getattr(module, server_attr)

    # Get tools from the server, sorted alphabetically by name
    tool_docs = []
    tools = sorted(mcp._tool_manager._tools.values(), key=lambda t: t.name)
    for tool in tools:
        tool_docs.append(_generate_tool_doc(tool, header_level))

    return "\n".join(tool_docs)


class MetaxyMCPPreprocessor(Preprocessor):
    """Preprocessor for Metaxy MCP directives."""

    def __init__(self, md: Markdown | None, **kwargs: Any) -> None:
        """Initialize the preprocessor."""
        super().__init__(md)

        # Pattern to match directive blocks:
        # ::: metaxy-mcp-tools
        #     key: value (optional)
        self.directive_pattern = re.compile(
            r"^:::\s+metaxy-mcp-tools\s*\n?((?:[ \t]+.+\n)*)?", re.MULTILINE
        )

    def run(self, lines: list[str]) -> list[str]:
        """Process markdown lines."""
        text = "\n".join(lines)
        result_lines: list[str] = []
        pos = 0

        for match in self.directive_pattern.finditer(text):
            # Add text before directive
            result_lines.append(text[pos : match.start()])

            # Extract directive content
            directive_content = match.group(1) or ""

            # Dedent the content
            if directive_content:
                lines_content = directive_content.split("\n")
                non_empty_lines = [line for line in lines_content if line.strip()]
                if non_empty_lines:
                    min_indent = min(
                        len(line) - len(line.lstrip()) for line in non_empty_lines
                    )
                    directive_content = "\n".join(
                        line[min_indent:] if len(line) >= min_indent else line
                        for line in lines_content
                    ).strip()

            pos = match.end()

            try:
                markdown = self._process_directive(directive_content)
                result_lines.append(markdown)
            except Exception as e:
                log.warning(f"Failed to process metaxy-mcp-tools directive: {e}")
                error_msg = (
                    f'!!! error "Failed to process metaxy-mcp-tools directive"\n\n'
                    f"    {str(e)}\n\n"
                )
                result_lines.append(error_msg)

        result_lines.append(text[pos:])

        final_text = "".join(result_lines)
        final_lines = final_text.split("\n")
        return final_lines

    def _process_directive(self, content: str) -> str:
        """Process a single directive."""
        # Parse YAML content (optional)
        params: dict[str, Any] = {}
        if content.strip():
            try:
                params = yaml.safe_load(content) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in directive: {e}") from e

        module_path = params.get("module", "metaxy.ext.mcp.server")
        server_attr = params.get("server", "mcp")
        header_level = params.get("header_level", 3)

        return _generate_mcp_tools_doc(module_path, server_attr, header_level)


class MetaxyMCPExtension(Extension):
    """Markdown extension for Metaxy MCP tools documentation."""

    def extendMarkdown(self, md: Markdown) -> None:
        """Register the preprocessor with markdown."""
        preprocessor = MetaxyMCPPreprocessor(md)

        try:
            superfences_prep = md.preprocessors.deregister("fenced_code_block")
            md.preprocessors.register(preprocessor, "metaxy_mcp", 24)
            if superfences_prep is not None:
                md.preprocessors.register(superfences_prep, "fenced_code_block", 26)
        except (ValueError, KeyError):
            md.preprocessors.register(preprocessor, "metaxy_mcp", 24)


def makeExtension(**kwargs: Any) -> MetaxyMCPExtension:
    """Create the extension."""
    return MetaxyMCPExtension(**kwargs)
