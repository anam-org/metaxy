"""Utilities for generating configuration documentation from Pydantic BaseSettings."""

from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings


def get_env_var_name(
    field_path: list[str],
    env_prefix: str = "",
    env_nested_delimiter: str = "__",
) -> str:
    """Convert a field path to an environment variable name.

    Args:
        field_path: List of field names representing the path (e.g., ["ext", "sqlmodel", "enable"])
        env_prefix: Environment variable prefix (e.g., "METAXY_")
        env_nested_delimiter: Delimiter for nested fields (e.g., "__")

    Returns:
        Environment variable name (e.g., "METAXY_EXT__SQLMODEL__ENABLE")

    Example:
        >>> get_env_var_name(["store"], "METAXY_")
        'METAXY_STORE'
        >>> get_env_var_name(["ext", "sqlmodel", "enable"], "METAXY_")
        'METAXY_EXT__SQLMODEL__ENABLE'
    """
    var_name = env_nested_delimiter.join(field_path)
    return f"{env_prefix}{var_name}".upper()


def get_toml_path(field_path: list[str]) -> str:
    """Convert a field path to a TOML key path.

    Args:
        field_path: List of field names representing the path

    Returns:
        TOML key path (e.g., "ext.sqlmodel.enable")

    Example:
        >>> get_toml_path(["store"])
        'store'
        >>> get_toml_path(["ext", "sqlmodel", "enable"])
        'ext.sqlmodel.enable'
    """
    return ".".join(field_path)


def format_field_type(field_info: FieldInfo, add_links: bool = False) -> str:
    """Format a field type for display in documentation.

    Args:
        field_info: Pydantic field info
        add_links: Whether to add markdown links to config types

    Returns:
        Human-readable type string with optional links

    Example:
        >>> format_field_type(field_info)
        'str | None'
        >>> format_field_type(field_info, add_links=True)
        '[StoreConfig](#storeconfig) | None'
    """
    annotation = field_info.annotation
    if annotation is None:
        return "Any"

    # Handle type representation
    type_str = str(annotation)

    # Clean up common patterns
    type_str = type_str.replace("typing.", "")
    type_str = type_str.replace("<class '", "").replace("'>", "")

    # Handle Union/Optional types
    if "Union" in type_str or "|" in type_str:
        type_str = type_str.replace("Union[", "").replace("]", "")
        type_str = type_str.replace("NoneType", "None")

    # Add links to config types if requested
    if add_links:
        # Map of type names to anchor IDs
        type_links = {
            "StoreConfig": "#storeconfig",
            "ExtConfig": "#extconfig",
            "SQLModelConfig": "#sqlmodelconfig",
            "PluginConfig": "#pluginconfig",
        }

        for type_name, anchor in type_links.items():
            if type_name in type_str:
                # Create markdown link
                type_str = type_str.replace(type_name, f"[{type_name}]({anchor})")

    return type_str


def format_default_value(default: Any) -> str:
    """Format a default value for display in documentation.

    Args:
        default: Default value

    Returns:
        Human-readable default value string
    """
    if default is None:
        return "`None`"
    elif isinstance(default, str):
        return f'`"{default}"`'
    elif isinstance(default, bool):
        return f"`{default}`"
    elif isinstance(default, (int, float)):
        return f"`{default}`"
    elif isinstance(default, (list, dict)):
        if not default:
            return "`[]`" if isinstance(default, list) else "`{}`"
        return f"`{default}`"
    else:
        return f"`{default}`"


def extract_field_info(
    model: type[BaseSettings] | type[BaseModel],
    path_prefix: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract field information from a Pydantic model recursively.

    Args:
        model: Pydantic BaseSettings or BaseModel class
        path_prefix: Current path prefix for nested models

    Returns:
        List of field information dictionaries with keys:
        - name: Field name
        - path: List of field names from root
        - type: Field type string
        - default: Default value
        - description: Field description
        - required: Whether field is required
        - is_nested: Whether this represents a nested model

    Example:
        >>> info = extract_field_info(MetaxyConfig)
        >>> info[0]
        {'name': 'store', 'path': ['store'], 'type': 'str', 'default': 'dev', ...}
    """
    path_prefix = path_prefix or []
    fields_info = []

    for field_name, field_info in model.model_fields.items():
        current_path = path_prefix + [field_name]

        # Check if this is a nested BaseSettings or BaseModel
        field_type = field_info.annotation
        is_nested_model = False

        # Try to get the actual type (unwrap Optional, etc.)
        try:
            # Handle Optional/Union types
            import typing

            if hasattr(typing, "get_origin"):
                origin = typing.get_origin(field_type)
                if origin is not None:
                    args = typing.get_args(field_type)
                    # Get non-None type from Union
                    for arg in args:
                        if arg is not type(None):  # noqa: E721
                            field_type = arg
                            break

            # Check if it's a Pydantic model
            if isinstance(field_type, type) and issubclass(
                field_type, (BaseSettings, BaseModel)
            ):
                is_nested_model = True
        except (TypeError, AttributeError):
            pass

        # Get field description from docstring or field description
        description = field_info.description or ""

        # Get default value
        default = field_info.default

        # Check for Pydantic sentinel values indicating no default
        from pydantic_core import PydanticUndefined

        if (
            default is Ellipsis or default is PydanticUndefined
        ):  # Required field or no default
            default = None
            required = default is Ellipsis  # Only Ellipsis means required
        else:
            required = False

        field_dict = {
            "name": field_name,
            "path": current_path,
            "type": format_field_type(field_info, add_links=True),
            "default": default,
            "description": description,
            "required": required,
            "is_nested": is_nested_model,
        }

        fields_info.append(field_dict)

        # Recursively extract nested model fields
        if is_nested_model and isinstance(field_type, type):
            try:
                nested_fields = extract_field_info(field_type, current_path)
                fields_info.extend(nested_fields)
            except Exception:
                # If we can't introspect, just skip nested fields
                pass

    return fields_info


def generate_toml_example(
    fields: list[dict[str, Any]],
    include_tool_section: bool = False,
) -> str:
    """Generate a TOML configuration example.

    Args:
        fields: List of field information dictionaries
        include_tool_section: Whether to wrap in [tool.metaxy] section

    Returns:
        TOML configuration string

    Example:
        >>> toml = generate_toml_example(fields, include_tool_section=True)
        >>> print(toml)
        [tool.metaxy]
        store = "dev"
        ...
    """
    lines = []

    if include_tool_section:
        lines.append("[tool.metaxy]")

    # Group fields by section (top-level vs nested)
    top_level = [f for f in fields if len(f["path"]) == 1]
    nested = [f for f in fields if len(f["path"]) > 1]

    # Generate top-level fields
    first_field = True
    for field in top_level:
        if field["is_nested"]:
            continue  # Skip, will be handled in nested section

        # Add blank line between fields (except before first field)
        if not first_field:
            lines.append("")
        first_field = False

        default = field["default"]
        if default is not None:
            value = format_toml_value(default)
            comment = f"  # {field['description']}" if field["description"] else ""
            lines.append(f"{field['name']} = {value}{comment}")
        else:
            # Show optional fields as commented out
            desc = field["description"] if field["description"] else None

            # Build comment - just description or "Optional"
            if desc:
                comment = f"  # Optional: {desc}"
            else:
                comment = "  # Optional"

            # Use appropriate placeholder based on type
            if "dict" in field["type"].lower():
                placeholder = "{}"
            elif "list" in field["type"].lower():
                placeholder = "[]"
            else:
                placeholder = "null"

            lines.append(f"# {field['name']} = {placeholder}{comment}")

    # Generate nested sections
    sections: dict[str, list[dict[str, Any]]] = {}
    for field in nested:
        section_path = ".".join(field["path"][:-1])
        if section_path not in sections:
            sections[section_path] = []
        sections[section_path].append(field)

    for section_path, section_fields in sections.items():
        # Filter out nested model headers - only include actual fields
        actual_fields = [f for f in section_fields if not f["is_nested"]]

        # Skip empty sections
        if not actual_fields:
            continue

        lines.append("")
        lines.append(f"[{section_path}]")

        first_field_in_section = True
        for field in actual_fields:
            # Add blank line between fields within section
            if not first_field_in_section:
                lines.append("")
            first_field_in_section = False

            default = field["default"]
            if default is not None:
                value = format_toml_value(default)
                comment = f"  # {field['description']}" if field["description"] else ""
                lines.append(f"{field['path'][-1]} = {value}{comment}")
            else:
                # Show optional fields as commented out
                desc = field["description"] if field["description"] else None

                # Build comment - just description or "Optional"
                if desc:
                    comment = f"  # Optional: {desc}"
                else:
                    comment = "  # Optional"

                # Use appropriate placeholder based on type
                if "dict" in field["type"].lower():
                    placeholder = "{}"
                elif "list" in field["type"].lower():
                    placeholder = "[]"
                else:
                    placeholder = "null"

                lines.append(f"# {field['path'][-1]} = {placeholder}{comment}")

    return "\n".join(lines)


def format_toml_value(value: Any) -> str:
    """Format a Python value as a TOML value.

    Args:
        value: Python value

    Returns:
        TOML-formatted value string
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        if not value:
            return "[]"
        # Simple list formatting
        items = [format_toml_value(v) for v in value]
        return f"[{', '.join(items)}]"
    elif isinstance(value, dict):
        # For dict, we'd need inline table syntax - simplified here
        return "{}"
    else:
        return str(value)


def generate_env_var_section(
    fields: list[dict[str, Any]],
    env_prefix: str = "METAXY_",
    env_nested_delimiter: str = "__",
) -> str:
    """Generate environment variables documentation section.

    Args:
        fields: List of field information dictionaries
        env_prefix: Environment variable prefix
        env_nested_delimiter: Delimiter for nested fields

    Returns:
        Markdown string with environment variable documentation
    """
    lines = ["## Environment Variables", ""]
    lines.append(
        "All configuration options can be set via environment variables using the `METAXY_` prefix."
    )
    lines.append("For nested fields, use double underscores (`__`) as delimiters.")
    lines.append("")

    for field in fields:
        if field["is_nested"]:
            continue  # Skip nested model headers

        env_var = get_env_var_name(field["path"], env_prefix, env_nested_delimiter)

        lines.append(f"### `{env_var}`")
        lines.append("")

        if field["description"]:
            lines.append(field["description"])
            lines.append("")

        lines.append(f"**Type:** `{field['type']}`")
        lines.append("")

        if field["default"] is not None:
            lines.append(f"**Default:** {format_default_value(field['default'])}")
            lines.append("")

        # Generate example
        example_value = field["default"] if field["default"] is not None else "value"
        if isinstance(example_value, bool):
            example_value = "true" if example_value else "false"
        elif isinstance(example_value, str):
            example_value = example_value
        elif isinstance(example_value, (int, float)):
            example_value = str(example_value)
        else:
            example_value = str(example_value)

        lines.append("**Example:**")
        lines.append("")
        lines.append("```bash")
        lines.append(f"export {env_var}={example_value}")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def generate_fields_table(fields: list[dict[str, Any]]) -> str:
    """Generate a markdown table of configuration fields.

    Args:
        fields: List of field information dictionaries

    Returns:
        Markdown table string
    """
    lines = [
        "## Configuration Fields",
        "",
        "| Field | Type | Default | Description |",
        "|-------|------|---------|-------------|",
    ]

    for field in fields:
        if field["is_nested"]:
            continue  # Skip nested model headers

        field_path = get_toml_path(field["path"])
        field_type = field["type"]
        default = (
            format_default_value(field["default"])
            if field["default"] is not None
            else "Required"
            if field["required"]
            else "-"
        )
        description = field["description"] or "-"

        lines.append(f"| `{field_path}` | `{field_type}` | {default} | {description} |")

    return "\n".join(lines)


def generate_individual_field_doc(
    field: dict[str, Any],
    env_prefix: str = "METAXY_",
    env_nested_delimiter: str = "__",
    include_tool_prefix: bool = False,
) -> str:
    """Generate documentation for an individual field with tabs and env var.

    Args:
        field: Field information dictionary
        env_prefix: Environment variable prefix
        env_nested_delimiter: Delimiter for nested fields
        include_tool_prefix: Whether this is for pyproject.toml (needs [tool.metaxy] prefix)

    Returns:
        Markdown string for the field documentation
    """
    lines = []

    # Field header
    field_path = get_toml_path(field["path"])
    lines.append(f"### `{field_path}`")
    lines.append("")

    # Description
    if field["description"]:
        lines.append(field["description"])
        lines.append("")

    # Type with default value
    # Don't wrap type in backticks if it contains markdown links
    if "[" in field["type"] and "](" in field["type"]:
        # Contains markdown links, don't wrap in backticks
        lines.append(f"**Type:** {field['type']}")
    else:
        # No links, wrap in backticks for code formatting
        lines.append(f"**Type:** `{field['type']}`")

    if field["default"] is not None:
        lines.append(f" | **Default:** {format_default_value(field['default'])}")
    elif field["required"]:
        lines.append(" | **Required**")
    lines.append("")

    # TOML Configuration (tabbed)
    lines.append('=== "metaxy.toml"')
    lines.append("")
    lines.append("    ```toml")

    # Generate TOML example
    if len(field["path"]) == 1:
        # Top-level field
        if field["default"] is not None:
            value = format_toml_value(field["default"])
            lines.append(f"    {field['name']} = {value}")
        else:
            # Use appropriate placeholder based on type
            if "dict" in field["type"].lower():
                placeholder = "{}"
            elif "list" in field["type"].lower():
                placeholder = "[]"
            else:
                placeholder = "null"

            lines.append(f"    # {field['name']} = {placeholder}  # Optional")
    else:
        # Nested field
        section_path = ".".join(field["path"][:-1])
        lines.append(f"    [{section_path}]")
        if field["default"] is not None:
            value = format_toml_value(field["default"])
            lines.append(f"    {field['path'][-1]} = {value}")
        else:
            # Use appropriate placeholder based on type
            if "dict" in field["type"].lower():
                placeholder = "{}"
            elif "list" in field["type"].lower():
                placeholder = "[]"
            else:
                placeholder = "null"

            lines.append(f"    # {field['path'][-1]} = {placeholder}  # Optional")

    lines.append("    ```")
    lines.append("")

    # pyproject.toml version
    lines.append('=== "pyproject.toml"')
    lines.append("")
    lines.append("    ```toml")
    lines.append("    [tool.metaxy]")

    if len(field["path"]) == 1:
        # Top-level field
        if field["default"] is not None:
            value = format_toml_value(field["default"])
            lines.append(f"    {field['name']} = {value}")
        else:
            # Use appropriate placeholder based on type
            if "dict" in field["type"].lower():
                placeholder = "{}"
            elif "list" in field["type"].lower():
                placeholder = "[]"
            else:
                placeholder = "null"

            lines.append(f"    # {field['name']} = {placeholder}  # Optional")
    else:
        # Nested field - need full path with [tool.metaxy.section]
        section_path = ".".join(field["path"][:-1])
        lines.append("")
        lines.append(f"    [tool.metaxy.{section_path}]")
        if field["default"] is not None:
            value = format_toml_value(field["default"])
            lines.append(f"    {field['path'][-1]} = {value}")
        else:
            # Use appropriate placeholder based on type
            if "dict" in field["type"].lower():
                placeholder = "{}"
            elif "list" in field["type"].lower():
                placeholder = "[]"
            else:
                placeholder = "null"

            lines.append(f"    # {field['path'][-1]} = {placeholder}  # Optional")

    lines.append("    ```")
    lines.append("")

    # Environment variable
    env_var = get_env_var_name(field["path"], env_prefix, env_nested_delimiter)
    lines.append("**Environment Variable:**")
    lines.append("")
    lines.append("```bash")

    # Generate example value
    if field["default"] is not None:
        example_value = field["default"]
        if isinstance(example_value, bool):
            example_value = "true" if example_value else "false"
        elif isinstance(example_value, str):
            example_value = example_value
        elif isinstance(example_value, (int, float)):
            example_value = str(example_value)
        elif isinstance(example_value, list):
            example_value = "[]"
        elif isinstance(example_value, dict):
            example_value = "{}"
        else:
            example_value = str(example_value)
    else:
        example_value = "..."

    lines.append(f"export {env_var}={example_value}")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)
