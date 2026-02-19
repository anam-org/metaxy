"""Utilities for generating configuration documentation from Pydantic BaseSettings."""

from __future__ import annotations

import typing
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings


def get_env_var_name(
    field_path: list[str],
    env_prefix: str = "",
    env_nested_delimiter: str = "__",
) -> str:
    """Convert a field path to an environment variable name."""
    var_name = env_nested_delimiter.join(field_path)
    return f"{env_prefix}{var_name}".upper()


def get_toml_path(field_path: list[str]) -> str:
    """Convert a field path to a TOML key path."""
    return ".".join(field_path)


def format_toml_value(value: Any) -> str:
    """Format a Python value as a TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [format_toml_value(v) for v in value]
        return f"[{', '.join(items)}]"
    if isinstance(value, dict):
        return "{}"
    return str(value)


def _contains_basemodel(annotation: Any) -> bool:
    """Check if an annotation contains BaseModel subclasses anywhere in its structure."""
    if annotation is None:
        return False
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    origin = typing.get_origin(annotation)
    if origin is None:
        return False
    for arg in typing.get_args(annotation):
        if arg is type(None):
            continue
        if _contains_basemodel(arg):
            return True
    return False


def format_field_type(annotation: Any) -> str:
    """Format a type annotation as a clean display string."""
    if annotation is None:
        return "Any"

    origin = typing.get_origin(annotation)

    # Annotated[X, ...] → format X only
    if origin is typing.Annotated:
        return format_field_type(typing.get_args(annotation)[0])

    # Union / X | Y
    if origin is typing.Union or (hasattr(typing, "UnionType") and isinstance(annotation, type | None)):
        import types

        if isinstance(annotation, types.UnionType) or origin is typing.Union:
            args = typing.get_args(annotation)
            non_none = [a for a in args if a is not type(None)]
            formatted = [format_field_type(a) for a in non_none]
            result = " | ".join(formatted)
            if len(non_none) < len(args):
                result += " | None"
            return result

    # Literal
    if origin is Literal:
        args = typing.get_args(annotation)
        return f"Literal[{', '.join(repr(a) for a in args)}]"

    # Generic types: dict[str, Any], list[str], etc.
    if origin is not None:
        args = typing.get_args(annotation)
        origin_name = getattr(origin, "__name__", str(origin)).replace("typing.", "")
        if args:
            return f"{origin_name}[{', '.join(format_field_type(a) for a in args)}]"
        return origin_name

    # Plain class types
    if isinstance(annotation, type):
        return annotation.__name__

    return str(annotation).replace("typing.", "")


def _is_single_value_literal(field_info: FieldInfo) -> bool:
    """Check if a field's annotation is a single-value Literal (discriminator marker)."""
    annotation = field_info.annotation
    if annotation is None:
        return False
    origin = typing.get_origin(annotation)
    if origin is Literal:
        args = typing.get_args(annotation)
        return len(args) == 1
    return False


def _get_literal_value(field_info: FieldInfo) -> str | None:
    """Get the single value from a Literal annotation, if applicable."""
    annotation = field_info.annotation
    if annotation is None:
        return None
    origin = typing.get_origin(annotation)
    if origin is Literal:
        args = typing.get_args(annotation)
        if len(args) == 1:
            return str(args[0])
    return None


def _get_field_placeholder(field_info: FieldInfo) -> str:
    """Get appropriate placeholder value for a field based on its annotation."""
    annotation = field_info.annotation
    if annotation is None:
        return '"..."'

    # Single-value Literal → use the literal value
    literal_val = _get_literal_value(field_info)
    if literal_val is not None:
        return f'"{literal_val}"'

    # Unwrap Optional/Union to get the base type (handles both typing.Union and X | Y)
    import types

    origin = typing.get_origin(annotation)
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        args = typing.get_args(annotation)
        for arg in args:
            if arg is not type(None):
                annotation = arg
                break

    if annotation is str:
        return '"..."'
    if annotation is int:
        return "0"
    if annotation is float:
        return "0.0"
    if annotation is bool:
        return "false"

    # Check for dict/list origins
    origin = typing.get_origin(annotation)
    if origin is dict:
        return "{}"
    if origin is list:
        return "[]"

    return '"..."'


def extract_field_info(
    model: type[BaseSettings] | type[BaseModel],
    path_prefix: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract field information from a Pydantic model recursively.

    Returns list of dicts with keys: name, path, default, description,
    required, is_nested, is_discriminator, hide_from_docs.
    """
    path_prefix = path_prefix or []
    fields_info: list[dict[str, Any]] = []

    # Build a map of field name → defining class for inherited field resolution
    defining_class: dict[str, type] = {}
    for cls in model.__mro__:
        for name in getattr(cls, "__annotations__", {}):
            if name not in defining_class:
                defining_class[name] = cls

    for field_name, field_info in model.model_fields.items():
        current_path = path_prefix + [field_name]

        field_type = field_info.annotation
        is_nested_model = False

        # Unwrap Optional/Union to find the actual type
        if field_type is not None and hasattr(typing, "get_origin"):
            origin = typing.get_origin(field_type)
            if origin is not None:
                args = typing.get_args(field_type)
                for arg in args:
                    if arg is not type(None):
                        field_type = arg
                        break

        # Check if it's a Pydantic model or a union containing models
        if isinstance(field_type, type) and issubclass(field_type, (BaseSettings, BaseModel)):
            is_nested_model = True
        elif _contains_basemodel(field_info.annotation):
            is_nested_model = True

        description = field_info.description or ""

        # Correct required detection
        required = field_info.is_required()

        # Get default value, handling sentinels and default_factory
        default = field_info.default
        if default is PydanticUndefined:
            if field_info.default_factory is not None:
                default = field_info.default_factory()
            else:
                default = None

        # Detect discriminator fields
        is_discriminator = _is_single_value_literal(field_info)

        # Check for mkdocs_metaxy_hide in json_schema_extra
        hide_from_docs = False
        if isinstance(field_info.json_schema_extra, dict):
            hide_from_docs = field_info.json_schema_extra.get("mkdocs_metaxy_hide", False)

        fields_info.append(
            {
                "name": field_name,
                "path": current_path,
                "field_info": field_info,
                "default": default,
                "description": description,
                "required": required,
                "is_nested": is_nested_model,
                "is_discriminator": is_discriminator,
                "hide_from_docs": hide_from_docs,
                "defining_class": defining_class.get(field_name, model),
            }
        )

        # Recursively extract nested model fields (only single concrete models)
        if is_nested_model and isinstance(field_type, type) and issubclass(field_type, (BaseSettings, BaseModel)):
            nested_fields = extract_field_info(field_type, current_path)
            fields_info.extend(nested_fields)

    return fields_info


def _format_env_example_value(value: Any) -> str:
    """Format a value for use in environment variable examples."""
    if value is None:
        return "..."
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[]"
    if isinstance(value, dict):
        return "{}"
    return str(value)


def generate_field_doc(
    field: dict[str, Any],
    path_prefix: list[str],
    env_prefix: str = "METAXY_",
    env_nested_delimiter: str = "__",
    header_level: int = 4,
    class_path: str = "",
) -> str:
    """Generate complete per-field documentation: header, type, description, TOML tabs."""
    parts: list[str] = []
    hashes = "#" * header_level

    # Anchor for cross-page linking (e.g. #metaxy.config.MetaxyConfig.store)
    if class_path:
        anchor_id = f"{class_path}.{field['name']}"
        parts.append(f'<a id="{anchor_id}"></a>')
        parts.append("")

    # Header
    parts.append(f"{hashes} `{field['name']}`")
    parts.append("")

    # Description
    if field["description"]:
        parts.append(field["description"])
        parts.append("")

    # Type + default/required line
    type_str = format_field_type(field["field_info"].annotation)
    if field["default"] is not None:
        default_str = _format_default_display(field["default"])
        parts.append(f"**Type:** `{type_str}` | **Default:** `{default_str}`")
    elif field["required"]:
        parts.append(f"**Type:** `{type_str}` | **Required**")
    else:
        parts.append(f"**Type:** `{type_str}`")
    parts.append("")

    # TOML tabs (skip for nested model fields — they are documented separately)
    if not field["is_nested"]:
        tabs = generate_field_tabs(field, path_prefix, env_prefix, env_nested_delimiter)
        if tabs:
            parts.append(tabs)

    parts.append("---")
    parts.append("")

    return "\n".join(parts)


def _format_default_display(default: Any) -> str:
    """Format a default value for display."""
    if isinstance(default, str):
        return f'"{default}"'
    if isinstance(default, bool):
        return str(default)
    if isinstance(default, dict) and not default:
        return "{}"
    if isinstance(default, list) and not default:
        return "[]"
    return str(default)


def generate_field_tabs(
    field: dict[str, Any],
    path_prefix: list[str],
    env_prefix: str = "METAXY_",
    env_nested_delimiter: str = "__",
) -> str:
    """Generate three-tab markdown block for a single field."""
    full_path = path_prefix + field["path"]
    field_name = field["name"]
    indent = "    "

    # TOML value
    if field["default"] is not None:
        toml_val = format_toml_value(field["default"])
    else:
        toml_val = _get_field_placeholder(field["field_info"])

    # Section header (everything except the last path segment)
    section = ".".join(full_path[:-1]) if len(full_path) > 1 else ""
    pyproject_section = f"tool.metaxy.{section}" if section else "tool.metaxy"
    toml_section = section

    # Env var
    env_var = get_env_var_name(full_path, env_prefix, env_nested_delimiter)
    env_val = _format_env_example_value(field["default"])

    parts: list[str] = []

    # metaxy.toml tab
    parts.append('=== "metaxy.toml"')
    parts.append("")
    parts.append(f"{indent}```toml")
    if toml_section:
        parts.append(f"{indent}[{toml_section}]")
    parts.append(f"{indent}{field_name} = {toml_val}")
    parts.append(f"{indent}```")
    parts.append("")

    # pyproject.toml tab
    parts.append('=== "pyproject.toml"')
    parts.append("")
    parts.append(f"{indent}```toml")
    parts.append(f"{indent}[{pyproject_section}]")
    parts.append(f"{indent}{field_name} = {toml_val}")
    parts.append(f"{indent}```")
    parts.append("")

    # Environment Variable tab
    parts.append('=== "Environment Variable"')
    parts.append("")
    parts.append(f"{indent}```bash")
    parts.append(f"{indent}export {env_var}={env_val}")
    parts.append(f"{indent}```")
    parts.append("")

    return "\n".join(parts)


def generate_config_tabs(
    fields: list[dict[str, Any]],
    path_prefix: list[str],
    env_prefix: str = "METAXY_",
    env_nested_delimiter: str = "__",
) -> str:
    """Generate three-tab markdown block: metaxy.toml, pyproject.toml, env vars.

    Produces a comprehensive example showing all fields in one block per format.
    Discriminator fields and nested union model fields are excluded.
    Required fields are uncommented; optional fields are commented out.
    """
    # Filter to leaf fields only, excluding discriminators and nested models
    leaf_fields = [
        f for f in fields if not f["is_nested"] and not f["is_discriminator"] and not f.get("hide_from_docs", False)
    ]

    if not leaf_fields:
        return ""

    # Build TOML lines grouped by section
    toml_lines = _build_toml_lines(leaf_fields, path_prefix, section_prefix="")
    pyproject_lines = _build_toml_lines(leaf_fields, path_prefix, section_prefix="tool.metaxy")
    env_lines = _build_env_lines(leaf_fields, path_prefix, env_prefix, env_nested_delimiter)

    indent = "    "
    parts: list[str] = []

    # metaxy.toml tab
    parts.append('=== "metaxy.toml"')
    parts.append("")
    parts.append(f"{indent}```toml")
    for line in toml_lines:
        parts.append(f"{indent}{line}" if line else "")
    parts.append(f"{indent}```")
    parts.append("")

    # pyproject.toml tab
    parts.append('=== "pyproject.toml"')
    parts.append("")
    parts.append(f"{indent}```toml")
    for line in pyproject_lines:
        parts.append(f"{indent}{line}" if line else "")
    parts.append(f"{indent}```")
    parts.append("")

    # Environment Variable tab
    parts.append('=== "Environment Variable"')
    parts.append("")
    parts.append(f"{indent}```bash")
    for line in env_lines:
        parts.append(f"{indent}{line}" if line else "")
    parts.append(f"{indent}```")
    parts.append("")

    return "\n".join(parts)


def _build_toml_lines(
    fields: list[dict[str, Any]],
    path_prefix: list[str],
    section_prefix: str,
) -> list[str]:
    """Build TOML example lines for a list of leaf fields."""
    lines: list[str] = []

    # Group fields by their parent section
    sections: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        full_path = path_prefix + field["path"]
        if len(full_path) > 1:
            section_key = ".".join(full_path[:-1])
        else:
            section_key = ""
        sections.setdefault(section_key, []).append(field)

    for section_key, section_fields in sections.items():
        if lines:
            lines.append("")

        # Section header
        if section_key:
            full_section = f"{section_prefix}.{section_key}" if section_prefix else section_key
            lines.append(f"[{full_section}]")
        elif section_prefix:
            lines.append(f"[{section_prefix}]")

        for field in section_fields:
            field_name = field["name"]
            if field["default"] is not None:
                value = format_toml_value(field["default"])
                lines.append(f"{field_name} = {value}")
            else:
                placeholder = _get_field_placeholder(field["field_info"])
                lines.append(f"{field_name} = {placeholder}")

    return lines


def _build_env_lines(
    fields: list[dict[str, Any]],
    path_prefix: list[str],
    env_prefix: str,
    env_nested_delimiter: str,
) -> list[str]:
    """Build environment variable example lines."""
    lines: list[str] = []
    for field in fields:
        full_path = path_prefix + field["path"]
        env_var = get_env_var_name(full_path, env_prefix, env_nested_delimiter)
        example_value = _format_env_example_value(field["default"])

        lines.append(f"export {env_var}={example_value}")

    return lines
