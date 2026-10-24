import json
from collections.abc import Mapping, Sequence, Set
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel


def _collect_dict_keys(d: dict[str, Any], prefix: str = "") -> list[str]:
    """Recursively collect all keys from a nested dict as dot-separated paths."""
    keys = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        keys.append(full_key)
        if isinstance(value, dict):
            keys.extend(_collect_dict_keys(value, full_key))
    return keys


def _remove_none_values(obj: Any) -> Any:
    """Recursively remove None values from a dict (TOML doesn't support None)."""
    if isinstance(obj, dict):
        return {k: _remove_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_remove_none_values(item) for item in obj if item is not None]
    return obj


def _complex_json_types(annotation: Any) -> tuple[type, ...]:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _complex_json_types(get_args(annotation)[0])
    if origin in (Union, UnionType):
        return tuple(json_type for arg in get_args(annotation) for json_type in _complex_json_types(arg))
    annotation = origin or annotation
    if annotation is type(None):
        return (type(None),)
    if not isinstance(annotation, type) or issubclass(annotation, (str, bytes, bytearray)):
        return ()
    if issubclass(annotation, (Mapping, BaseModel)):
        return (dict,)
    if issubclass(annotation, (Sequence, Set)):
        return (list,)
    return ()


def _decode_complex_values(config: dict[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    """Decode JSON for collection and model fields, leaving other values for validation."""
    result = config.copy()
    for name, field in model.model_fields.items():
        value = config.get(name)
        if not isinstance(value, str):
            continue
        json_types = _complex_json_types(field.annotation)
        if dict not in json_types and list not in json_types:
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            # Field validators may accept non-JSON strings, such as connection URIs.
            continue
        # Preserve strings accepted by another union member when the JSON shape does not match.
        if isinstance(decoded, json_types):
            result[name] = decoded
    return result
