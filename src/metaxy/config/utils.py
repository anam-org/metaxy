from typing import Any


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
