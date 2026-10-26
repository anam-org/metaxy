"""Settings source with stable handling of untyped dictionaries and source precedence."""

import json
from collections.abc import Mapping
from typing import Any, get_args, get_origin

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, EnvSettingsSource


def _decode_json_object(value: str) -> dict[str, Any] | str:
    if not value.lstrip().startswith("{"):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        # Non-JSON strings remain available for field validation.
        return value
    return decoded if isinstance(decoded, dict) else value


def _merge_nested_overrides(
    values: dict[str, Any], overrides: dict[str, Any], *, decode_override_objects: bool = True
) -> dict[str, Any]:
    result = values.copy()
    for key, value in overrides.items():
        parent = result.get(key)
        if isinstance(value, dict) and isinstance(parent, str):
            parent = _decode_json_object(parent)
        elif decode_override_objects and isinstance(value, str) and isinstance(parent, dict):
            value = _decode_json_object(value)
        if isinstance(value, dict) and isinstance(parent, dict):
            value = _merge_nested_overrides(parent, value, decode_override_objects=decode_override_objects)
        result[key] = value
    return result


class MetaxySettingsSource(EnvSettingsSource):
    def __init__(
        self,
        settings_cls: type[BaseSettings],
        *,
        init_values: dict[str, Any],
        file_values: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(settings_cls, **kwargs)
        self._file_values = file_values
        self._init_values = init_values

    def __call__(self) -> dict[str, Any]:
        # Constructor strings replace lower-priority objects and remain available for field validation.
        return _merge_nested_overrides(
            _merge_nested_overrides(self._file_values, super().__call__()),
            self._init_values,
            decode_override_objects=False,
        )

    def next_field(
        self, field: FieldInfo | Any | None, key: str, case_sensitive: bool | None = None
    ) -> FieldInfo | None:
        annotation = field.annotation if isinstance(field, FieldInfo) else field
        if get_origin(annotation) is dict:
            # Preserve the value type, including Any, while traversing named stores.
            return FieldInfo(annotation=get_args(annotation)[-1])
        return super().next_field(field, key, case_sensitive)

    def explode_env_vars(self, field_name: str, field: FieldInfo, env_vars: Mapping[str, str | None]) -> dict[str, Any]:
        if not self.env_nested_delimiter:
            return {}
        prefixes = [
            f"{env_name}{self.env_nested_delimiter}" for _, env_name, _ in self._extract_field_info(field, field_name)
        ]
        nested_vars = {
            name: value for name, value in env_vars.items() if any(name.startswith(prefix) for prefix in prefixes)
        }
        result: dict[str, Any] = {}
        # Apply containers before their nested overrides, regardless of environment order.
        for name in sorted(nested_vars, key=len):
            result = _merge_nested_overrides(
                result, super().explode_env_vars(field_name, field, {name: nested_vars[name]})
            )
        return result
