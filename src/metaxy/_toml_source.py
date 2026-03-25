"""TOML configuration source for pydantic-settings with extend chain resolution."""

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import tomli
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

# Pattern for ${VAR} or ${VAR:-default} syntax
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` in strings."""
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            return ""

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, Mapping):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [_expand_env_vars(item) for item in value]
    return value


def _pyproject_has_metaxy(path: Path) -> bool:
    """Check if a pyproject.toml contains a ``[tool.metaxy]`` section."""
    with open(path, "rb") as f:
        data = tomli.load(f)
    return bool(data.get("tool", {}).get("metaxy"))


def _discover_config_in_dir(directory: Path) -> Path | None:
    """Find a metaxy config file in *directory*.

    Prefers ``metaxy.toml``. Falls back to ``pyproject.toml`` only when it
    contains a ``[tool.metaxy]`` section.
    """
    metaxy_toml = directory / "metaxy.toml"
    if metaxy_toml.exists():
        return metaxy_toml

    pyproject_toml = directory / "pyproject.toml"
    if pyproject_toml.exists() and _pyproject_has_metaxy(pyproject_toml):
        return pyproject_toml

    return None


def discover_config_with_parents(start_dir: Path | None = None) -> Path | None:
    """Search *start_dir* and its ancestors for a metaxy config file."""
    current = start_dir or Path.cwd()

    while True:
        found = _discover_config_in_dir(current)
        if found is not None:
            return found

        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file and extract the metaxy config section."""
    with open(path, "rb") as f:
        data = tomli.load(f)

    if path.name == "pyproject.toml":
        return data.get("tool", {}).get("metaxy", {})
    return data


def _merge_toml(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Merge *child* on top of *parent* (raw dicts, pre-validation).

    - Scalars: child wins.
    - Dicts: shallow-merged (parent keys preserved, child keys override).
    - Lists: appended (parent + child).
    - Keys only in parent are preserved.
    """
    merged = dict(parent)
    for key, child_value in child.items():
        parent_value = parent.get(key)
        if isinstance(child_value, dict) and isinstance(parent_value, dict):
            merged[key] = {**parent_value, **child_value}
        elif isinstance(child_value, list) and isinstance(parent_value, list):
            merged[key] = parent_value + child_value
        else:
            merged[key] = child_value
    return merged


def _resolve_extend(
    data: dict[str, Any],
    config_file: Path,
    seen: set[Path],
) -> dict[str, Any]:
    """Recursively resolve the ``extend`` chain, returning a fully merged dict."""
    extend_raw = data.pop("extend", None)
    if extend_raw is None:
        return data

    extend_path = Path(extend_raw)
    if not extend_path.is_absolute():
        extend_path = (config_file.parent / extend_path).resolve()
    else:
        extend_path = extend_path.resolve()

    if not extend_path.exists():
        from metaxy.config import InvalidConfigError

        raise InvalidConfigError(
            f"Config file referenced by 'extend' does not exist: {extend_path}",
            config_file=config_file,
        )

    if extend_path in seen:
        from metaxy.config import InvalidConfigError

        chain = " -> ".join(str(p) for p in seen)
        raise InvalidConfigError(
            f"Circular config inheritance detected: {chain} -> {extend_path}",
            config_file=config_file,
        )

    seen.add(extend_path)
    parent_data = _read_toml(extend_path)
    parent_data = _resolve_extend(parent_data, extend_path, seen)

    return _merge_toml(parent_data, data)


class MetaxyTomlSource(PydanticBaseSettingsSource):
    """Pydantic-settings source that loads TOML config with ``extend`` inheritance.

    Resolves the full inheritance chain at the raw-dict level so that
    pydantic only ever validates the final merged configuration.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        config_file: Path | None,
    ):
        super().__init__(settings_cls)
        self._config_file = config_file
        self._data = self._load()

    @property
    def config_file(self) -> Path | None:
        return self._config_file

    def _load(self) -> dict[str, Any]:
        if self._config_file is None:
            return {}

        data = _read_toml(self._config_file)
        seen: set[Path] = {self._config_file.resolve()}
        data = _resolve_extend(data, self._config_file, seen)
        return _expand_env_vars(data)

    # --- PydanticBaseSettingsSource interface ---

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data
