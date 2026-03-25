"""Tests for MetaxyTomlSource."""

from pathlib import Path

import pytest
from pydantic_settings import BaseSettings

from metaxy.config.metaxy_source import MetaxyTomlSource, _discover_config_in_dir


class _DummySettings(BaseSettings):
    pass


STORE_TYPE = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"


# --- Basic loading ---


def test_load_metaxy_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text('project = "test"\nstore = "dev"\n')

    source = MetaxyTomlSource(_DummySettings, config_file=config_file)
    data = source()

    assert data["project"] == "test"
    assert data["store"] == "dev"


def test_load_pyproject_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text('[project]\nname = "foo"\n\n[tool.metaxy]\nproject = "from-pyproject"\n')

    source = MetaxyTomlSource(_DummySettings, config_file=config_file)
    data = source()

    assert data["project"] == "from-pyproject"
    assert "tool" not in data


def test_env_var_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_DB_HOST", "prod.db.example.com")
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text('[stores.prod.config]\nhost = "${TEST_DB_HOST}"\n')

    source = MetaxyTomlSource(_DummySettings, config_file=config_file)
    data = source()

    assert data["stores"]["prod"]["config"]["host"] == "prod.db.example.com"


def test_no_config_file_returns_empty() -> None:
    source = MetaxyTomlSource(_DummySettings, config_file=None)
    assert source() == {}


# --- Extend chain resolution ---


def test_extend_child_overrides_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent.toml"
    parent.write_text('project = "parent"\nstore = "parent_store"\n')

    child = tmp_path / "child.toml"
    child.write_text('extend = "parent.toml"\nproject = "child"\n')

    source = MetaxyTomlSource(_DummySettings, config_file=child)
    data = source()

    assert data["project"] == "child"
    assert data["store"] == "parent_store"
    assert "extend" not in data


def test_extend_stores_merged(tmp_path: Path) -> None:
    parent = tmp_path / "parent.toml"
    parent.write_text(
        f'[stores.prod]\ntype = "{STORE_TYPE}"\n[stores.prod.config]\nroot_path = "/prod"\n'
        f'[stores.parent_only]\ntype = "{STORE_TYPE}"\n'
    )
    child = tmp_path / "child.toml"
    child.write_text(
        f'extend = "parent.toml"\n[stores.prod]\ntype = "{STORE_TYPE}"\n[stores.prod.config]\nroot_path = "/override"\n'
    )

    source = MetaxyTomlSource(_DummySettings, config_file=child)
    data = source()

    assert data["stores"]["prod"]["config"]["root_path"] == "/override"
    assert "parent_only" in data["stores"]


def test_extend_entrypoints_appended(tmp_path: Path) -> None:
    parent = tmp_path / "parent.toml"
    parent.write_text('entrypoints = ["parent.mod"]\n')
    child = tmp_path / "child.toml"
    child.write_text('extend = "parent.toml"\nentrypoints = ["child.mod"]\n')

    source = MetaxyTomlSource(_DummySettings, config_file=child)
    data = source()

    assert data["entrypoints"] == ["parent.mod", "child.mod"]


def test_extend_circular_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.toml"
    a.write_text('extend = "b.toml"\n')
    b = tmp_path / "b.toml"
    b.write_text('extend = "a.toml"\n')

    from metaxy.config import InvalidConfigError

    with pytest.raises(InvalidConfigError, match="Circular"):
        MetaxyTomlSource(_DummySettings, config_file=a)


def test_extend_multi_level(tmp_path: Path) -> None:
    gp = tmp_path / "grandparent.toml"
    gp.write_text('project = "gp"\nstore = "gp_store"\ntheme = "gp_theme"\n')
    parent = tmp_path / "parent.toml"
    parent.write_text('extend = "grandparent.toml"\ntheme = "parent_theme"\n')
    child = tmp_path / "child.toml"
    child.write_text('extend = "parent.toml"\nstore = "child_store"\n')

    source = MetaxyTomlSource(_DummySettings, config_file=child)
    data = source()

    assert data["project"] == "gp"
    assert data["store"] == "child_store"
    assert data["theme"] == "parent_theme"


def test_extend_missing_parent_raises(tmp_path: Path) -> None:
    child = tmp_path / "child.toml"
    child.write_text('extend = "nonexistent.toml"\n')

    from metaxy.config import InvalidConfigError

    with pytest.raises(InvalidConfigError, match="does not exist"):
        MetaxyTomlSource(_DummySettings, config_file=child)


def test_extend_relative_path_resolution(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    parent = tmp_path / "parent.toml"
    parent.write_text('project = "base"\n')

    child = sub / "child.toml"
    child.write_text('extend = "../parent.toml"\n')

    source = MetaxyTomlSource(_DummySettings, config_file=child)
    data = source()

    assert data["project"] == "base"


# --- Config file discovery ---


def test_discover_metaxy_toml(tmp_path: Path) -> None:
    (tmp_path / "metaxy.toml").write_text('project = "discovered"\n')

    assert _discover_config_in_dir(tmp_path) == tmp_path / "metaxy.toml"


def test_discover_skips_pyproject_without_metaxy(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "unrelated"\n')

    assert _discover_config_in_dir(tmp_path) is None


def test_discover_uses_pyproject_with_metaxy(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.metaxy]\nproject = "from-pyproject"\n')

    assert _discover_config_in_dir(tmp_path) == tmp_path / "pyproject.toml"


def test_discover_prefers_metaxy_toml_over_pyproject(tmp_path: Path) -> None:
    (tmp_path / "metaxy.toml").write_text('project = "from-metaxy"\n')
    (tmp_path / "pyproject.toml").write_text('[tool.metaxy]\nproject = "from-pyproject"\n')

    assert _discover_config_in_dir(tmp_path) == tmp_path / "metaxy.toml"


def test_discover_empty_dir(tmp_path: Path) -> None:
    assert _discover_config_in_dir(tmp_path) is None


def test_config_file_property(tmp_path: Path) -> None:
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text('project = "test"\n')

    source = MetaxyTomlSource(_DummySettings, config_file=config_file)

    assert source.config_file == config_file


def test_config_file_none_when_no_file() -> None:
    source = MetaxyTomlSource(_DummySettings, config_file=None)
    assert source.config_file is None


def test_config_file_points_to_child_not_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent.toml"
    parent.write_text('project = "base"\n')
    child = tmp_path / "child.toml"
    child.write_text('extend = "parent.toml"\n')

    source = MetaxyTomlSource(_DummySettings, config_file=child)

    assert source.config_file == child
