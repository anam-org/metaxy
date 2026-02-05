"""Tests for feature project detection and assignment.

This module tests the package-based project detection, verifying that:
1. Features get correct projects from their Python package name
2. Features can override project via __metaxy_project__ variable
3. Config project field doesn't affect feature project detection
4. Project names are validated correctly (for CLI filtering)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from metaxy_testing.models import SampleFeatureSpec

from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
from metaxy._packaging import detect_project_from_package
from metaxy.config import MetaxyConfig
from metaxy.models.feature import FeatureGraph

if TYPE_CHECKING:
    from .conftest import FakePackageFactory


class TestDetectProjectFromPackage:
    """Tests for detect_project_from_package function."""

    def test_returns_top_level_package_name(self, fake_package_factory: FakePackageFactory) -> None:
        """Uses top-level package name when __metaxy_project__ not defined."""
        fake_package_factory.create("fake_test_package")
        result = detect_project_from_package("fake_test_package.features.user")
        assert result == "fake_test_package"

    def test_returns_metaxy_project_variable(self, fake_package_factory: FakePackageFactory) -> None:
        """Uses __metaxy_project__ when defined in top-level package."""
        fake_package_factory.create("fake_custom_project_package", "custom-project-name")
        result = detect_project_from_package("fake_custom_project_package.features.user")
        assert result == "custom-project-name"

    def test_returns_module_name_when_not_in_sys_modules(self) -> None:
        """Falls back to top-level package name when module not loaded."""
        result = detect_project_from_package("nonexistent_package.features")
        assert result == "nonexistent_package"

    def test_raises_for_non_string_metaxy_project(self, fake_package_factory: FakePackageFactory) -> None:
        """Raises TypeError if __metaxy_project__ is not a string."""
        fake_package = fake_package_factory.create("fake_test_package")
        fake_package.__metaxy_project__ = 123  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="must be a string"):
            detect_project_from_package("fake_test_package.features")


class TestFeatureProjectDetection:
    """Tests for Feature.project detection from package."""

    def test_feature_gets_project_from_package_name(self) -> None:
        """Features get project from their top-level package name."""

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["package", "test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # Feature gets project from its module's top-level package
        module_name = TestFeature.__module__
        expected_project = module_name.split(".")[0]
        assert TestFeature.metaxy_project() == expected_project

    def test_config_project_does_not_affect_feature_project(self) -> None:
        """MetaxyConfig.project doesn't change feature's project assignment."""
        config = MetaxyConfig(project="config_project")
        MetaxyConfig.set(config)

        try:

            class ConfigIgnoredFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["config", "ignored", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

            # Feature should get project from package, not config
            module_name = ConfigIgnoredFeature.__module__
            expected_project = module_name.split(".")[0]
            assert ConfigIgnoredFeature.metaxy_project() == expected_project
            assert ConfigIgnoredFeature.metaxy_project() != "config_project"

        finally:
            MetaxyConfig.reset()


class TestProjectNameValidation:
    """Tests for project name validation in MetaxyConfig (for CLI filtering)."""

    def test_empty_name_rejected(self) -> None:
        """Empty project names are rejected."""
        with pytest.raises(ValueError, match="project name cannot be empty"):
            MetaxyConfig(project="")

    def test_forward_slash_rejected(self) -> None:
        """Project names cannot contain forward slashes."""
        with pytest.raises(ValueError, match="project name .* cannot contain forward slashes"):
            MetaxyConfig(project="my/project")

    def test_double_underscore_rejected(self) -> None:
        """Project names cannot contain double underscores."""
        with pytest.raises(ValueError, match="project name .* cannot contain double underscores"):
            MetaxyConfig(project="my__project")

    def test_invalid_chars_rejected(self) -> None:
        """Project names must be alphanumeric with underscores and hyphens."""
        with pytest.raises(
            ValueError,
            match="project name .* must contain only alphanumeric characters, underscores, and hyphens",
        ):
            MetaxyConfig(project="my.project")

        with pytest.raises(
            ValueError,
            match="project name .* must contain only alphanumeric characters, underscores, and hyphens",
        ):
            MetaxyConfig(project="my project")

    def test_valid_names_accepted(self) -> None:
        """Valid project names are accepted."""
        assert MetaxyConfig(project="myproject").project == "myproject"
        assert MetaxyConfig(project="my_project").project == "my_project"
        assert MetaxyConfig(project="my-project").project == "my-project"
        assert MetaxyConfig(project="my_project-123").project == "my_project-123"


class TestConfigProjectLoading:
    """Tests for loading project from config files (used for CLI filtering)."""

    def test_default_project(self) -> None:
        """MetaxyConfig.get() returns None project when not initialized."""
        MetaxyConfig.reset()
        config = MetaxyConfig.get()
        assert config.project is None

    def test_project_from_metaxy_toml(self, tmp_path: Path) -> None:
        """Project loads from metaxy.toml."""
        config_file = tmp_path / "metaxy.toml"
        delta_path = (tmp_path / "delta_dev").as_posix()
        config_file.write_text(
            f"""
project = "my_metaxy_project"

[stores.dev]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{delta_path}"
"""
        )

        config = MetaxyConfig.load(config_file)
        assert config.project == "my_metaxy_project"
        MetaxyConfig.reset()

    def test_project_from_pyproject_toml(self, tmp_path: Path) -> None:
        """Project loads from pyproject.toml [tool.metaxy] section."""
        config_file = tmp_path / "pyproject.toml"
        delta_path = (tmp_path / "delta_dev").as_posix()
        config_file.write_text(
            f"""
[project]
name = "test"

[tool.metaxy]
project = "pyproject_metaxy"

[tool.metaxy.stores.dev]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"

[tool.metaxy.stores.dev.config]
root_path = "{delta_path}"
"""
        )

        config = MetaxyConfig.load(config_file)
        assert config.project == "pyproject_metaxy"
        MetaxyConfig.reset()

    def test_project_override_via_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """METAXY_PROJECT env var overrides config file."""
        config_file = tmp_path / "metaxy.toml"
        delta_path = (tmp_path / "delta_dev").as_posix()
        config_file.write_text(
            f"""
project = "file_project"

[stores.dev]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{delta_path}"
"""
        )

        monkeypatch.setenv("METAXY_PROJECT", "env_project")
        config = MetaxyConfig.load(config_file)
        assert config.project == "env_project"
        MetaxyConfig.reset()


class TestFeatureGraphOperations:
    """Tests for project handling in graph operations."""

    def test_multiple_features_same_project(self) -> None:
        """Multiple features from same package share the same project."""

        class Feature1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["graph", "feature1"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class Feature2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["graph", "feature2"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # Both features should have the same project (from their common package)
        assert Feature1.metaxy_project() == Feature2.metaxy_project()
        # The project should be the top-level package name
        expected_project = Feature1.__module__.split(".")[0]
        assert Feature1.metaxy_project() == expected_project

    def test_feature_project_persists_in_snapshot(self) -> None:
        """Project is preserved in graph snapshot."""
        graph = FeatureGraph()

        with graph.use():

            class SnapshotTestFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["snapshot", "test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

        expected_project = SnapshotTestFeature.__module__.split(".")[0]
        assert SnapshotTestFeature.metaxy_project() == expected_project

        snapshot_data = graph.to_snapshot()
        feature_key_str = "snapshot/test/feature"

        assert feature_key_str in snapshot_data
        assert snapshot_data[feature_key_str]["project"] == expected_project
        assert "metaxy_definition_version" in snapshot_data[feature_key_str]
