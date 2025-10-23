"""Tests for entrypoint discovery and loading."""

import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metaxy import (
    Feature,
    FeatureKey,
    FeatureRegistry,
    FeatureSpec,
    FieldKey,
    FieldSpec,
)
from metaxy.entrypoints import (
    EntrypointLoadError,
    discover_and_load_entrypoints,
    load_config_entrypoints,
    load_module_entrypoint,
    load_package_entrypoints,
)


@pytest.fixture
def registry() -> Iterator[FeatureRegistry]:
    """Create a clean FeatureRegistry for testing."""
    reg = FeatureRegistry()
    with reg.use():
        yield reg


@pytest.fixture
def clean_imports():
    """Clean up imported test modules after test."""
    modules_before = set(sys.modules.keys())
    yield
    # Remove any modules added during test
    modules_after = set(sys.modules.keys())
    for module in modules_after - modules_before:
        if module.startswith("test_entrypoint_modules"):
            del sys.modules[module]


# Test fixtures - Define features dynamically in tests
def create_test_feature(name: str, key: list[str]) -> type[Feature]:
    """Helper to create test feature classes."""
    return type(
        name,
        (Feature,),
        {},
        spec=FeatureSpec(
            key=FeatureKey(key),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
        ),
    )


# ========== Tests for load_module_entrypoint ==========


def test_load_module_entrypoint_basic(registry: FeatureRegistry, tmp_path: Path):
    """Test loading a single module entrypoint."""
    # Create a temporary module with a Feature
    module_dir = tmp_path / "test_entrypoint_modules"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    feature_module = module_dir / "feature1.py"
    feature_module.write_text("""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class TestFeature1(Feature, spec=FeatureSpec(
    key=FeatureKey(["test", "feature1"]),
    deps=None,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
)):
    pass
""")

    # Add to sys.path
    sys.path.insert(0, str(tmp_path))
    try:
        # Load the entrypoint
        load_module_entrypoint("test_entrypoint_modules.feature1", registry=registry)

        # Verify feature was registered
        assert FeatureKey(["test", "feature1"]) in registry.features_by_key
        feature_cls = registry.features_by_key[FeatureKey(["test", "feature1"])]
        assert feature_cls.__name__ == "TestFeature1"
    finally:
        sys.path.remove(str(tmp_path))


def test_load_module_entrypoint_nonexistent_module(registry: FeatureRegistry):
    """Test that loading a nonexistent module raises EntrypointLoadError."""
    with pytest.raises(EntrypointLoadError, match="Failed to import entrypoint module"):
        load_module_entrypoint("nonexistent.module.path", registry=registry)


def test_load_module_entrypoint_import_error(registry: FeatureRegistry, tmp_path: Path):
    """Test that module with import errors raises EntrypointLoadError."""
    # Create a module that raises ImportError
    module_dir = tmp_path / "test_bad_module"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    bad_module = module_dir / "bad.py"
    bad_module.write_text("""
import this_module_does_not_exist
""")

    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(
            EntrypointLoadError, match="Failed to import entrypoint module"
        ):
            load_module_entrypoint("test_bad_module.bad", registry=registry)
    finally:
        sys.path.remove(str(tmp_path))


def test_load_module_entrypoint_uses_active_registry(tmp_path: Path):
    """Test that entrypoint loads into the active registry by default."""
    # Create a temporary module with a Feature
    module_dir = tmp_path / "test_active_registry"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    feature_module = module_dir / "feature.py"
    feature_module.write_text("""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class ActiveRegistryFeature(Feature, spec=FeatureSpec(
    key=FeatureKey(["active", "test"]),
    deps=None,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))
    try:
        # Create custom registry and set as active
        custom_registry = FeatureRegistry()
        with custom_registry.use():
            # Load without passing registry parameter
            load_module_entrypoint("test_active_registry.feature")

            # Should be in custom_registry since it was active
            assert FeatureKey(["active", "test"]) in custom_registry.features_by_key
    finally:
        sys.path.remove(str(tmp_path))


# ========== Tests for load_config_entrypoints ==========


def test_load_config_entrypoints_multiple_modules(
    registry: FeatureRegistry, tmp_path: Path
):
    """Test loading multiple modules from config list."""
    # Create multiple test modules
    module_dir = tmp_path / "test_multi_modules"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    for i in range(3):
        feature_module = module_dir / f"feature{i}.py"
        feature_module.write_text(f"""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class Feature{i}(Feature, spec=FeatureSpec(
    key=FeatureKey(["multi", "feature{i}"]),
    deps=None,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))
    try:
        # Load all modules
        load_config_entrypoints(
            [
                "test_multi_modules.feature0",
                "test_multi_modules.feature1",
                "test_multi_modules.feature2",
            ],
            registry=registry,
        )

        # Verify all features were registered
        assert len(registry.features_by_key) == 3
        assert FeatureKey(["multi", "feature0"]) in registry.features_by_key
        assert FeatureKey(["multi", "feature1"]) in registry.features_by_key
        assert FeatureKey(["multi", "feature2"]) in registry.features_by_key
    finally:
        sys.path.remove(str(tmp_path))


def test_load_config_entrypoints_empty_list(registry: FeatureRegistry):
    """Test loading empty entrypoint list does nothing."""
    initial_count = len(registry.features_by_key)
    load_config_entrypoints([], registry=registry)
    assert len(registry.features_by_key) == initial_count


def test_load_config_entrypoints_failure_raises(registry: FeatureRegistry):
    """Test that failure to load one entrypoint raises error."""
    with pytest.raises(EntrypointLoadError, match="Failed to import entrypoint module"):
        load_config_entrypoints(
            ["valid.module", "nonexistent.module"], registry=registry
        )


# ========== Tests for load_package_entrypoints ==========


def test_load_package_entrypoints_discovers_and_loads(registry: FeatureRegistry):
    """Test package entrypoint discovery and loading."""
    # Create mock entry point
    mock_ep = MagicMock()
    mock_ep.name = "test_plugin"
    mock_ep.value = "test_plugin.features"

    # Mock the load() method to define a feature
    def mock_load():
        # Define a feature when load() is called
        with registry.use():

            class PluginFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["plugin", "feature"]),
                    deps=None,
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
                ),
            ):
                pass

    mock_ep.load = mock_load

    # Mock entry_points() to return our mock entry point
    with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
        # Handle both Python 3.10+ and 3.9 return types
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        # Load package entrypoints
        load_package_entrypoints(registry=registry)

        # Verify feature was registered
        assert FeatureKey(["plugin", "feature"]) in registry.features_by_key


def test_load_package_entrypoints_no_entrypoints_found(registry: FeatureRegistry):
    """Test graceful handling when no package entrypoints found."""
    with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
        # Mock no entry points found
        mock_eps = MagicMock()
        mock_eps.select.return_value = []
        mock_entry_points.return_value = mock_eps

        initial_count = len(registry.features_by_key)
        load_package_entrypoints(registry=registry)

        # Should not raise, registry unchanged
        assert len(registry.features_by_key) == initial_count


def test_load_package_entrypoints_custom_group(registry: FeatureRegistry):
    """Test loading from custom entry point group."""
    mock_ep = MagicMock()
    mock_ep.name = "custom_plugin"
    mock_ep.value = "custom.features"

    def mock_load():
        with registry.use():

            class CustomFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["custom", "feature"]),
                    deps=None,
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
                ),
            ):
                pass

    mock_ep.load = mock_load

    with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        load_package_entrypoints(group="custom.group", registry=registry)

        mock_eps.select.assert_called_once_with(group="custom.group")

        assert FeatureKey(["custom", "feature"]) in registry.features_by_key


def test_load_package_entrypoints_load_failure_raises(registry: FeatureRegistry):
    """Test that entry point load failure raises EntrypointLoadError."""
    mock_ep = MagicMock()
    mock_ep.name = "broken_plugin"
    mock_ep.value = "broken.features"
    mock_ep.load.side_effect = ImportError("Module not found")

    with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        with pytest.raises(
            EntrypointLoadError, match="Failed to load package entrypoint"
        ):
            load_package_entrypoints(registry=registry)


# ========== Tests for discover_and_load_entrypoints ==========


def test_discover_and_load_entrypoints_both_sources(
    registry: FeatureRegistry, tmp_path: Path
):
    """Test loading from both config and package sources."""
    # Create config-based module
    module_dir = tmp_path / "config_module"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    feature_module = module_dir / "feature.py"
    feature_module.write_text("""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class ConfigFeature(Feature, spec=FeatureSpec(
    key=FeatureKey(["config", "feature"]),
    deps=None,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))

    try:
        # Mock package entrypoint
        mock_ep = MagicMock()
        mock_ep.name = "package_plugin"
        mock_ep.value = "package.features"

        def mock_load():
            with registry.use():

                class PackageFeature(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["package", "feature"]),
                        deps=None,
                        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
                    ),
                ):
                    pass

        mock_ep.load = mock_load

        with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_ep]
            mock_entry_points.return_value = mock_eps

            # Load from both sources
            result_registry = discover_and_load_entrypoints(
                config_entrypoints=["config_module.feature"],
                load_config=True,
                load_packages=True,
                registry=registry,
            )

            # Verify both features were registered
            assert len(result_registry.features_by_key) == 2
            assert FeatureKey(["config", "feature"]) in result_registry.features_by_key
            assert FeatureKey(["package", "feature"]) in result_registry.features_by_key
            assert result_registry is registry
    finally:
        sys.path.remove(str(tmp_path))


def test_discover_and_load_entrypoints_config_only(
    registry: FeatureRegistry, tmp_path: Path
):
    """Test loading only config entrypoints."""
    module_dir = tmp_path / "config_only"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    feature_module = module_dir / "feature.py"
    feature_module.write_text("""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class ConfigOnlyFeature(Feature, spec=FeatureSpec(
    key=FeatureKey(["config_only", "feature"]),
    deps=None,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))

    try:
        with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
            # Should not be called when load_packages=False
            discover_and_load_entrypoints(
                config_entrypoints=["config_only.feature"],
                load_config=True,
                load_packages=False,
                registry=registry,
            )

            mock_entry_points.assert_not_called()
            assert FeatureKey(["config_only", "feature"]) in registry.features_by_key
    finally:
        sys.path.remove(str(tmp_path))


def test_discover_and_load_entrypoints_packages_only(registry: FeatureRegistry):
    """Test loading only package entrypoints."""
    mock_ep = MagicMock()
    mock_ep.name = "package_only"
    mock_ep.value = "package.only"

    def mock_load():
        with registry.use():

            class PackageOnlyFeature(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["package_only", "feature"]),
                    deps=None,
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
                ),
            ):
                pass

    mock_ep.load = mock_load

    with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        discover_and_load_entrypoints(
            config_entrypoints=None,
            load_config=False,
            load_packages=True,
            registry=registry,
        )

        assert FeatureKey(["package_only", "feature"]) in registry.features_by_key


def test_discover_and_load_entrypoints_returns_registry(registry: FeatureRegistry):
    """Test that discover_and_load_entrypoints returns the populated registry."""
    result = discover_and_load_entrypoints(
        config_entrypoints=None,
        load_config=False,
        load_packages=False,
        registry=registry,
    )

    assert result is registry


def test_discover_and_load_entrypoints_uses_active_registry():
    """Test that discover_and_load_entrypoints uses active registry by default."""
    custom_registry = FeatureRegistry()

    with custom_registry.use():
        with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = []
            mock_entry_points.return_value = mock_eps

            result = discover_and_load_entrypoints(
                config_entrypoints=None, load_packages=False
            )

            assert result is custom_registry


# ========== Integration Tests ==========


def test_duplicate_feature_key_raises_error(registry: FeatureRegistry, tmp_path: Path):
    """Test that duplicate feature keys raise EntrypointLoadError."""
    # Create two modules with the same feature key
    module_dir = tmp_path / "duplicate_test"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    module1 = module_dir / "module1.py"
    module1.write_text("""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class Feature1(Feature, spec=FeatureSpec(
    key=FeatureKey(["duplicate", "key"]),
    deps=None,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
)):
    pass
""")

    module2 = module_dir / "module2.py"
    module2.write_text("""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class Feature2(Feature, spec=FeatureSpec(
    key=FeatureKey(["duplicate", "key"]),  # Same key!
    deps=None,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))

    try:
        # Load first module - should succeed
        load_module_entrypoint("duplicate_test.module1", registry=registry)

        # Load second module with duplicate key - should raise EntrypointLoadError
        # wrapping the underlying ValueError
        with pytest.raises(EntrypointLoadError, match="already registered"):
            load_module_entrypoint("duplicate_test.module2", registry=registry)
    finally:
        sys.path.remove(str(tmp_path))


def test_entrypoints_with_dependencies(registry: FeatureRegistry, tmp_path: Path):
    """Test loading features that have dependencies on each other."""
    module_dir = tmp_path / "deps_test"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    # Upstream feature
    upstream = module_dir / "upstream.py"
    upstream.write_text("""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class UpstreamFeature(Feature, spec=FeatureSpec(
    key=FeatureKey(["deps", "upstream"]),
    deps=None,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
)):
    pass
""")

    # Downstream feature
    downstream = module_dir / "downstream.py"
    downstream.write_text("""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey, FeatureDep

class DownstreamFeature(Feature, spec=FeatureSpec(
    key=FeatureKey(["deps", "downstream"]),
    deps=[FeatureDep(key=FeatureKey(["deps", "upstream"]))],
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))

    try:
        # Load both modules
        load_config_entrypoints(
            ["deps_test.upstream", "deps_test.downstream"], registry=registry
        )

        # Verify both registered
        assert FeatureKey(["deps", "upstream"]) in registry.features_by_key
        assert FeatureKey(["deps", "downstream"]) in registry.features_by_key

        # Verify dependency relationship
        downstream_spec = registry.feature_specs_by_key[
            FeatureKey(["deps", "downstream"])
        ]
        assert downstream_spec.deps is not None
        assert len(downstream_spec.deps) == 1
        assert downstream_spec.deps[0].key == FeatureKey(["deps", "upstream"])
    finally:
        sys.path.remove(str(tmp_path))
