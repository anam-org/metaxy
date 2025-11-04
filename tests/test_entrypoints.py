"""Tests for entrypoint discovery and loading."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metaxy import (
    Feature,
    FeatureGraph,
    FeatureKey,
    FieldKey,
    FieldSpec,
    TestingFeatureSpec,
)
from metaxy.entrypoints import (
    EntrypointLoadError,
    load_entrypoints,
    load_features,
    load_module_entrypoint,
    load_package_entrypoints,
)


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
        spec=TestingFeatureSpec(
            key=FeatureKey(key),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    )


# ========== Tests for load_module_entrypoint ==========


def test_load_module_entrypoint_basic(graph: FeatureGraph, tmp_path: Path):
    """Test loading a single module entrypoint."""
    # Create a temporary module with a Feature
    module_dir = tmp_path / "test_entrypoint_modules"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    feature_module = module_dir / "feature1.py"
    feature_module.write_text("""
from metaxy import Feature, TestingFeatureSpec, FeatureKey, FieldSpec, FieldKey

class TestFeature1(Feature, spec=TestingFeatureSpec(
    key=FeatureKey(["test", "feature1"]),

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    # Add to sys.path
    sys.path.insert(0, str(tmp_path))
    try:
        # Load the entrypoint
        load_module_entrypoint("test_entrypoint_modules.feature1", graph=graph)

        # Verify feature was registered
        assert FeatureKey(["test", "feature1"]) in graph.features_by_key
        feature_cls = graph.features_by_key[FeatureKey(["test", "feature1"])]
        assert feature_cls.__name__ == "TestFeature1"
    finally:
        sys.path.remove(str(tmp_path))


def test_load_module_entrypoint_nonexistent_module(graph: FeatureGraph):
    """Test that loading a nonexistent module raises EntrypointLoadError."""
    with pytest.raises(EntrypointLoadError, match="Failed to import entrypoint module"):
        load_module_entrypoint("nonexistent.module.path", graph=graph)


def test_load_module_entrypoint_import_error(graph: FeatureGraph, tmp_path: Path):
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
            load_module_entrypoint("test_bad_module.bad", graph=graph)
    finally:
        sys.path.remove(str(tmp_path))


def test_load_module_entrypoint_uses_active_graph(tmp_path: Path):
    """Test that entrypoint loads into the active graph by default."""
    # Create a temporary module with a Feature
    module_dir = tmp_path / "test_active_graph"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    feature_module = module_dir / "feature.py"
    feature_module.write_text("""
from metaxy import Feature, TestingFeatureSpec, FeatureKey, FieldSpec, FieldKey

class ActiveRegistryFeature(Feature, spec=TestingFeatureSpec(
    key=FeatureKey(["active", "test"]),

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))
    try:
        # Create custom graph and set as active
        custom_graph = FeatureGraph()
        with custom_graph.use():
            # Load without passing graph parameter
            load_module_entrypoint("test_active_graph.feature")

            # Should be in custom_graph since it was active
            assert FeatureKey(["active", "test"]) in custom_graph.features_by_key
    finally:
        sys.path.remove(str(tmp_path))


# ========== Tests for load_config_entrypoints ==========


def test_load_config_entrypoints_multiple_modules(graph: FeatureGraph, tmp_path: Path):
    """Test loading multiple modules from config list."""
    # Create multiple test modules
    module_dir = tmp_path / "test_multi_modules"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    for i in range(3):
        feature_module = module_dir / f"feature{i}.py"
        feature_module.write_text(f"""
from metaxy import Feature, TestingFeatureSpec, FeatureKey, FieldSpec, FieldKey

class Feature{i}(Feature, spec=TestingFeatureSpec(
    key=FeatureKey(["multi", "feature{i}"]),

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))
    try:
        # Load all modules
        load_entrypoints(
            [
                "test_multi_modules.feature0",
                "test_multi_modules.feature1",
                "test_multi_modules.feature2",
            ],
            graph=graph,
        )

        # Verify all features were registered
        assert len(graph.features_by_key) == 3
        assert FeatureKey(["multi", "feature0"]) in graph.features_by_key
        assert FeatureKey(["multi", "feature1"]) in graph.features_by_key
        assert FeatureKey(["multi", "feature2"]) in graph.features_by_key
    finally:
        sys.path.remove(str(tmp_path))


def test_load_config_entrypoints_empty_list(graph: FeatureGraph):
    """Test loading empty entrypoint list does nothing."""
    initial_count = len(graph.features_by_key)
    load_entrypoints([], graph=graph)
    assert len(graph.features_by_key) == initial_count


def test_load_config_entrypoints_failure_raises(graph: FeatureGraph):
    """Test that failure to load one entrypoint raises error."""
    with pytest.raises(EntrypointLoadError, match="Failed to import entrypoint module"):
        load_entrypoints(["valid.module", "nonexistent.module"], graph=graph)


# ========== Tests for load_package_entrypoints ==========


def test_load_package_entrypoints_discovers_and_loads(graph: FeatureGraph):
    """Test package entrypoint discovery and loading."""
    # Create mock entry point
    mock_ep = MagicMock()
    mock_ep.name = "test_plugin"
    mock_ep.value = "test_plugin.features"

    # Mock the load() method to define a feature
    def mock_load():
        # Define a feature when load() is called
        with graph.use():

            class PluginFeature(
                Feature,
                spec=TestingFeatureSpec(
                    key=FeatureKey(["plugin", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
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
        load_package_entrypoints(graph=graph)

        # Verify feature was registered
        assert FeatureKey(["plugin", "feature"]) in graph.features_by_key


def test_load_package_entrypoints_no_entrypoints_found(graph: FeatureGraph):
    """Test graceful handling when no package entrypoints found."""
    with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
        # Mock no entry points found
        mock_eps = MagicMock()
        mock_eps.select.return_value = []
        mock_entry_points.return_value = mock_eps

        initial_count = len(graph.features_by_key)
        load_package_entrypoints(graph=graph)

        # Should not raise, graph unchanged
        assert len(graph.features_by_key) == initial_count


def test_load_package_entrypoints_custom_group(graph: FeatureGraph):
    """Test loading from custom entry point group."""
    mock_ep = MagicMock()
    mock_ep.name = "custom_plugin"
    mock_ep.value = "custom.features"

    def mock_load():
        with graph.use():

            class CustomFeature(
                Feature,
                spec=TestingFeatureSpec(
                    key=FeatureKey(["custom", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

    mock_ep.load = mock_load

    with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        load_package_entrypoints(group="custom.group", graph=graph)

        mock_eps.select.assert_called_once_with(group="custom.group")

        assert FeatureKey(["custom", "feature"]) in graph.features_by_key


def test_load_package_entrypoints_load_failure_raises(graph: FeatureGraph):
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
            load_package_entrypoints(graph=graph)


# ========== Tests for load_features ==========


def test_load_features_both_sources(graph: FeatureGraph, tmp_path: Path):
    """Test loading from both config and package sources."""
    # Create config-based module
    module_dir = tmp_path / "config_module"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    feature_module = module_dir / "feature.py"
    feature_module.write_text("""
from metaxy import Feature, TestingFeatureSpec, FeatureKey, FieldSpec, FieldKey

class ConfigFeature(Feature, spec=TestingFeatureSpec(
    key=FeatureKey(["config", "feature"]),

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
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
            with graph.use():

                class PackageFeature(
                    Feature,
                    spec=TestingFeatureSpec(
                        key=FeatureKey(["package", "feature"]),
                        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                    ),
                ):
                    pass

        mock_ep.load = mock_load

        with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_ep]
            mock_entry_points.return_value = mock_eps

            # Load from both sources
            with graph.use():
                result_graph = load_features(
                    entrypoints=["config_module.feature"],
                    load_config=True,
                    load_packages=True,
                )

            # Verify both features were registered
            assert len(result_graph.features_by_key) == 2
            assert FeatureKey(["config", "feature"]) in result_graph.features_by_key
            assert FeatureKey(["package", "feature"]) in result_graph.features_by_key
            assert result_graph is graph
    finally:
        sys.path.remove(str(tmp_path))


def test_load_features_config_only(graph: FeatureGraph, tmp_path: Path):
    """Test loading only config entrypoints."""
    module_dir = tmp_path / "config_only"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    feature_module = module_dir / "feature.py"
    feature_module.write_text("""
from metaxy import Feature, TestingFeatureSpec, FeatureKey, FieldSpec, FieldKey

class ConfigOnlyFeature(Feature, spec=TestingFeatureSpec(
    key=FeatureKey(["config_only", "feature"]),

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))

    try:
        with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
            # Should not be called when load_packages=False
            with graph.use():
                load_features(
                    entrypoints=["config_only.feature"],
                    load_config=True,
                    load_packages=False,
                )

            mock_entry_points.assert_not_called()
            assert FeatureKey(["config_only", "feature"]) in graph.features_by_key
    finally:
        sys.path.remove(str(tmp_path))


def test_load_features_packages_only(graph: FeatureGraph):
    """Test loading only package entrypoints."""
    mock_ep = MagicMock()
    mock_ep.name = "package_only"
    mock_ep.value = "package.only"

    def mock_load():
        with graph.use():

            class PackageOnlyFeature(
                Feature,
                spec=TestingFeatureSpec(
                    key=FeatureKey(["package_only", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

    mock_ep.load = mock_load

    with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        with graph.use():
            load_features(
                entrypoints=None,
                load_config=False,
                load_packages=True,
            )

        assert FeatureKey(["package_only", "feature"]) in graph.features_by_key


def test_load_features_returns_graph(graph: FeatureGraph):
    """Test that load_features returns the populated graph."""
    with graph.use():
        result = load_features(
            entrypoints=None,
            load_config=False,
            load_packages=False,
        )

    assert result is graph


def test_load_features_uses_active_graph():
    """Test that load_features uses active graph by default."""
    custom_graph = FeatureGraph()

    with custom_graph.use():
        with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = []
            mock_entry_points.return_value = mock_eps

            result = load_features(entrypoints=None, load_packages=False)

            assert result is custom_graph


# ========== Integration Tests ==========


def test_duplicate_feature_key_raises_error(graph: FeatureGraph, tmp_path: Path):
    """Test that duplicate feature keys raise EntrypointLoadError."""
    # Create two modules with the same feature key
    module_dir = tmp_path / "duplicate_test"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    module1 = module_dir / "module1.py"
    module1.write_text("""
from metaxy import Feature, TestingFeatureSpec, FeatureKey, FieldSpec, FieldKey

class Feature1(Feature, spec=TestingFeatureSpec(
    key=FeatureKey(["duplicate", "key"]),

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    module2 = module_dir / "module2.py"
    module2.write_text("""
from metaxy import Feature, TestingFeatureSpec, FeatureKey, FieldSpec, FieldKey

class Feature2(Feature, spec=TestingFeatureSpec(
    key=FeatureKey(["duplicate", "key"]),  # Same key!

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))

    try:
        # Load first module - should succeed
        load_module_entrypoint("duplicate_test.module1", graph=graph)

        # Load second module with duplicate key - should raise EntrypointLoadError
        # wrapping the underlying ValueError
        with pytest.raises(EntrypointLoadError, match="already registered"):
            load_module_entrypoint("duplicate_test.module2", graph=graph)
    finally:
        sys.path.remove(str(tmp_path))


def test_entrypoints_with_dependencies(graph: FeatureGraph, tmp_path: Path):
    """Test loading features that have dependencies on each other."""
    module_dir = tmp_path / "deps_test"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    # Upstream feature
    upstream = module_dir / "upstream.py"
    upstream.write_text("""
from metaxy import Feature, TestingFeatureSpec, FeatureKey, FieldSpec, FieldKey

class UpstreamFeature(Feature, spec=TestingFeatureSpec(
    key=FeatureKey(["deps", "upstream"]),

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    # Downstream feature
    downstream = module_dir / "downstream.py"
    downstream.write_text("""
from metaxy import Feature, TestingFeatureSpec, FeatureKey, FieldSpec, FieldKey, FeatureDep

class DownstreamFeature(Feature, spec=TestingFeatureSpec(
    key=FeatureKey(["deps", "downstream"]),
    deps=[FeatureDep(feature=FeatureKey(["deps", "upstream"]))],
    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))

    try:
        # Load both modules
        load_entrypoints(["deps_test.upstream", "deps_test.downstream"], graph=graph)

        # Verify both registered
        assert FeatureKey(["deps", "upstream"]) in graph.features_by_key
        assert FeatureKey(["deps", "downstream"]) in graph.features_by_key

        # Verify dependency relationship
        downstream_spec = graph.feature_specs_by_key[FeatureKey(["deps", "downstream"])]
        assert downstream_spec.deps is not None
        assert len(downstream_spec.deps) == 1
        assert downstream_spec.deps[0].feature == FeatureKey(["deps", "upstream"])
    finally:
        sys.path.remove(str(tmp_path))
