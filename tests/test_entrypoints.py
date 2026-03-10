"""Tests for entrypoint discovery and loading."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from metaxy_testing.models import SampleFeatureSpec
from packaging.utils import canonicalize_name

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldKey,
    FieldSpec,
    MetaxyMissingFeatureDependency,
)
from metaxy.entrypoints import (
    EntrypointLoadError,
    _filter_entry_points_by_distribution,
    _get_allowed_distributions,
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
def create_test_feature(name: str, key: list[str]) -> type[BaseFeature]:
    """Helper to create test feature classes."""
    return type(
        name,
        (BaseFeature,),
        {},
        spec=SampleFeatureSpec(
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
from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey
from metaxy_testing.models import SampleFeatureSpec

class TestFeature1(BaseFeature, spec=SampleFeatureSpec(
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
        assert FeatureKey(["test", "feature1"]) in graph.feature_definitions_by_key
        definition = graph.get_feature_definition(FeatureKey(["test", "feature1"]))
        assert definition.feature_class_path is not None and definition.feature_class_path.endswith(".TestFeature1")
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
        with pytest.raises(EntrypointLoadError, match="Failed to import entrypoint module"):
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
from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey
from metaxy_testing.models import SampleFeatureSpec

class ActiveRegistryFeature(BaseFeature, spec=SampleFeatureSpec(
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
            assert FeatureKey(["active", "test"]) in custom_graph.feature_definitions_by_key
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
from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey
from metaxy_testing.models import SampleFeatureSpec

class Feature{i}(BaseFeature, spec=SampleFeatureSpec(
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
        assert len(graph.feature_definitions_by_key) == 3
        assert FeatureKey(["multi", "feature0"]) in graph.feature_definitions_by_key
        assert FeatureKey(["multi", "feature1"]) in graph.feature_definitions_by_key
        assert FeatureKey(["multi", "feature2"]) in graph.feature_definitions_by_key
    finally:
        sys.path.remove(str(tmp_path))


def test_load_config_entrypoints_empty_list(graph: FeatureGraph):
    """Test loading empty entrypoint list does nothing."""
    initial_count = len(graph.feature_definitions_by_key)
    load_entrypoints([], graph=graph)
    assert len(graph.feature_definitions_by_key) == initial_count


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

        class PluginFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
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
        assert FeatureKey(["plugin", "feature"]) in graph.feature_definitions_by_key


def test_load_package_entrypoints_no_entrypoints_found(graph: FeatureGraph):
    """Test graceful handling when no package entrypoints found."""
    with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
        # Mock no entry points found
        mock_eps = MagicMock()
        mock_eps.select.return_value = []
        mock_entry_points.return_value = mock_eps

        initial_count = len(graph.feature_definitions_by_key)
        load_package_entrypoints(graph=graph)

        # Should not raise, graph unchanged
        assert len(graph.feature_definitions_by_key) == initial_count


def test_load_package_entrypoints_custom_group(graph: FeatureGraph):
    """Test loading from custom entry point group."""
    mock_ep = MagicMock()
    mock_ep.name = "custom_plugin"
    mock_ep.value = "custom.features"

    def mock_load():
        class CustomFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
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

        assert FeatureKey(["custom", "feature"]) in graph.feature_definitions_by_key


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

        with pytest.raises(EntrypointLoadError, match="Failed to load package entrypoint"):
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
from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey
from metaxy_testing.models import SampleFeatureSpec

class ConfigFeature(BaseFeature, spec=SampleFeatureSpec(
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
            class PackageFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
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
            result_graph = load_features(
                entrypoints=["config_module.feature"],
                load_packages=True,
            )

            # Verify both features were registered
            assert len(result_graph.feature_definitions_by_key) == 2
            assert FeatureKey(["config", "feature"]) in result_graph.feature_definitions_by_key
            assert FeatureKey(["package", "feature"]) in result_graph.feature_definitions_by_key
            assert result_graph is graph
    finally:
        sys.path.remove(str(tmp_path))


def test_load_features_entrypoints_only(graph: FeatureGraph, tmp_path: Path):
    """Test loading only explicit entrypoints."""
    module_dir = tmp_path / "entrypoints_only"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")

    feature_module = module_dir / "feature.py"
    feature_module.write_text("""
from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey
from metaxy_testing.models import SampleFeatureSpec

class EntrypointsOnlyFeature(BaseFeature, spec=SampleFeatureSpec(
    key=FeatureKey(["entrypoints_only", "feature"]),

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    sys.path.insert(0, str(tmp_path))

    try:
        with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
            # Should not be called when load_packages=False
            load_features(
                entrypoints=["entrypoints_only.feature"],
                load_packages=False,
            )

            mock_entry_points.assert_not_called()
            assert FeatureKey(["entrypoints_only", "feature"]) in graph.feature_definitions_by_key
    finally:
        sys.path.remove(str(tmp_path))


def test_load_features_packages_only(graph: FeatureGraph):
    """Test loading only package entrypoints."""
    mock_ep = MagicMock()
    mock_ep.name = "package_only"
    mock_ep.value = "package.only"

    def mock_load():
        class PackageOnlyFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
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

        load_features(
            entrypoints=None,
            load_packages=True,
        )

        assert FeatureKey(["package_only", "feature"]) in graph.feature_definitions_by_key


def test_load_features_returns_graph(graph: FeatureGraph):
    """Test that load_features returns the populated graph."""
    result = load_features(
        entrypoints=None,
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
from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey
from metaxy_testing.models import SampleFeatureSpec

class Feature1(BaseFeature, spec=SampleFeatureSpec(
    key=FeatureKey(["duplicate", "key"]),

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    module2 = module_dir / "module2.py"
    module2.write_text("""
from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey
from metaxy_testing.models import SampleFeatureSpec

class Feature2(BaseFeature, spec=SampleFeatureSpec(
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
from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey
from metaxy_testing.models import SampleFeatureSpec

class UpstreamFeature(BaseFeature, spec=SampleFeatureSpec(
    key=FeatureKey(["deps", "upstream"]),

    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
)):
    pass
""")

    # Downstream feature
    downstream = module_dir / "downstream.py"
    downstream.write_text("""
from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey, FeatureDep
from metaxy_testing.models import SampleFeatureSpec

class DownstreamFeature(BaseFeature, spec=SampleFeatureSpec(
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
        assert FeatureKey(["deps", "upstream"]) in graph.feature_definitions_by_key
        assert FeatureKey(["deps", "downstream"]) in graph.feature_definitions_by_key

        # Verify dependency relationship
        downstream_def = graph.get_feature_definition(FeatureKey(["deps", "downstream"]))
        assert downstream_def.spec.deps is not None
        assert len(downstream_def.spec.deps) == 1
        assert downstream_def.spec.deps[0].feature == FeatureKey(["deps", "upstream"])
    finally:
        sys.path.remove(str(tmp_path))


def test_reverse_order_loading(graph: FeatureGraph):
    """Features can load in reverse dependency order via package entrypoints.

    When features are discovered via importlib.metadata.entry_points(), the
    loading order is non-deterministic. This test verifies that a downstream
    feature loading before its upstream dependency still works correctly.
    """
    # Create mock entry points that load in reverse dependency order
    # (downstream before upstream)
    mock_eps = []

    # Package B (downstream) - loads first
    mock_ep_downstream = MagicMock()
    mock_ep_downstream.name = "package_b"
    mock_ep_downstream.value = "package_b.features"

    def load_downstream():
        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["reverse", "downstream"]),
                deps=[__import__("metaxy").FeatureDep(feature=FeatureKey(["reverse", "upstream"]))],
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    mock_ep_downstream.load = load_downstream
    mock_eps.append(mock_ep_downstream)

    # Package A (upstream) - loads second
    mock_ep_upstream = MagicMock()
    mock_ep_upstream.name = "package_a"
    mock_ep_upstream.value = "package_a.features"

    def load_upstream():
        class UpstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["reverse", "upstream"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    mock_ep_upstream.load = load_upstream
    mock_eps.append(mock_ep_upstream)

    with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
        mock_eps_group = MagicMock()
        mock_eps_group.select.return_value = mock_eps
        mock_entry_points.return_value = mock_eps_group

        # Load features - downstream loads before upstream
        result_graph = load_features(load_packages=True, load_env=False)

        # Verify both features were registered
        assert FeatureKey(["reverse", "upstream"]) in result_graph.feature_definitions_by_key
        assert FeatureKey(["reverse", "downstream"]) in result_graph.feature_definitions_by_key

        # Verify get_feature_plan works for the downstream feature
        # Dependencies are late-bound, so this works after all features are loaded
        plan = result_graph.get_feature_plan(FeatureKey(["reverse", "downstream"]))
        assert plan.feature.key == FeatureKey(["reverse", "downstream"])
        assert plan.deps is not None
        assert len(plan.deps) == 1
        assert plan.deps[0].key == FeatureKey(["reverse", "upstream"])


def test_missing_dependency_raises_clear_error(graph: FeatureGraph) -> None:
    """Accessing feature plan with missing dependency raises MetaxyMissingFeatureDependency."""

    # Create a feature that depends on a non-existent upstream
    class OrphanFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["orphan", "feature"]),
            deps=[FeatureDep(feature=FeatureKey(["nonexistent", "upstream"]))],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Feature registers fine (no eager validation)
    assert FeatureKey(["orphan", "feature"]) in graph.feature_definitions_by_key

    # But accessing the plan raises a clear error
    with pytest.raises(MetaxyMissingFeatureDependency, match="nonexistent/upstream"):
        graph.get_feature_plan(FeatureKey(["orphan", "feature"]))


# ========== Tests for dependency-based filtering ==========


class TestGetAllowedDistributions:
    """Tests for _get_allowed_distributions."""

    def test_returns_none_when_package_not_installed(self) -> None:
        allowed = _get_allowed_distributions("nonexistent-package-xyz-12345")
        assert allowed is None

    def test_includes_project_itself(self) -> None:
        def mock_requires(name: str) -> None:
            return None

        with patch("metaxy.entrypoints.importlib.metadata.requires", side_effect=mock_requires):
            allowed = _get_allowed_distributions("my-project")
            assert allowed is not None
            assert "my-project" in allowed

    def test_includes_direct_dependencies(self) -> None:
        def mock_requires(name: str) -> list[str] | None:
            if name == "my-project":
                return ["dep-one>=1.0", "dep_two", "Dep-Three[extra]"]
            return None

        with patch("metaxy.entrypoints.importlib.metadata.requires", side_effect=mock_requires):
            allowed = _get_allowed_distributions("my-project")
            assert allowed is not None
            assert "my-project" in allowed
            assert "dep-one" in allowed
            assert "dep-two" in allowed
            assert "dep-three" in allowed

    def test_includes_transitive_dependencies(self) -> None:
        def mock_requires(name: str) -> list[str] | None:
            return {
                "my-project": ["core>=1.0"],
                "core": ["utils"],
            }.get(name)

        with patch("metaxy.entrypoints.importlib.metadata.requires", side_effect=mock_requires):
            allowed = _get_allowed_distributions("my-project")
            assert allowed is not None
            assert "my-project" in allowed
            assert "core" in allowed
            assert "utils" in allowed

    def test_canonicalizes_project_name(self) -> None:
        with patch("metaxy.entrypoints.importlib.metadata.requires", return_value=None):
            allowed = _get_allowed_distributions("My_Project")
            assert allowed is not None
            assert "my-project" in allowed

    def test_excludes_extras_only_dependencies(self) -> None:
        """Dependencies gated behind extras are not included in the allowed set."""

        def mock_requires(name: str) -> list[str] | None:
            if name == "proj":
                return [
                    "unconditional-dep",
                    'extras-only-dep ; extra == "dev"',
                    'another-extras-dep ; extra == "test"',
                ]
            return None

        with patch("metaxy.entrypoints.importlib.metadata.requires", side_effect=mock_requires):
            allowed = _get_allowed_distributions("proj")
            assert allowed is not None
            assert "unconditional-dep" in allowed
            assert "extras-only-dep" not in allowed
            assert "another-extras-dep" not in allowed

    def test_includes_deps_matching_current_environment(self) -> None:
        """Dependencies with environment markers that match the current env are included."""
        import sys

        def mock_requires(name: str) -> list[str] | None:
            if name == "proj":
                return [
                    f"env-dep ; python_version >= '{sys.version_info.major}.{sys.version_info.minor}'",
                    "impossible-dep ; python_version < '2.0'",
                ]
            return None

        with patch("metaxy.entrypoints.importlib.metadata.requires", side_effect=mock_requires):
            allowed = _get_allowed_distributions("proj")
            assert allowed is not None
            assert "env-dep" in allowed
            assert "impossible-dep" not in allowed


class TestFilterEntryPointsByDistribution:
    """Tests for _filter_entry_points_by_distribution."""

    def _make_ep(self, dist_name: str | None) -> MagicMock:
        ep = MagicMock(spec=["dist", "name", "value", "load"])
        if dist_name is None:
            ep.dist = None
        else:
            ep.dist = MagicMock()
            ep.dist.name = dist_name
        return ep

    def test_filters_to_allowed(self) -> None:
        ep_a = self._make_ep("package-a")
        ep_b = self._make_ep("package-b")
        ep_c = self._make_ep("package-c")

        result = _filter_entry_points_by_distribution(
            [ep_a, ep_b, ep_c],
            frozenset({canonicalize_name("package-a"), canonicalize_name("package-c")}),
        )
        assert result == [ep_a, ep_c]

    def test_includes_eps_with_no_dist(self) -> None:
        ep_no_dist = self._make_ep(None)
        ep_allowed = self._make_ep("allowed")

        result = _filter_entry_points_by_distribution(
            [ep_no_dist, ep_allowed],
            frozenset({canonicalize_name("allowed")}),
        )
        assert result == [ep_no_dist, ep_allowed]

    def test_canonicalizes_dist_names(self) -> None:
        ep = self._make_ep("My_Package")
        result = _filter_entry_points_by_distribution([ep], frozenset({canonicalize_name("my-package")}))
        assert result == [ep]

    def test_empty_input(self) -> None:
        result = _filter_entry_points_by_distribution([], frozenset({canonicalize_name("any")}))
        assert result == []


class TestLoadPackageEntrypointsFiltering:
    """Tests for filter_project parameter in load_package_entrypoints."""

    def test_filters_by_project_dependencies(self, graph: FeatureGraph) -> None:
        """Only entry points from project dependencies are loaded."""
        allowed_ep = MagicMock()
        allowed_ep.name = "dep_plugin"
        allowed_ep.value = "dep_plugin.features"
        allowed_ep.dist = MagicMock()
        allowed_ep.dist.name = "my-dep"

        def load_allowed():
            class AllowedFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["allowed", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

        allowed_ep.load = load_allowed

        blocked_ep = MagicMock()
        blocked_ep.name = "other_plugin"
        blocked_ep.value = "other_plugin.features"
        blocked_ep.dist = MagicMock()
        blocked_ep.dist.name = "unrelated-pkg"

        def load_blocked():
            class BlockedFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["blocked", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

        blocked_ep.load = load_blocked

        with (
            patch("metaxy.entrypoints.entry_points") as mock_entry_points,
            patch(
                "metaxy.entrypoints.importlib.metadata.requires",
                return_value=["my-dep>=1.0"],
            ),
        ):
            mock_eps = MagicMock()
            mock_eps.select.return_value = [allowed_ep, blocked_ep]
            mock_entry_points.return_value = mock_eps

            load_package_entrypoints(graph=graph, filter_project="my-project")

            assert FeatureKey(["allowed", "feature"]) in graph.feature_definitions_by_key
            assert FeatureKey(["blocked", "feature"]) not in graph.feature_definitions_by_key

    def test_no_filtering_without_filter_project(self, graph: FeatureGraph) -> None:
        """All entry points loaded when filter_project is not set."""
        ep = MagicMock()
        ep.name = "any_plugin"
        ep.value = "any.features"
        ep.dist = MagicMock()
        ep.dist.name = "any-package"

        def mock_load():
            class AnyFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["any", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

        ep.load = mock_load

        with patch("metaxy.entrypoints.entry_points") as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [ep]
            mock_entry_points.return_value = mock_eps

            load_package_entrypoints(graph=graph)

            assert FeatureKey(["any", "feature"]) in graph.feature_definitions_by_key

    def test_extras_deps_not_in_allowed_set(self, graph: FeatureGraph) -> None:
        """Entry points from extras-only dependencies are filtered out."""
        # ep_extras comes from a package that is only an extras dep of the project
        ep_extras = MagicMock()
        ep_extras.name = "extras_plugin"
        ep_extras.value = "extras_plugin.features"
        ep_extras.dist = MagicMock()
        ep_extras.dist.name = "extras-only-pkg"

        def load_extras():
            class ExtrasFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["extras", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

        ep_extras.load = load_extras

        def mock_requires(name: str) -> list[str] | None:
            if name == "my-project":
                return [
                    "real-dep>=1.0",
                    'extras-only-pkg ; extra == "dev"',
                ]
            return None

        with (
            patch("metaxy.entrypoints.entry_points") as mock_entry_points,
            patch(
                "metaxy.entrypoints.importlib.metadata.requires",
                side_effect=mock_requires,
            ),
        ):
            mock_eps = MagicMock()
            mock_eps.select.return_value = [ep_extras]
            mock_entry_points.return_value = mock_eps

            load_package_entrypoints(graph=graph, filter_project="my-project")

            assert FeatureKey(["extras", "feature"]) not in graph.feature_definitions_by_key

    def test_no_filtering_when_package_not_installed(self, graph: FeatureGraph) -> None:
        """All entry points loaded when filter_project package is not installed."""
        ep = MagicMock()
        ep.name = "plugin"
        ep.value = "plugin.features"
        ep.dist = MagicMock()
        ep.dist.name = "some-package"

        def mock_load():
            class SomeFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["some", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
                ),
            ):
                pass

        ep.load = mock_load

        with (
            patch("metaxy.entrypoints.entry_points") as mock_entry_points,
            patch(
                "metaxy.entrypoints._get_allowed_distributions",
                return_value=None,
            ),
        ):
            mock_eps = MagicMock()
            mock_eps.select.return_value = [ep]
            mock_entry_points.return_value = mock_eps

            load_package_entrypoints(graph=graph, filter_project="not-installed-pkg")

            assert FeatureKey(["some", "feature"]) in graph.feature_definitions_by_key
