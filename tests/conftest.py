import pytest

from metaxy.config import MetaxyConfig
from metaxy.models.feature import FeatureGraph


def pytest_configure(config):
    """Set up test configuration early, before test collection.

    This ensures that features defined at module level in test files
    get the correct project='test' instead of 'default'.
    """
    # Create and set test configuration globally
    test_config = MetaxyConfig(project="test")
    MetaxyConfig.set(test_config)


def pytest_runtest_setup(item):
    """Reset the global feature graph before each test.

    This hook runs after test collection but before each test execution.
    It ensures that features defined in other test files don't pollute
    the global graph for the current test.
    """
    import sys

    from metaxy.models import feature as feature_module

    # Reset the global graph to a fresh instance
    feature_module.graph = FeatureGraph()

    # Also clear any dynamically loaded feature modules from sys.modules
    # This prevents feature classes from previous tests persisting
    modules_to_remove = [
        name
        for name in sys.modules.keys()
        if name.startswith("features.") or name == "features"
    ]
    for name in modules_to_remove:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def test_config():
    """Ensure test configuration with project='test' is active for all tests.

    This fixture yields the test configuration that was set in pytest_configure.
    Individual tests can still override this by setting their own MetaxyConfig if needed.
    """
    # Always ensure test config is set, even if tests reset it
    # This handles cases where tests call MetaxyConfig.reset()
    test_config = MetaxyConfig(project="test")
    MetaxyConfig.set(test_config)

    yield test_config

    # Clean up after test - reset to test config for next test
    # (pytest_runtest_setup will handle feature graph reset)


@pytest.fixture(autouse=True)
def graph(test_config):
    """Create a clean FeatureGraph for testing.

    This will set up a fresh FeatureGraph for each test.
    Features defined in tests will be bound to this graph unless they specify their own graph.
    """
    with FeatureGraph().use() as graph:
        yield graph


@pytest.fixture
def metaxy_project(tmp_path):
    """Create a temporary Metaxy project for testing.

    Provides TempMetaxyProject instance with context manager API for
    dynamically creating feature modules and running CLI commands.

    Example:
        def test_example(metaxy_project):
            def features():
                from metaxy import Feature, TestingFeatureSpec, FeatureKey, FieldSpec, FieldKey

                class MyFeature(Feature, spec=TestingFeatureSpec(
                    key=FeatureKey(["my_feature"]),
                    deps=None,
                    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
                )):
                    pass

            with metaxy_project.with_features(features):
                result = metaxy_project.run_cli("graph", "push")
                assert result.returncode == 0
    """
    from metaxy._testing import TempMetaxyProject

    return TempMetaxyProject(tmp_path)
