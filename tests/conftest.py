import pytest

from metaxy.models.feature import FeatureGraph


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


@pytest.fixture
def graph():
    """Create a clean FeatureGraph for testing.

    This will set up a fresh FeatureGraph for each test.
    Features defined in such tests will be bound to the graph.
    """
    with FeatureGraph().use() as graph:
        yield graph
