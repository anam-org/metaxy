import pytest

from metaxy.models.feature import FeatureGraph


@pytest.fixture
def graph():
    """Create a clean FeatureGraph for testing.

    This will set up a fresh FeatureGraph for each test.
    Features defined in such tests will be bound to the graph.
    """
    with FeatureGraph().use() as graph:
        yield graph
