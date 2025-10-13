import pytest

from metaxy.models.feature import FeatureRegistry


@pytest.fixture
def registry():
    """Create a clean FeatureRegistry for testing.

    This will set up a fresh FeatureRegistry for each test.
    Features defined in such tests will be bound to the registry.
    """
    with FeatureRegistry().use() as registry:
        yield registry
