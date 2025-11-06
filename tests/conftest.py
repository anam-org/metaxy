import random

import pytest

from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    MetadataStore,
    SampleFeatureSpec,
)
from metaxy._testing import HashAlgorithmCases, TempFeatureModule
from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.models.feature import FeatureGraph

assert HashAlgorithmCases is not None  # ensure the import is not removed


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
def config(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    test_config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": data_dir / "test.duckdb"},
            )
        },
    )
    MetaxyConfig.set(test_config)

    yield test_config

    # Clean up after test - reset to test config for next test
    # (pytest_runtest_setup will handle feature graph reset)


@pytest.fixture
def store(config: MetaxyConfig) -> MetadataStore:
    """Clean MetadataStore for testing"""
    return config.get_store("dev")


@pytest.fixture(autouse=True)
def graph():
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
                from metaxy import Feature, SampleFeatureSpec, FeatureKey, FieldSpec, FieldKey

                class MyFeature(Feature, spec=SampleFeatureSpec(
                    key=FeatureKey(["my_feature"]),

                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
                )):
                    pass

            with metaxy_project.with_features(features):
                result = metaxy_project.run_cli("graph", "push")
                assert result.returncode == 0
    """
    from metaxy._testing import TempMetaxyProject

    return TempMetaxyProject(tmp_path)


@pytest.fixture
def test_graph():
    """Create a clean FeatureGraph for testing with test features registered.

    Returns a tuple of (graph, features_dict) where features_dict provides
    easy access to feature classes by simple names.

    Uses TempFeatureModule to make features importable for historical graph reconstruction.
    """
    temp_module = TempFeatureModule("test_stores_features")

    # Define specs
    upstream_a_spec = SampleFeatureSpec(
        key=FeatureKey(["test_stores", "upstream_a"]),
        fields=[
            FieldSpec(key=FieldKey(["frames"]), code_version="1"),
            FieldSpec(key=FieldKey(["audio"]), code_version="1"),
        ],
    )

    upstream_b_spec = SampleFeatureSpec(
        key=FeatureKey(["test_stores", "upstream_b"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="1"),
        ],
    )

    downstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test_stores", "downstream"]),
        deps=[
            FeatureDep(feature=FeatureKey(["test_stores", "upstream_a"])),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test_stores", "upstream_a"]),
                        fields=[
                            FieldKey(["frames"]),
                            FieldKey(["audio"]),
                        ],
                    )
                ],
            ),
        ],
    )

    # Write to temp module
    temp_module.write_features(
        {
            "UpstreamFeatureA": upstream_a_spec,
            "UpstreamFeatureB": upstream_b_spec,
            "DownstreamFeature": downstream_spec,
        }
    )

    # Get graph from module
    graph = temp_module.graph

    # Create features dict for easy access
    features = {
        "UpstreamFeatureA": graph.features_by_key[
            FeatureKey(["test_stores", "upstream_a"])
        ],
        "UpstreamFeatureB": graph.features_by_key[
            FeatureKey(["test_stores", "upstream_b"])
        ],
        "DownstreamFeature": graph.features_by_key[
            FeatureKey(["test_stores", "downstream"])
        ],
    }

    yield graph, features

    temp_module.cleanup()


@pytest.fixture
def test_features(test_graph):
    """Provide dict of test feature classes for easy access in tests.

    This fixture extracts just the features dict from test_graph for convenience.
    """
    _, features = test_graph
    return features


def pytest_addoption(parser):
    parser.addoption(
        "--random-selection",
        metavar="N",
        action="store",
        default=-1,
        type=int,
        help="Only run random selected subset of N tests.",
    )


def pytest_collection_modifyitems(session, config, items):
    random_sample_size = config.getoption("--random-selection")

    if random_sample_size >= 0:
        items[:] = random.sample(items, k=random_sample_size)
