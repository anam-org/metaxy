import random
import uuid
from collections.abc import Generator
from typing import Any

import ibis
import pytest
from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    MetadataStore,
)
from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.models.feature import FeatureGraph
from metaxy_testing.models import SampleFeatureSpec
from metaxy_testing.services import dagger_service, service_image

from metaxy_testing import HashAlgorithmCases, TempFeatureModule

assert HashAlgorithmCases is not None  # ensure the import is not removed


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Treat 'no tests collected' as success when tach deselected everything."""
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        from tach.pytest_plugin import tach_state_key

        state = session.config.stash.get(tach_state_key, None)
        if state is not None and state.skip_enabled:
            session.exitstatus = pytest.ExitCode.OK


def require_fixture(request: pytest.FixtureRequest, name: str) -> Any:
    """Resolve a pytest fixture by name at runtime, bypassing static dependency analysis.

    Use this in pytest-cases case functions to avoid fixture dependency merging.
    When @fixture + @parametrize_with_cases collects case functions, it merges ALL
    fixture parameters from ALL cases into every parametrized variant. This means
    a case that depends on `clickhouse_db` would force non-clickhouse variants to
    also resolve `clickhouse_db`, causing unwanted skips.

    By resolving fixtures through `request.getfixturevalue` instead of declaring
    them as function parameters, we break the static dependency chain so each case
    only resolves its own dependencies.

    Example::

        class MyStoreCases:
            def case_duckdb(self, tmp_path: Path) -> MetadataStore:
                return DuckDBMetadataStore(database=tmp_path / "test.duckdb")

            def case_clickhouse(self, request) -> MetadataStore:
                conn = require_fixture(request, "clickhouse_db")
                return ClickHouseMetadataStore(connection_string=conn)
    """
    return request.getfixturevalue(name)


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
    modules_to_remove = [name for name in sys.modules.keys() if name.startswith("features.") or name == "features"]
    for name in modules_to_remove:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def config(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    test_config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.ext.duckdb.DuckDBMetadataStore",
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
                from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey
                from metaxy_testing.models import SampleFeatureSpec

                class MyFeature(BaseFeature, spec=SampleFeatureSpec(
                    key=FeatureKey(["my_feature"]),

                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
                )):
                    pass

            with metaxy_project.with_features(features):
                result = metaxy_project.run_cli(["push"])
                assert result.returncode == 0
    """
    from metaxy_testing import TempMetaxyProject

    return TempMetaxyProject(tmp_path)


@pytest.fixture
def test_graph_and_features():
    """Create a clean FeatureGraph for testing with test features registered.

    Returns a tuple of (graph, features_dict) where features_dict provides
    easy access to feature classes by simple names.

    Uses TempFeatureModule to make features importable for historical graph reconstruction.
    """
    temp_module = TempFeatureModule("test_stores_features", project="test")

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
        "UpstreamFeatureA": graph.feature_definitions_by_key[FeatureKey(["test_stores", "upstream_a"])],
        "UpstreamFeatureB": graph.feature_definitions_by_key[FeatureKey(["test_stores", "upstream_b"])],
        "DownstreamFeature": graph.feature_definitions_by_key[FeatureKey(["test_stores", "downstream"])],
    }

    yield graph, features

    temp_module.cleanup()


@pytest.fixture
def test_graph(test_graph_and_features):
    """Provide dict of test feature classes for easy access in tests.

    This fixture extracts just the features dict from test_graph for convenience.
    """
    graph, _ = test_graph_and_features
    with graph.use():
        yield graph


@pytest.fixture
def test_features(test_graph_and_features):
    """Provide dict of test feature classes for easy access in tests.

    This fixture extracts just the features dict from test_graph for convenience.
    """
    graph, features = test_graph_and_features
    with graph.use():
        yield features


def pytest_addoption(parser):
    parser.addoption(
        "--sample",
        metavar="N",
        action="store",
        default=-1,
        type=int,
        help="Only run random selected subset of N tests.",
    )


def pytest_collection_modifyitems(session, config, items):
    random_sample_size = config.getoption("--sample")

    if random_sample_size >= 0:
        items[:] = random.sample(items, k=random_sample_size)


@pytest.fixture(scope="session")
def clickhouse_server(pytestconfig: pytest.Config) -> Generator[dict[str, Any]]:
    """A ClickHouse server for the test session, run as a Dagger service."""
    image = service_image("clickhouse", pytestconfig.rootpath / "tests.toml")
    with dagger_service(image, [8123], env={"CLICKHOUSE_SKIP_USER_SETUP": "1"}) as ports:
        yield {"host": "127.0.0.1", "port": ports[8123]}


@pytest.fixture
def clickhouse_db(clickhouse_server):
    """Create a clean test database for each test (function-scoped).

    Creates a unique database, yields connection string, then drops the database.
    """
    host = clickhouse_server["host"]
    port = clickhouse_server["port"]

    # Generate unique database name
    db_name = f"test_{uuid.uuid4().hex[:8]}"

    # Connect to default database to create test database
    default_conn_string = f"clickhouse://{host}:{port}/default"
    conn: Any = ibis.connect(default_conn_string)

    conn.raw_sql(f"CREATE DATABASE {db_name}")
    test_conn_string = f"clickhouse://{host}:{port}/{db_name}"

    yield test_conn_string

    # Cleanup: drop test database
    try:
        conn.raw_sql(f"DROP DATABASE IF EXISTS {db_name}")
    except Exception:
        pass  # Best effort cleanup
