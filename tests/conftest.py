import random
import shutil
import socket
import subprocess
import time
import uuid
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
from metaxy._testing import HashAlgorithmCases, TempFeatureModule
from metaxy._testing.models import SampleFeatureSpec
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
                from metaxy import BaseFeature as BaseFeature, FeatureKey, FieldSpec, FieldKey
                from metaxy._testing.models import SampleFeatureSpec

                class MyFeature(BaseFeature, spec=SampleFeatureSpec(
                    key=FeatureKey(["my_feature"]),

                    fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")]
                )):
                    pass

            with metaxy_project.with_features(features):
                result = metaxy_project.run_cli(["graph", "push"])
                assert result.returncode == 0
    """
    from metaxy._testing import TempMetaxyProject

    return TempMetaxyProject(tmp_path)


@pytest.fixture
def test_graph_and_features():
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


def find_free_port() -> int:
    """Find a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _check_clickhouse_available() -> str:
    """Check if ClickHouse binary and ibis backend are available.

    Returns the path to the clickhouse binary, or skips the test if unavailable.
    """
    clickhouse_bin = shutil.which("clickhouse") or shutil.which("clickhouse-server")
    if clickhouse_bin is None:
        pytest.skip("ClickHouse binary not found in PATH")
        raise AssertionError("unreachable")

    try:
        import ibis.backends.clickhouse  # noqa: F401
    except ImportError:
        pytest.skip("ibis-clickhouse not installed")

    return clickhouse_bin


def _create_clickhouse_directories(tmp_path_factory) -> dict[str, Any]:
    """Create required directories for ClickHouse server."""
    base_dir = tmp_path_factory.mktemp("clickhouse")
    dirs = {
        "data": base_dir / "data",
        "log": base_dir / "log",
        "tmp": base_dir / "tmp",
        "user_files": base_dir / "user_files",
        "format_schemas": base_dir / "format_schemas",
    }
    for d in dirs.values():
        d.mkdir()
    return dirs


def _start_clickhouse_process(
    clickhouse_bin: str,
    port: int,
    http_port: int,
    dirs: dict[str, Any],
) -> subprocess.Popen[bytes]:
    """Start the ClickHouse server process."""
    try:
        return subprocess.Popen(  # ty: ignore[no-matching-overload]
            [
                clickhouse_bin,
                "server",
                "--",
                f"--tcp_port={port}",
                f"--http_port={http_port}",
                f"--path={dirs['data']}/",
                f"--tmp_path={dirs['tmp']}/",
                f"--user_files_path={dirs['user_files']}/",
                f"--format_schema_path={dirs['format_schemas']}/",
                "--logger.console=1",
                "--logger.level=warning",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        pytest.skip(f"Failed to start ClickHouse server: {e}")


def _get_process_error(process: subprocess.Popen[bytes]) -> str:
    """Get error output from a process."""
    try:
        _, stderr = process.communicate(timeout=5)
        return stderr.decode()[:500]
    except Exception:
        return "Could not get error output"


def _wait_for_clickhouse_port(
    process: subprocess.Popen[bytes],
    port: int,
    max_retries: int = 30,
) -> None:
    """Wait for ClickHouse TCP port to be ready, or skip test on failure."""
    last_error: Exception | None = None

    for _ in range(max_retries):
        if process.poll() is not None:
            _, stderr = process.communicate()
            pytest.skip(
                f"ClickHouse server process terminated unexpectedly: {stderr.decode()[:500]}"
            )

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(("localhost", port))
                return  # Port is ready
        except (TimeoutError, ConnectionRefusedError, OSError) as e:
            last_error = e
            time.sleep(1)

    process.terminate()
    error_msg = _get_process_error(process)
    pytest.skip(
        f"ClickHouse server port not ready. Last error: {last_error}. Stderr: {error_msg}"
    )


def _verify_clickhouse_connection(
    process: subprocess.Popen[bytes],
    http_port: int,
) -> None:
    """Verify that ibis can connect to ClickHouse, or skip test on failure."""
    connection_string = f"clickhouse://localhost:{http_port}/default"
    try:
        conn: Any = ibis.connect(connection_string)
        conn.list_tables()
    except Exception as e:
        process.terminate()
        error_msg = _get_process_error(process)
        pytest.skip(f"ClickHouse Ibis connection failed: {e}. Stderr: {error_msg}")


def _terminate_clickhouse_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate the ClickHouse server process gracefully."""
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest.fixture(scope="session")
def clickhouse_server(tmp_path_factory):
    """Start a ClickHouse server for testing (session-scoped).

    Uses clickhouse binary to start a local server.
    Cleans up the process after all tests complete.

    Yields connection params (host, port) if ClickHouse is available, otherwise skips tests.
    """
    clickhouse_bin = _check_clickhouse_available()

    port = find_free_port()
    http_port = find_free_port()

    dirs = _create_clickhouse_directories(tmp_path_factory)
    process = _start_clickhouse_process(clickhouse_bin, port, http_port, dirs)

    _wait_for_clickhouse_port(process, port)
    _verify_clickhouse_connection(process, http_port)

    yield {"host": "localhost", "port": http_port}

    _terminate_clickhouse_process(process)


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

    conn.raw_sql(f"CREATE DATABASE {db_name}")  # ty: ignore[unresolved-attribute]
    test_conn_string = f"clickhouse://{host}:{port}/{db_name}"

    yield test_conn_string

    # Cleanup: drop test database
    try:
        conn.raw_sql(f"DROP DATABASE IF EXISTS {db_name}")  # ty: ignore[unresolved-attribute]
    except Exception:
        pass  # Best effort cleanup
