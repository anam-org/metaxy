"""Tests for Ray Data actor integration with Metaxy."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

import metaxy as mx
from metaxy.config import StoreConfig
from metaxy.metadata_store.delta import DeltaMetadataStore

if TYPE_CHECKING:
    import ray
else:
    ray = pytest.importorskip("ray")

# NOTE: Do NOT import RayTestFeature at module level!
# It must be imported AFTER init_metaxy() to pick up the correct project.

FEATURE_KEY = ["test", "ray_feature"]


@pytest.fixture(autouse=True)
def reset_ray_features():
    """Reset Ray test features module to ensure fresh registration per test.

    When pytest resets the feature graph between tests, the feature class
    is NOT re-registered because Python's import system caches the module.
    This fixture clears the cache so the feature class is properly registered
    into the new graph when init_metaxy loads entrypoints.
    """

    # Remove the features module from cache before the test
    features_module = "metaxy._testing.ray_features"
    if features_module in sys.modules:
        del sys.modules[features_module]
    yield
    # Clean up after test
    if features_module in sys.modules:
        del sys.modules[features_module]


@pytest.fixture
def ray_context():
    """Initialize Ray for testing.

    Uses /tmp/ray as the temp directory to ensure Ray's session directory is in
    a stable location that won't be cleaned up prematurely (important for CI
    environments like Nix shells where the default temp directory may be
    ephemeral). We use /tmp/ray directly instead of pytest's tmp_path because
    Ray creates deeply nested paths for Unix sockets that can exceed the 107
    byte AF_UNIX path limit.
    """
    import ray

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=2)
    yield
    ray.shutdown()


@pytest.fixture
def delta_store(tmp_path: Path) -> DeltaMetadataStore:
    """Create a Delta Lake metadata store for testing."""
    return DeltaMetadataStore(root_path=tmp_path / "delta")


@pytest.fixture
def ray_config(tmp_path: Path, delta_store: DeltaMetadataStore) -> mx.MetaxyConfig:
    """Create a MetaxyConfig with entrypoints pointing to Ray test features."""
    features_module = "metaxy._testing.ray_features"

    return mx.MetaxyConfig(
        project="test",
        entrypoints=[features_module],
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.delta.DeltaMetadataStore",
                config={"root_path": str(delta_store._root_uri)},
            )
        },
    )


def make_test_data(sample_uids: list[str], values: list[int]) -> pl.DataFrame:
    """Create test data with required Metaxy system columns.

    For Ray workers that write metadata directly (without resolve_update),
    we must provide all required columns:
    - metaxy_provenance_by_field: per-field provenance hashes
    - metaxy_provenance: combined provenance hash (to avoid graph lookup)
    """
    provenance_by_field = [{"value": f"hash_{v}"} for v in values]
    provenance = [f"combined_hash_{v}" for v in values]

    return pl.DataFrame(
        {
            "sample_uid": sample_uids,
            "value": values,
            "metaxy_provenance_by_field": provenance_by_field,
            "metaxy_provenance": provenance,
        }
    )


@pytest.fixture
def test_data() -> pl.DataFrame:
    """Create test dataset with 5 samples."""
    return make_test_data(
        sample_uids=["a", "b", "c", "d", "e"],
        values=[1, 2, 3, 4, 5],
    )


def test_batched_metadata_writer_actor_with_map_batches_pyarrow(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test BatchedMetadataWriterActor with map_batches and PyArrow batch format."""
    from metaxy.ext.ray.actors import BatchedMetadataWriterActor

    mx.init_metaxy(ray_config)

    ds = ray.data.from_arrow(test_data.to_arrow())

    ds.map_batches(
        BatchedMetadataWriterActor,
        fn_constructor_kwargs={
            "feature": FEATURE_KEY,
            "store": delta_store,
            "metaxy_config": ray_config,
            "accept_batches": True,
        },
        concurrency=1,
        batch_format="pyarrow",
    ).materialize()

    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()

        assert len(df) == 5
        assert set(df["sample_uid"].to_list()) == {"a", "b", "c", "d", "e"}
        assert set(df["value"].to_list()) == {1, 2, 3, 4, 5}
        assert "metaxy_provenance_by_field" in df.columns


def test_batched_metadata_writer_actor_passthrough(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test that BatchedMetadataWriterActor passes through data unchanged."""
    from metaxy.ext.ray.actors import BatchedMetadataWriterActor

    mx.init_metaxy(ray_config)

    ds = ray.data.from_arrow(test_data.to_arrow())

    result_ds = ds.map_batches(
        BatchedMetadataWriterActor,
        fn_constructor_kwargs={
            "feature": FEATURE_KEY,
            "store": delta_store,
            "metaxy_config": ray_config,
            "accept_batches": True,
        },
        concurrency=1,
        batch_format="pyarrow",
    )

    result_df = pl.DataFrame(result_ds.materialize().take_all())

    assert len(result_df) == 5
    assert set(result_df["sample_uid"].to_list()) == {"a", "b", "c", "d", "e"}
    assert set(result_df["value"].to_list()) == {1, 2, 3, 4, 5}


def test_batched_metadata_writer_actor_auto_config(
    ray_context,
    test_data: pl.DataFrame,
    tmp_path: Path,
):
    """Test that MetaxyConfig is auto-loaded from METAXY_CONFIG env var."""
    from metaxy.ext.ray.actors import BatchedMetadataWriterActor

    # Create a fresh delta store for this test
    delta_root = tmp_path / "delta_auto"
    delta_store = DeltaMetadataStore(root_path=delta_root)

    # Write config to a TOML file
    config_path = tmp_path / "metaxy.toml"
    config_content = f'''project = "test"
store = "dev"
auto_create_tables = true
entrypoints = ["metaxy._testing.ray_features"]

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "{delta_root}"
'''
    config_path.write_text(config_content)

    ds = ray.data.from_arrow(test_data.to_arrow())

    # Use runtime_env to set METAXY_CONFIG for the Ray worker
    ds.map_batches(
        BatchedMetadataWriterActor,
        fn_constructor_kwargs={
            "feature": FEATURE_KEY,
            "store": delta_store,
            "accept_batches": True,
            # Note: metaxy_config is NOT passed - should be auto-loaded
        },
        concurrency=1,
        runtime_env={"env_vars": {"METAXY_CONFIG": str(config_path)}},
        batch_format="pyarrow",
    ).materialize()

    # Initialize metaxy in test process to read metadata
    # (The actor auto-loaded it from METAXY_CONFIG, but we need it here too)
    mx.init_metaxy(mx.MetaxyConfig.load(config_path))

    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()
        assert len(df) == 5


def test_batched_metadata_writer_actor_flushes_on_shutdown(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
):
    """Test that pending metadata is flushed when Ray actor shuts down.

    Uses a very long flush_interval to ensure data stays pending,
    then verifies shutdown triggers the flush.
    """
    from metaxy.ext.ray.actors import BatchedMetadataWriterActor

    mx.init_metaxy(ray_config)

    ds = ray.data.from_arrow(test_data.to_arrow())

    ds.map_batches(
        BatchedMetadataWriterActor,
        fn_constructor_kwargs={
            "feature": FEATURE_KEY,
            "store": delta_store,
            "metaxy_config": ray_config,
            "accept_batches": True,
            "flush_interval": 60.0,  # Long interval - data won't auto-flush
        },
        concurrency=1,
        batch_format="pyarrow",
    ).materialize()

    # After materialize(), Ray tears down the actor, triggering __ray_shutdown__
    # which should flush all pending data

    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()

        assert len(df) == 5, "All 5 records should be flushed on shutdown"
        assert set(df["sample_uid"].to_list()) == {"a", "b", "c", "d", "e"}


@pytest.mark.parametrize(
    "feature_param",
    [
        pytest.param(FEATURE_KEY, id="feature_key_list"),
        pytest.param("test/ray_feature", id="feature_key_string"),
    ],
)
def test_batched_metadata_writer_actor_feature_formats(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
    feature_param,
):
    """Test that feature can be passed as list or string path."""
    from metaxy.ext.ray.actors import BatchedMetadataWriterActor

    mx.init_metaxy(ray_config)

    ds = ray.data.from_arrow(test_data.to_arrow())

    ds.map_batches(
        BatchedMetadataWriterActor,
        fn_constructor_kwargs={
            "feature": feature_param,
            "store": delta_store,
            "metaxy_config": ray_config,
            "accept_batches": True,
        },
        concurrency=1,
        batch_format="pyarrow",
    ).materialize()

    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()
        assert len(df) == 5


@pytest.mark.parametrize(
    "batch_format",
    [
        pytest.param("pyarrow", id="pyarrow"),
        pytest.param("numpy", id="numpy"),
        pytest.param("pandas", id="pandas"),
    ],
)
def test_batched_metadata_writer_actor_batch_formats(
    ray_config: mx.MetaxyConfig,
    ray_context,
    delta_store: DeltaMetadataStore,
    test_data: pl.DataFrame,
    batch_format: str,
):
    """Test BatchedMetadataWriterActor with different batch_format options for map_batches."""
    from metaxy.ext.ray.actors import BatchedMetadataWriterActor

    mx.init_metaxy(ray_config)

    ds = ray.data.from_arrow(test_data.to_arrow())

    ds.map_batches(
        BatchedMetadataWriterActor,
        fn_constructor_kwargs={
            "feature": FEATURE_KEY,
            "store": delta_store,
            "metaxy_config": ray_config,
            "accept_batches": True,
        },
        concurrency=1,
        batch_format=batch_format,
    ).materialize()

    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()

        assert len(df) == 5
        assert set(df["sample_uid"].to_list()) == {"a", "b", "c", "d", "e"}
        assert set(df["value"].to_list()) == {1, 2, 3, 4, 5}


def test_batched_metadata_writer_actor_with_map(
    ray_config: mx.MetaxyConfig,
    delta_store: DeltaMetadataStore,
):
    """Test BatchedMetadataWriterActor with accept_batches=False (default) for map input.

    This tests the code path for Dataset.map() which passes single rows as dicts
    with scalar values. We test directly without Ray to avoid resource allocation issues.
    """
    from metaxy.ext.ray.actors import BatchedMetadataWriterActor

    mx.init_metaxy(ray_config)

    # Create actor with accept_batches=False (default) for scalar dict input
    actor = BatchedMetadataWriterActor(
        feature=FEATURE_KEY,
        store=delta_store,
        metaxy_config=ray_config,
        # accept_batches=False is the default
    )

    # Test with scalar dict input (simulating what .map() sends)
    test_rows = [
        {
            "sample_uid": "a",
            "value": 1,
            "metaxy_provenance_by_field": {"value": "hash_1"},
            "metaxy_provenance": "combined_hash_1",
        },
        {
            "sample_uid": "b",
            "value": 2,
            "metaxy_provenance_by_field": {"value": "hash_2"},
            "metaxy_provenance": "combined_hash_2",
        },
    ]

    for row in test_rows:
        result = actor(row)
        # Should return the same dict
        assert result == row

    # Stop the writer to flush data
    actor.stop()

    # Verify data was written
    with delta_store.open("read"):
        result = delta_store.read_metadata(FEATURE_KEY)
        assert result is not None
        df = result.collect()

        assert len(df) == 2
        assert set(df["sample_uid"].to_list()) == {"a", "b"}
