from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest
from metaxy_testing import RAY_FEATURES_MODULE

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
DERIVED_FEATURE_KEY = ["test", "ray_derived"]


@pytest.fixture(autouse=True)
def reset_ray_features():
    """Reset Ray test features module to ensure fresh registration per test.

    When pytest resets the feature graph between tests, the feature class
    is NOT re-registered because Python's import system caches the module.
    This fixture clears the cache so the feature class is properly registered
    into the new graph when init_metaxy loads entrypoints.
    """

    # Remove the features module from cache before the test
    if RAY_FEATURES_MODULE in sys.modules:
        del sys.modules[RAY_FEATURES_MODULE]
    yield
    # Clean up after test
    if RAY_FEATURES_MODULE in sys.modules:
        del sys.modules[RAY_FEATURES_MODULE]


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
    # https://github.com/ray-project/ray/issues/60453#issuecomment-3791399501
    ray.data.DataContext.get_current()._enable_actor_pool_on_exit_hook = True
    yield
    ray.shutdown()


@pytest.fixture
def delta_store(tmp_path: Path) -> DeltaMetadataStore:
    """Create a Delta Lake metadata store for testing."""
    return DeltaMetadataStore(root_path=tmp_path / "delta")


@pytest.fixture
def ray_config(tmp_path: Path, delta_store: DeltaMetadataStore) -> mx.MetaxyConfig:
    """Create a MetaxyConfig with entrypoints pointing to Ray test features."""
    return mx.MetaxyConfig(
        project="test",
        entrypoints=[RAY_FEATURES_MODULE],
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
