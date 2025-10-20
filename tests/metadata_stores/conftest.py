"""Common fixtures for metadata store tests."""

from pathlib import Path

import pytest
from pytest_cases import fixture, parametrize_with_cases

from metaxy import (
    ContainerDep,
    ContainerKey,
    ContainerSpec,
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
)
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store import InMemoryMetadataStore, MetadataStore
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.models.feature import FeatureRegistry


@pytest.fixture
def test_registry():
    """Create a clean FeatureRegistry for testing with test features registered.

    Returns a tuple of (registry, features_dict) where features_dict provides
    easy access to feature classes by simple names.
    """
    with FeatureRegistry().use() as registry:
        # Define features within the registry context
        class UpstreamFeatureA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test_stores", "upstream_a"]),
                deps=None,
                containers=[
                    ContainerSpec(key=ContainerKey(["frames"]), code_version=1),
                    ContainerSpec(key=ContainerKey(["audio"]), code_version=1),
                ],
            ),
        ):
            pass

        class UpstreamFeatureB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test_stores", "upstream_b"]),
                deps=None,
                containers=[
                    ContainerSpec(key=ContainerKey(["default"]), code_version=1),
                ],
            ),
        ):
            pass

        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test_stores", "downstream"]),
                deps=[
                    FeatureDep(key=FeatureKey(["test_stores", "upstream_a"])),
                ],
                containers=[
                    ContainerSpec(
                        key=ContainerKey(["default"]),
                        code_version=1,
                        deps=[
                            ContainerDep(
                                feature_key=FeatureKey(["test_stores", "upstream_a"]),
                                containers=[
                                    ContainerKey(["frames"]),
                                    ContainerKey(["audio"]),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ):
            pass

        # Create features dict for easy access
        features = {
            "UpstreamFeatureA": UpstreamFeatureA,
            "UpstreamFeatureB": UpstreamFeatureB,
            "DownstreamFeature": DownstreamFeature,
        }

        yield registry, features


@pytest.fixture
def test_features(test_registry):
    """Provide dict of test feature classes for easy access in tests.

    This fixture extracts just the features dict from test_registry for convenience.
    """
    _, features = test_registry
    return features


# Store case functions for pytest-cases


class StoreCases:
    """Store configuration cases for parametrization."""

    def case_inmemory(
        self, test_registry: FeatureRegistry
    ) -> tuple[type[MetadataStore], dict]:
        """InMemory store case."""
        # Registry is accessed globally via FeatureRegistry.get_active()
        return (InMemoryMetadataStore, {})

    def case_duckdb(
        self, tmp_path: Path, test_registry: FeatureRegistry
    ) -> tuple[type[MetadataStore], dict]:
        """DuckDB store case."""
        db_path = tmp_path / "test.duckdb"
        # Registry is accessed globally via FeatureRegistry.get_active()
        return (DuckDBMetadataStore, {"database": db_path})


@fixture
@parametrize_with_cases("store_config", cases=StoreCases)
def persistent_store(store_config: tuple[type[MetadataStore], dict]) -> MetadataStore:
    """Parametrized persistent store fixture.

    This fixture runs tests for all persistent store implementations.
    Returns an unopened store - tests should use it with a context manager.

    Usage:
        def test_something(persistent_store, test_registry):
            with persistent_store as store:
                # Test code runs for all store types
                # Access feature classes via test_registry.UpstreamFeatureA, etc.
    """
    store_type, config = store_config
    return store_type(**config)  # type: ignore[abstract]


class HashAlgorithmCases:
    """Test cases for different hash algorithms."""

    def case_xxhash64(self) -> HashAlgorithm:
        """xxHash64 algorithm."""
        return HashAlgorithm.XXHASH64

    def case_xxhash32(self) -> HashAlgorithm:
        """xxHash32 algorithm."""
        return HashAlgorithm.XXHASH32

    def case_wyhash(self) -> HashAlgorithm:
        """WyHash algorithm."""
        return HashAlgorithm.WYHASH

    def case_sha256(self) -> HashAlgorithm:
        """SHA256 algorithm."""
        return HashAlgorithm.SHA256

    def case_md5(self) -> HashAlgorithm:
        """MD5 algorithm."""
        return HashAlgorithm.MD5
