"""Tests for feature-level data version calculation."""

from syrupy.assertion import SnapshotAssertion

from metaxy import ContainerSpec, Feature, FeatureKey, FeatureSpec
from metaxy.models.types import ContainerKey


def test_single_feature_data_version(snapshot: SnapshotAssertion) -> None:
    """Test data version for a simple feature with default container."""

    class MyFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["my_feature"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["default"]), code_version=1),
            ],
        ),
    ):
        pass

    data_version = MyFeature.data_version()

    # Should return a dict with string keys
    assert isinstance(data_version, dict)
    assert len(data_version) == 1
    assert "default" in data_version
    assert isinstance(data_version["default"], str)
    assert len(data_version["default"]) > 0

    # Snapshot the hash value
    assert data_version == snapshot


def test_multi_container_feature_data_version(snapshot: SnapshotAssertion) -> None:
    """Test data version for a feature with multiple containers."""

    class MultiContainerFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["multi", "container"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["frames"]), code_version=1),
                ContainerSpec(key=ContainerKey(["audio"]), code_version=1),
                ContainerSpec(key=ContainerKey(["metadata"]), code_version=2),
            ],
        ),
    ):
        pass

    data_version = MultiContainerFeature.data_version()

    # Should have all three containers
    assert set(data_version.keys()) == {"frames", "audio", "metadata"}

    # All should be non-empty strings
    for container, hash_val in data_version.items():
        assert isinstance(hash_val, str)
        assert len(hash_val) > 0

    # Different code versions should produce different hashes
    # (metadata has code_version=2, others have code_version=1)
    assert data_version["frames"] != data_version["metadata"]

    # Snapshot all hash values
    assert data_version == snapshot


def test_feature_code_version_change(snapshot: SnapshotAssertion) -> None:
    """Test that changing code version changes the data version hash."""

    class FeatureV1(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["versioned", "feature", "v1"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["default"]), code_version=1),
            ],
        ),
    ):
        pass

    class FeatureV2(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["versioned", "feature", "v2"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["default"]), code_version=2),
            ],
        ),
    ):
        pass

    v1_data_version = FeatureV1.data_version()
    v2_data_version = FeatureV2.data_version()

    # Different code versions should produce different hashes
    assert v1_data_version["default"] != v2_data_version["default"]

    # Snapshot both versions
    assert {"v1": v1_data_version, "v2": v2_data_version} == snapshot
