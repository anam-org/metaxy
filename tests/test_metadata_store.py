"""Tests for metadata store."""

from collections.abc import Iterator

import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    ContainerDep,
    ContainerKey,
    ContainerSpec,
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
)
from metaxy.metadata_store import (
    DependencyError,
    FeatureNotFoundError,
    InMemoryMetadataStore,
    MetadataSchemaError,
)

# Test fixtures - Define features for testing


class UpstreamFeatureA(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["upstream", "a"]),
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
        key=FeatureKey(["upstream", "b"]),
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
        key=FeatureKey(["downstream"]),
        deps=[
            FeatureDep(key=FeatureKey(["upstream", "a"])),
        ],
        containers=[
            ContainerSpec(
                key=ContainerKey(["default"]),
                code_version=1,
                deps=[
                    ContainerDep(
                        feature_key=FeatureKey(["upstream", "a"]),
                        containers=[ContainerKey(["frames"]), ContainerKey(["audio"])],
                    )
                ],
            ),
        ],
    ),
):
    pass


@pytest.fixture
def empty_store() -> Iterator[InMemoryMetadataStore]:
    """Empty in-memory store."""
    with InMemoryMetadataStore() as store:
        yield store


@pytest.fixture
def populated_store() -> Iterator[InMemoryMetadataStore]:
    """Store with sample upstream data."""
    with InMemoryMetadataStore() as store:
        # Add upstream feature A
        upstream_a_data = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "path": ["/data/1.mp4", "/data/2.mp4", "/data/3.mp4"],
                "data_version": [
                    {"frames": "hash_a1_frames", "audio": "hash_a1_audio"},
                    {"frames": "hash_a2_frames", "audio": "hash_a2_audio"},
                    {"frames": "hash_a3_frames", "audio": "hash_a3_audio"},
                ],
            }
        )
        store.write_metadata(UpstreamFeatureA, upstream_a_data)

        yield store


@pytest.fixture
def multi_env_stores() -> dict[str, InMemoryMetadataStore]:
    """Multi-environment store setup (prod, staging, dev).

    Note: Stores are not opened. Tests must use them with context managers.

    Example:
        with multi_env_stores['prod'] as prod:
            prod.write_metadata(...)
    """
    prod = InMemoryMetadataStore()
    staging = InMemoryMetadataStore(fallback_stores=[prod])
    dev = InMemoryMetadataStore(fallback_stores=[staging])

    # Populate prod with upstream data (requires opening)
    with prod:
        upstream_data = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "data_version": [
                    {"frames": "prod_hash1", "audio": "prod_hash1"},
                    {"frames": "prod_hash2", "audio": "prod_hash2"},
                    {"frames": "prod_hash3", "audio": "prod_hash3"},
                ],
            }
        )
        prod.write_metadata(UpstreamFeatureA, upstream_data)

    return {"prod": prod, "staging": staging, "dev": dev}


# Basic CRUD Tests


def test_write_and_read_metadata(empty_store: InMemoryMetadataStore) -> None:
    """Test basic write and read operations."""
    metadata = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [
                {"frames": "hash1", "audio": "hash1"},
                {"frames": "hash2", "audio": "hash2"},
                {"frames": "hash3", "audio": "hash3"},
            ],
        }
    )

    empty_store.write_metadata(UpstreamFeatureA, metadata)
    result = empty_store.read_metadata(UpstreamFeatureA)

    assert len(result) == 3
    assert "sample_id" in result.columns
    assert "data_version" in result.columns


def test_write_invalid_schema(empty_store: InMemoryMetadataStore) -> None:
    """Test that writing without data_version column raises error."""
    invalid_df = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "path": ["/a", "/b", "/c"],
        }
    )

    with pytest.raises(MetadataSchemaError, match="data_version"):
        empty_store.write_metadata(UpstreamFeatureA, invalid_df)


def test_write_append(empty_store: InMemoryMetadataStore) -> None:
    """Test that writes are append-only."""
    df1 = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [
                {"frames": "h1", "audio": "h1"},
                {"frames": "h2", "audio": "h2"},
            ],
        }
    )

    df2 = pl.DataFrame(
        {
            "sample_id": [3, 4],
            "data_version": [
                {"frames": "h3", "audio": "h3"},
                {"frames": "h4", "audio": "h4"},
            ],
        }
    )

    empty_store.write_metadata(UpstreamFeatureA, df1)
    empty_store.write_metadata(UpstreamFeatureA, df2)

    result = empty_store.read_metadata(UpstreamFeatureA)
    assert len(result) == 4
    assert set(result["sample_id"].to_list()) == {1, 2, 3, 4}


def test_read_with_filters(populated_store: InMemoryMetadataStore) -> None:
    """Test reading with Polars filter expressions."""
    result = populated_store.read_metadata(
        UpstreamFeatureA, filters=pl.col("sample_id") > 1
    )

    assert len(result) == 2
    assert set(result["sample_id"].to_list()) == {2, 3}


def test_read_with_column_selection(populated_store: InMemoryMetadataStore) -> None:
    """Test reading specific columns."""
    result = populated_store.read_metadata(
        UpstreamFeatureA, columns=["sample_id", "data_version"]
    )

    assert set(result.columns) == {"sample_id", "data_version"}
    assert "path" not in result.columns


def test_read_nonexistent_feature(empty_store: InMemoryMetadataStore) -> None:
    """Test that reading nonexistent feature raises error."""
    with pytest.raises(FeatureNotFoundError):
        empty_store.read_metadata(UpstreamFeatureA)


# Feature Existence Tests


def test_has_feature_local(populated_store: InMemoryMetadataStore) -> None:
    """Test has_feature for local store."""
    assert populated_store.has_feature(UpstreamFeatureA, check_fallback=False)
    assert not populated_store.has_feature(UpstreamFeatureB, check_fallback=False)


def test_has_feature_with_fallback(
    multi_env_stores: dict[str, InMemoryMetadataStore],
) -> None:
    """Test has_feature checking fallback stores."""
    with multi_env_stores["dev"] as dev:
        # UpstreamFeatureA is in prod, not in dev
        assert not dev.has_feature(UpstreamFeatureA, check_fallback=False)
        assert dev.has_feature(UpstreamFeatureA, check_fallback=True)


def test_list_features(populated_store: InMemoryMetadataStore) -> None:
    """Test listing features."""
    features = populated_store.list_features()

    assert len(features) == 1
    assert any(f.to_string() == "upstream_a" for f in features)


def test_list_features_with_fallback(
    multi_env_stores: dict[str, InMemoryMetadataStore],
) -> None:
    """Test listing features including fallbacks."""
    with multi_env_stores["dev"] as dev:
        # Without fallback
        local_features = dev.list_features(include_fallback=False)
        assert len(local_features) == 0

        # With fallback
        all_features = dev.list_features(include_fallback=True)
        assert len(all_features) == 1
        assert any(f.to_string() == "upstream_a" for f in all_features)


# Fallback Store Tests


def test_read_from_fallback(multi_env_stores: dict[str, InMemoryMetadataStore]) -> None:
    """Test reading from fallback store."""
    with multi_env_stores["dev"] as dev:
        # Read from prod via fallback chain
        result = dev.read_metadata(UpstreamFeatureA, allow_fallback=True)
        assert len(result) == 3


def test_read_no_fallback(multi_env_stores: dict[str, InMemoryMetadataStore]) -> None:
    """Test that allow_fallback=False doesn't check fallback stores."""
    with multi_env_stores["dev"] as dev:
        with pytest.raises(FeatureNotFoundError):
            dev.read_metadata(UpstreamFeatureA, allow_fallback=False)


def test_write_to_dev_not_prod(
    multi_env_stores: dict[str, InMemoryMetadataStore],
) -> None:
    """Test that writes go to dev, not prod."""
    with multi_env_stores["dev"] as dev, multi_env_stores["prod"] as prod:
        new_data = pl.DataFrame(
            {
                "sample_id": [4, 5],
                "data_version": [{"default": "hash4"}, {"default": "hash5"}],
            }
        )

        dev.write_metadata(UpstreamFeatureB, new_data)

        # Should be in dev
        assert dev.has_feature(UpstreamFeatureB, check_fallback=False)

        # Should NOT be in prod
        assert not prod.has_feature(UpstreamFeatureB, check_fallback=False)


# Dependency Resolution Tests


def test_read_upstream_metadata(populated_store: InMemoryMetadataStore) -> None:
    """Test reading upstream dependencies."""
    upstream = populated_store.read_upstream_metadata(DownstreamFeature)

    assert "upstream_a" in upstream
    assert len(upstream["upstream_a"]) == 3
    assert "data_version" in upstream["upstream_a"].columns


def test_read_upstream_metadata_missing_dep(empty_store: InMemoryMetadataStore) -> None:
    """Test that missing dependencies raise error."""
    with pytest.raises(DependencyError, match="upstream_a"):
        empty_store.read_upstream_metadata(DownstreamFeature, allow_fallback=False)


def test_read_upstream_metadata_from_fallback(
    multi_env_stores: dict[str, InMemoryMetadataStore],
) -> None:
    """Test reading upstream from fallback stores."""
    with multi_env_stores["dev"] as dev:
        upstream = dev.read_upstream_metadata(DownstreamFeature, allow_fallback=True)

        assert "upstream_a" in upstream
        assert len(upstream["upstream_a"]) == 3


# Data Version Calculation Tests


def test_calculate_and_write_data_versions(
    populated_store: InMemoryMetadataStore, snapshot: SnapshotAssertion
) -> None:
    """Test automatic data version calculation."""
    new_samples = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "custom_field": ["a", "b", "c"],
        }
    )

    result = populated_store.calculate_and_write_data_versions(
        feature=DownstreamFeature,
        sample_df=new_samples,
        allow_upstream_fallback=True,
    )

    # Should have data_version column
    assert "data_version" in result.columns

    # Should be a struct
    assert isinstance(result.schema["data_version"], pl.Struct)

    # Should have 'default' container
    data_version_sample = result["data_version"][0]
    assert "default" in data_version_sample

    # Should be written to store
    stored = populated_store.read_metadata(DownstreamFeature)
    assert len(stored) == 3

    # Snapshot the hash values for all samples
    hash_values = result["data_version"].struct.field("default").to_list()
    assert hash_values == snapshot


def test_calculate_deterministic(
    populated_store: InMemoryMetadataStore, snapshot: SnapshotAssertion
) -> None:
    """Test that data version calculation is deterministic."""
    samples = pl.DataFrame(
        {
            "sample_id": [1],
            "field": ["x"],
        }
    )

    result1 = populated_store.calculate_and_write_data_versions(
        feature=DownstreamFeature,
        sample_df=samples,
    )

    # Clear and recalculate
    populated_store.clear()

    # Re-add upstream
    upstream_a_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [
                {"frames": "hash_a1_frames", "audio": "hash_a1_audio"},
                {"frames": "hash_a2_frames", "audio": "hash_a2_audio"},
                {"frames": "hash_a3_frames", "audio": "hash_a3_audio"},
            ],
        }
    )
    populated_store.write_metadata(UpstreamFeatureA, upstream_a_data)

    result2 = populated_store.calculate_and_write_data_versions(
        feature=DownstreamFeature,
        sample_df=samples,
    )

    # Should be identical
    version1 = result1["data_version"][0]["default"]
    version2 = result2["data_version"][0]["default"]
    assert version1 == version2

    # Snapshot the deterministic hash value
    assert version1 == snapshot


def test_calculate_with_fallback_upstream(
    multi_env_stores: dict[str, InMemoryMetadataStore], snapshot: SnapshotAssertion
) -> None:
    """Test calculating data versions with upstream in fallback store."""
    with multi_env_stores["dev"] as dev, multi_env_stores["prod"] as prod:
        new_samples = pl.DataFrame(
            {
                "sample_id": [1, 2],
            }
        )

        # Should work - loads upstream from prod
        result = dev.calculate_and_write_data_versions(
            feature=DownstreamFeature,
            sample_df=new_samples,
            allow_upstream_fallback=True,
        )

        assert "data_version" in result.columns
        assert len(result) == 2

        # Should be written to dev only
        assert dev.has_feature(DownstreamFeature, check_fallback=False)
        assert not prod.has_feature(DownstreamFeature, check_fallback=False)

        # Snapshot the hash values
        hash_values = result["data_version"].struct.field("default").to_list()
        assert hash_values == snapshot


# Branch Deployment Scenario Test


def test_branch_deployment_workflow(
    multi_env_stores: dict[str, InMemoryMetadataStore], snapshot: SnapshotAssertion
) -> None:
    """
    Test complete branch deployment workflow.

    Scenario:
    - UpstreamFeatureA exists in prod
    - Testing DownstreamFeature in dev (depends on UpstreamFeatureA)
    - Dev should read UpstreamFeatureA from prod
    - Dev should write DownstreamFeature locally
    """
    with multi_env_stores["dev"] as dev, multi_env_stores["prod"] as prod:
        # 1. Verify upstream is in prod, not dev
        assert prod.has_feature(UpstreamFeatureA, check_fallback=False)
        assert not dev.has_feature(UpstreamFeatureA, check_fallback=False)

        # 2. Process downstream in dev (reads upstream from prod)
        new_samples = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
            }
        )

        result = dev.calculate_and_write_data_versions(
            feature=DownstreamFeature,
            sample_df=new_samples,
            allow_upstream_fallback=True,
        )

        # 3. Verify downstream is in dev, not prod
        assert dev.has_feature(DownstreamFeature, check_fallback=False)
        assert not prod.has_feature(DownstreamFeature, check_fallback=False)

        # 4. Verify we can read downstream from dev
        dev_downstream = dev.read_metadata(DownstreamFeature, allow_fallback=False)
        assert len(dev_downstream) == 3

        # 5. Promotion: Copy from dev to prod
        prod.write_metadata(DownstreamFeature, dev_downstream)

        # 6. Now both have it
        assert dev.has_feature(DownstreamFeature, check_fallback=False)
        assert prod.has_feature(DownstreamFeature, check_fallback=False)

        # Snapshot the hash values from the workflow
        hash_values = result["data_version"].struct.field("default").to_list()
        assert hash_values == snapshot


# Incremental Processing Test


def test_incremental_processing(populated_store: InMemoryMetadataStore) -> None:
    """Test only processing new samples."""
    # Initial batch
    batch1 = pl.DataFrame(
        {
            "sample_id": [1, 2],
        }
    )

    populated_store.calculate_and_write_data_versions(
        feature=DownstreamFeature,
        sample_df=batch1,
    )

    # Get existing IDs
    existing = populated_store.read_metadata(DownstreamFeature, columns=["sample_id"])
    existing_ids = set(existing["sample_id"].to_list())

    # All samples (new + old)
    all_samples = pl.DataFrame(
        {
            "sample_id": [1, 2, 3, 4],
        }
    )

    # Filter to only new
    new_samples = all_samples.filter(~pl.col("sample_id").is_in(existing_ids))

    assert len(new_samples) == 2
    assert set(new_samples["sample_id"].to_list()) == {3, 4}

    # Process only new
    populated_store.calculate_and_write_data_versions(
        feature=DownstreamFeature,
        sample_df=new_samples,
    )

    # Now should have all 4
    final = populated_store.read_metadata(DownstreamFeature)
    assert len(final) == 4


# Clear/Reset Tests


def test_clear_store(populated_store: InMemoryMetadataStore) -> None:
    """Test clearing store."""
    assert populated_store.has_feature(UpstreamFeatureA)

    populated_store.clear()

    assert not populated_store.has_feature(UpstreamFeatureA)
    assert len(populated_store.list_features()) == 0


def test_store_repr(empty_store: InMemoryMetadataStore) -> None:
    """Test string representation."""
    repr_str = repr(empty_store)

    assert "InMemoryMetadataStore" in repr_str
    assert "features=0" in repr_str
