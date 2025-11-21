"""Tests for Dagster IOManager parallelization patterns."""

from __future__ import annotations

from unittest.mock import Mock

import dagster as dg
import narwhals as nw
import polars as pl
import pytest

from metaxy import (
    BaseFeature,
    FeatureKey,
    FieldKey,
    FieldSpec,
    MetaxyConfig,
)
from metaxy._testing.models import SampleFeatureSpec
from metaxy.config import StoreConfig
from metaxy.ext.dagster import (
    MetaxyIOManager,
    MetaxyMetadataStoreResource,
    apply_increment_filter,
    build_partitioned_asset_spec_from_feature,
    filter_increment_by_partition,
    limit_increment,
)
from metaxy.versioning.types import Increment

# ============= HELPER FUNCTION TESTS =============


def test_filter_increment_by_partition_basic() -> None:
    """Test filtering an Increment to a specific partition value."""
    # Create test data with a partition column
    added_df = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v1", "v2", "v2"],
                "frame_id": [1, 2, 1, 2],
                "value": [10, 20, 30, 40],
            }
        )
    )
    changed_df = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "frame_id": [3, 3],
                "value": [50, 60],
            }
        )
    )
    removed_df = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1"],
                "frame_id": [4],
                "value": [70],
            }
        )
    )

    increment = Increment(added=added_df, changed=changed_df, removed=removed_df)

    # Filter to v1
    filtered = filter_increment_by_partition(increment, "video_id", "v1")

    # Check added (should have 2 rows for v1)
    assert len(filtered.added) == 2
    assert set(filtered.added.to_native()["frame_id"].to_list()) == {1, 2}

    # Check changed (should have 1 row for v1)
    assert len(filtered.changed) == 1
    assert filtered.changed.to_native()["frame_id"].to_list() == [3]

    # Check removed (should have 1 row for v1)
    assert len(filtered.removed) == 1
    assert filtered.removed.to_native()["frame_id"].to_list() == [4]


def test_filter_increment_by_partition_missing_column() -> None:
    """Test error when partition column doesn't exist."""
    added_df = nw.from_native(
        pl.DataFrame(
            {
                "frame_id": [1, 2],
                "value": [10, 20],
            }
        )
    )
    increment = Increment(
        added=added_df,
        changed=nw.from_native(pl.DataFrame({"frame_id": [], "value": []})),
        removed=nw.from_native(pl.DataFrame({"frame_id": [], "value": []})),
    )

    with pytest.raises(ValueError, match="Partition column 'video_id' not found"):
        filter_increment_by_partition(increment, "video_id", "v1")


def test_filter_increment_by_partition_empty_dataframes() -> None:
    """Test filtering empty DataFrames doesn't error."""
    empty_df = nw.from_native(
        pl.DataFrame(
            {
                "video_id": [],
                "frame_id": [],
                "value": [],
            }
        )
    )
    increment = Increment(added=empty_df, changed=empty_df, removed=empty_df)

    # Should not raise
    filtered = filter_increment_by_partition(increment, "video_id", "v1")

    assert len(filtered.added) == 0
    assert len(filtered.changed) == 0
    assert len(filtered.removed) == 0


def test_limit_increment_basic() -> None:
    """Test limiting the number of samples in an Increment."""
    added_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_id": [1, 2, 3, 4, 5],
                "value": [10, 20, 30, 40, 50],
            }
        )
    )
    changed_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_id": [6, 7, 8],
                "value": [60, 70, 80],
            }
        )
    )
    removed_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_id": [9, 10],
                "value": [90, 100],
            }
        )
    )

    increment = Increment(added=added_df, changed=changed_df, removed=removed_df)

    # Limit to 2 samples
    limited = limit_increment(increment, limit=2)

    # Check all DataFrames are limited
    assert len(limited.added) == 2
    assert len(limited.changed) == 2
    assert len(limited.removed) == 2

    # Check we got the first rows
    assert limited.added.to_native()["sample_id"].to_list() == [1, 2]
    assert limited.changed.to_native()["sample_id"].to_list() == [6, 7]
    assert limited.removed.to_native()["sample_id"].to_list() == [9, 10]


def test_limit_increment_larger_than_data() -> None:
    """Test limiting with a larger limit than data size."""
    added_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_id": [1, 2],
                "value": [10, 20],
            }
        )
    )
    increment = Increment(
        added=added_df,
        changed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
        removed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
    )

    # Limit to 10 samples (more than we have)
    limited = limit_increment(increment, limit=10)

    # Should return all data
    assert len(limited.added) == 2


def test_apply_increment_filter_basic() -> None:
    """Test applying a custom filter expression."""
    added_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_id": [1, 2, 3, 4],
                "confidence": [0.5, 0.9, 0.7, 0.95],
            }
        )
    )
    changed_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_id": [5, 6],
                "confidence": [0.6, 0.92],
            }
        )
    )
    increment = Increment(
        added=added_df,
        changed=changed_df,
        removed=nw.from_native(pl.DataFrame({"sample_id": [], "confidence": []})),
    )

    # Filter to high confidence samples
    filtered = apply_increment_filter(increment, nw.col("confidence") > 0.8)

    # Check filtering worked
    assert len(filtered.added) == 2
    assert set(filtered.added.to_native()["sample_id"].to_list()) == {2, 4}

    assert len(filtered.changed) == 1
    assert filtered.changed.to_native()["sample_id"].to_list() == [6]


# ============= IOManager CONFIGURATION TESTS =============


def test_io_manager_default_no_filters() -> None:
    """Test IOManager with no filters (data parallel pattern)."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(store_resource)

        # Verify default configuration
        assert io_manager.partition_key_column is None
        assert io_manager.sample_limit is None
        assert io_manager.sample_filter_expr is None
    finally:
        MetaxyConfig.reset()


def test_io_manager_with_partition_key_column() -> None:
    """Test IOManager configured for event parallel pattern."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            partition_key_column="video_id",
        )

        # Verify configuration
        assert io_manager.partition_key_column == "video_id"
        assert io_manager.sample_limit is None
        assert io_manager.sample_filter_expr is None
    finally:
        MetaxyConfig.reset()


def test_io_manager_with_sample_limit() -> None:
    """Test IOManager configured for subsampling."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            sample_limit=10,
        )

        # Verify configuration
        assert io_manager.partition_key_column is None
        assert io_manager.sample_limit == 10
        assert io_manager.sample_filter_expr is None
    finally:
        MetaxyConfig.reset()


def test_io_manager_with_filter_expr() -> None:
    """Test IOManager configured with custom filter expression."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            sample_filter_expr='nw.col("confidence") > 0.9',
        )

        # Verify configuration
        assert io_manager.partition_key_column is None
        assert io_manager.sample_limit is None
        assert isinstance(io_manager.sample_filter_expr, str)
        assert io_manager.sample_filter_expr == 'nw.col("confidence") > 0.9'
    finally:
        MetaxyConfig.reset()


# ============= FILTER APPLICATION TESTS =============


def test_apply_parallelization_filters_no_filters() -> None:
    """Test _apply_parallelization_filters with no filters configured."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(store_resource)

        # Create test increment
        added_df = nw.from_native(
            pl.DataFrame(
                {
                    "sample_id": [1, 2, 3],
                    "value": [10, 20, 30],
                }
            )
        )
        increment = Increment(
            added=added_df,
            changed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
            removed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
        )

        # Create mock context
        context = Mock(spec=dg.InputContext)

        # Apply filters (should be no-op)
        filtered = io_manager._apply_parallelization_filters(context, increment)

        # Should be unchanged
        assert len(filtered.added) == 3
    finally:
        MetaxyConfig.reset()


def test_apply_parallelization_filters_partition() -> None:
    """Test _apply_parallelization_filters with partition filtering."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            partition_key_column="video_id",
        )

        # Create test increment
        added_df = nw.from_native(
            pl.DataFrame(
                {
                    "video_id": ["v1", "v1", "v2"],
                    "sample_id": [1, 2, 3],
                    "value": [10, 20, 30],
                }
            )
        )
        increment = Increment(
            added=added_df,
            changed=nw.from_native(
                pl.DataFrame({"video_id": [], "sample_id": [], "value": []})
            ),
            removed=nw.from_native(
                pl.DataFrame({"video_id": [], "sample_id": [], "value": []})
            ),
        )

        # Create mock context with partition
        context = Mock(spec=dg.InputContext)
        context.partition_key = "v1"

        # Apply filters
        filtered = io_manager._apply_parallelization_filters(context, increment)

        # Should only have v1 data
        assert len(filtered.added) == 2
        assert set(filtered.added.to_native()["sample_id"].to_list()) == {1, 2}
    finally:
        MetaxyConfig.reset()


def test_apply_parallelization_filters_limit() -> None:
    """Test _apply_parallelization_filters with sample limit."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            sample_limit=2,
        )

        # Create test increment
        added_df = nw.from_native(
            pl.DataFrame(
                {
                    "sample_id": [1, 2, 3, 4, 5],
                    "value": [10, 20, 30, 40, 50],
                }
            )
        )
        increment = Increment(
            added=added_df,
            changed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
            removed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
        )

        # Create mock context
        context = Mock(spec=dg.InputContext)

        # Apply filters
        filtered = io_manager._apply_parallelization_filters(context, increment)

        # Should only have 2 samples
        assert len(filtered.added) == 2
        assert filtered.added.to_native()["sample_id"].to_list() == [1, 2]
    finally:
        MetaxyConfig.reset()


def test_apply_parallelization_filters_filter_expr() -> None:
    """Test _apply_parallelization_filters with a safe filter expression string."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            sample_filter_expr='nw.col("value") > 10',
        )

        added_df = nw.from_native(
            pl.DataFrame(
                {
                    "sample_id": [1, 2, 3],
                    "value": [5, 15, 20],
                }
            )
        )
        increment = Increment(
            added=added_df,
            changed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
            removed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
        )

        context = Mock(spec=dg.InputContext)

        filtered = io_manager._apply_parallelization_filters(context, increment)

        assert len(filtered.added) == 2
        assert filtered.added.to_native()["sample_id"].to_list() == [2, 3]
    finally:
        MetaxyConfig.reset()


def test_apply_parallelization_filters_filter_expr_rejects_unsafe() -> None:
    """Unsafe filter expressions are rejected before evaluation."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            sample_filter_expr="__import__('os').system('echo hacked')",
        )

        increment = Increment(
            added=nw.from_native(pl.DataFrame({"sample_id": [1], "value": [10]})),
            changed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
            removed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
        )

        context = Mock(spec=dg.InputContext)

        with pytest.raises(ValueError):
            io_manager._apply_parallelization_filters(context, increment)
    finally:
        MetaxyConfig.reset()


def test_apply_parallelization_filters_target_keys_from_manager() -> None:
    """Target key filtering via IOManager configuration."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            target_key_column="video_id",
            target_keys=["v1", "v3"],
        )

        increment = Increment(
            added=nw.from_native(
                pl.DataFrame({"video_id": ["v1", "v2", "v3"], "value": [1, 2, 3]})
            ),
            changed=nw.from_native(pl.DataFrame({"video_id": [], "value": []})),
            removed=nw.from_native(pl.DataFrame({"video_id": [], "value": []})),
        )

        context = Mock(spec=dg.InputContext)
        filtered = io_manager._apply_parallelization_filters(context, increment)

        assert filtered.added.to_native()["video_id"].to_list() == ["v1", "v3"]
    finally:
        MetaxyConfig.reset()


def test_apply_parallelization_filters_target_keys_from_metadata() -> None:
    """Asset metadata can override IOManager key filtering."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            target_key_column="video_id",
            target_keys=["v1"],
        )

        increment = Increment(
            added=nw.from_native(
                pl.DataFrame({"video_id": ["v1", "v2"], "value": [1, 2]})
            ),
            changed=nw.from_native(pl.DataFrame({"video_id": [], "value": []})),
            removed=nw.from_native(pl.DataFrame({"video_id": [], "value": []})),
        )

        context = Mock(spec=dg.InputContext)
        context.metadata = {
            "metaxy/target_keys": ["v2"],
            "metaxy/target_key_column": "video_id",
        }

        filtered = io_manager._apply_parallelization_filters(context, increment)

        assert filtered.added.to_native()["video_id"].to_list() == ["v2"]
    finally:
        MetaxyConfig.reset()


def test_apply_parallelization_filters_target_keys_missing_column() -> None:
    """Missing target column results in a clear error."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            target_key_column="video_id",
            target_keys=["v1"],
        )

        increment = Increment(
            added=nw.from_native(pl.DataFrame({"sample_id": [1], "value": [10]})),
            changed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
            removed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
        )

        context = Mock(spec=dg.InputContext)

        with pytest.raises(ValueError, match="Target key column 'video_id' not found"):
            io_manager._apply_parallelization_filters(context, increment)
    finally:
        MetaxyConfig.reset()


def test_apply_parallelization_filters_combined() -> None:
    """Test _apply_parallelization_filters with partition and limit."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            partition_key_column="video_id",
            sample_limit=1,
        )

        # Create test increment
        added_df = nw.from_native(
            pl.DataFrame(
                {
                    "video_id": ["v1", "v1", "v2"],
                    "sample_id": [1, 2, 3],
                    "value": [10, 20, 30],
                }
            )
        )
        increment = Increment(
            added=added_df,
            changed=nw.from_native(
                pl.DataFrame({"video_id": [], "sample_id": [], "value": []})
            ),
            removed=nw.from_native(
                pl.DataFrame({"video_id": [], "sample_id": [], "value": []})
            ),
        )

        # Create mock context with partition
        context = Mock(spec=dg.InputContext)
        context.partition_key = "v1"

        # Apply filters
        filtered = io_manager._apply_parallelization_filters(context, increment)

        # Should only have 1 sample from v1
        assert len(filtered.added) == 1
        assert filtered.added.to_native()["sample_id"].to_list() == [1]
    finally:
        MetaxyConfig.reset()


def test_apply_parallelization_filters_partition_missing() -> None:
    """Test error when partition_key_column is set but partition_key is missing."""
    config = MetaxyConfig(
        project="test",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="dev",
    )
    MetaxyConfig.set(config)

    try:
        store_resource = MetaxyMetadataStoreResource.from_config(store_name="dev")
        io_manager = MetaxyIOManager.from_store(
            store_resource,
            partition_key_column="video_id",
        )

        # Create test increment
        increment = Increment(
            added=nw.from_native(pl.DataFrame({"sample_id": [1], "value": [10]})),
            changed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
            removed=nw.from_native(pl.DataFrame({"sample_id": [], "value": []})),
        )

        # Create mock context WITHOUT partition
        context = Mock(spec=dg.InputContext)
        context.partition_key = None

        # Should raise error
        with pytest.raises(
            ValueError,
            match="IOManager configured with partition_key_column.*but context has no partition_key",
        ):
            io_manager._apply_parallelization_filters(context, increment)
    finally:
        MetaxyConfig.reset()


# ============= PARTITIONED ASSET SPEC TESTS =============


def test_build_partitioned_asset_spec_from_feature() -> None:
    """Test building a partitioned AssetSpec from a feature."""

    class SimpleFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["simple", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Create partitions
    partitions = dg.DynamicPartitionsDefinition(name="test_partitions")

    # Build spec
    spec = build_partitioned_asset_spec_from_feature(SimpleFeature, partitions)

    # Verify asset key
    assert spec.key == dg.AssetKey(["simple", "feature"])

    # Verify partitions are set
    assert spec.partitions_def is partitions

    # Verify metadata is preserved
    assert spec.metadata is not None
    assert spec.metadata["feature_key"] == "simple/feature"


# ============= NEW HELPER FUNCTION TESTS =============


def test_filter_samples_by_partition_basic() -> None:
    """Test filtering samples DataFrame to a specific partition value."""
    from metaxy.ext.dagster import filter_samples_by_partition

    # Create test samples with a partition column
    samples_df = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v1", "v2", "v2", "v3"],
                "frame_id": [1, 2, 1, 2, 1],
                "value": [10, 20, 30, 40, 50],
            }
        )
    )

    # Filter to v1
    filtered = filter_samples_by_partition(samples_df, "video_id", "v1")

    # Check we got only v1 rows
    assert len(filtered) == 2
    assert set(filtered.to_native()["frame_id"].to_list()) == {1, 2}
    assert all(vid == "v1" for vid in filtered.to_native()["video_id"].to_list())

    # Filter to v2
    filtered = filter_samples_by_partition(samples_df, "video_id", "v2")
    assert len(filtered) == 2

    # Filter to v3
    filtered = filter_samples_by_partition(samples_df, "video_id", "v3")
    assert len(filtered) == 1


def test_filter_samples_by_partition_empty_df() -> None:
    """Test filtering empty DataFrame doesn't error."""
    from metaxy.ext.dagster import filter_samples_by_partition

    empty_df = nw.from_native(
        pl.DataFrame(
            {
                "video_id": [],
                "frame_id": [],
                "value": [],
            }
        )
    )

    # Should not raise
    filtered = filter_samples_by_partition(empty_df, "video_id", "v1")

    assert len(filtered) == 0


def test_filter_samples_by_partition_missing_column() -> None:
    """Test error when partition column doesn't exist in samples."""
    from metaxy.ext.dagster import filter_samples_by_partition

    samples_df = nw.from_native(
        pl.DataFrame(
            {
                "frame_id": [1, 2],
                "value": [10, 20],
            }
        )
    )

    with pytest.raises(ValueError, match="Partition column 'video_id' not found"):
        filter_samples_by_partition(samples_df, "video_id", "v1")


def test_filter_samples_by_partition_no_matches() -> None:
    """Test filtering when no rows match the partition key."""
    from metaxy.ext.dagster import filter_samples_by_partition

    samples_df = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "frame_id": [1, 2],
                "value": [10, 20],
            }
        )
    )

    # Filter to v3 (doesn't exist)
    filtered = filter_samples_by_partition(samples_df, "video_id", "v3")

    # Should return empty DataFrame with same schema
    assert len(filtered) == 0
    assert "video_id" in filtered.columns
    assert "frame_id" in filtered.columns
    assert "value" in filtered.columns


def test_create_video_partitions_def_default() -> None:
    """Test creating video partitions with default parameters."""
    from metaxy.ext.dagster import create_video_partitions_def

    partitions = create_video_partitions_def()

    # Verify it's a DynamicPartitionsDefinition
    assert isinstance(partitions, dg.DynamicPartitionsDefinition)

    # Verify default name
    assert partitions.name == "videos"


def test_create_video_partitions_def_custom_name() -> None:
    """Test creating video partitions with custom name."""
    from metaxy.ext.dagster import create_video_partitions_def

    partitions = create_video_partitions_def(name="my_videos")

    assert isinstance(partitions, dg.DynamicPartitionsDefinition)
    assert partitions.name == "my_videos"


def test_create_video_partitions_def_with_initial_keys() -> None:
    """Test creating video partitions with initial partition keys."""
    from metaxy.ext.dagster import create_video_partitions_def

    # Note: initial_partition_keys parameter exists but DynamicPartitionsDefinition
    # doesn't use it at creation time. Partitions are added dynamically later.
    partitions = create_video_partitions_def(
        name="videos",
        initial_partition_keys=["video_1", "video_2", "video_3"],
    )

    assert isinstance(partitions, dg.DynamicPartitionsDefinition)
    assert partitions.name == "videos"
