"""Tests for metadata store copy functionality."""

from collections.abc import Iterator

import narwhals as nw
import polars as pl
import pytest

from metaxy import Feature, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store import InMemoryMetadataStore
from metaxy.metadata_store.base import allow_feature_version_override


@pytest.fixture
def sample_features(graph) -> Iterator[tuple[type[Feature], type[Feature]]]:
    """Create sample features for testing."""

    class FeatureA(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "feature_a"]),
            fields=[FieldSpec(key=FieldKey("field_a"), code_version="1")],
        ),
    ):
        """First test feature."""

        pass

    class FeatureB(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "feature_b"]),
            fields=[FieldSpec(key=FieldKey("field_b"), code_version="1")],
        ),
    ):
        """Second test feature."""

        pass

    yield FeatureA, FeatureB


def test_copy_metadata_all_features(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test copying all features from one store to another."""
    FeatureA, FeatureB = sample_features

    # Create source and destination stores
    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write to source store
    with source_store.open("write"):
        # Write metadata to source store
        source_data_a = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3"],
                "field_a": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"field_a": "hash1"},
                    {"field_a": "hash2"},
                    {"field_a": "hash3"},
                ],
            }
        )
        source_store.write_metadata(FeatureA, source_data_a)

        source_data_b = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2"],
                "field_b": [10, 20],
                "metaxy_provenance_by_field": [
                    {"field_b": "hash10"},
                    {"field_b": "hash20"},
                ],
            }
        )
        source_store.write_metadata(FeatureB, source_data_b)

        # Get the snapshot_version that was written
        written_data = (
            source_store.read_metadata(FeatureA, current_only=False)
            .collect()
            .to_polars()
        )
        snapshot_version = written_data["metaxy_snapshot_version"][0]

    # Copy with destination store (source must be opened for reading)
    with source_store.open("read"), dest_store.open("write"):
        stats = dest_store.copy_metadata(
            from_store=source_store,
            features=None,  # Copy all
            from_snapshot=snapshot_version,
        )

        # Verify stats
        assert stats["features_copied"] == 2
        assert stats["rows_copied"] == 5  # 3 + 2

        # Verify data in destination has same snapshot_version
        dest_data_a = (
            dest_store.read_metadata(FeatureA, current_only=False).collect().to_polars()
        )
        assert dest_data_a.height == 3
        assert set(dest_data_a["sample_uid"].to_list()) == {"s1", "s2", "s3"}
        assert all(
            sid == snapshot_version for sid in dest_data_a["metaxy_snapshot_version"]
        )

        dest_data_b = (
            dest_store.read_metadata(FeatureB, current_only=False).collect().to_polars()
        )
        assert dest_data_b.height == 2
        assert set(dest_data_b["sample_uid"].to_list()) == {"s1", "s2"}
        assert all(
            sid == snapshot_version for sid in dest_data_b["metaxy_snapshot_version"]
        )


def test_copy_metadata_specific_features(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test copying specific features."""
    FeatureA, FeatureB = sample_features

    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write to source
    with source_store.open("write"):
        source_data_a = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "field_a": [1],
                "metaxy_provenance_by_field": [{"field_a": "hash1"}],
            }
        )
        source_store.write_metadata(FeatureA, source_data_a)

        source_data_b = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "field_b": [10],
                "metaxy_provenance_by_field": [{"field_b": "hash10"}],
            }
        )
        source_store.write_metadata(FeatureB, source_data_b)

        # Get the snapshot_version that was written
        written_data = (
            source_store.read_metadata(FeatureA, current_only=False)
            .collect()
            .to_polars()
        )
        snapshot_version = written_data["metaxy_snapshot_version"][0]

    # Copy only FeatureA
    with dest_store.open("write"):
        stats = dest_store.copy_metadata(
            from_store=source_store,
            features=[FeatureA.spec().key],
            from_snapshot=snapshot_version,
        )

        # Verify stats
        assert stats["features_copied"] == 1
        assert stats["rows_copied"] == 1

        # Verify FeatureA exists in destination
        dest_data_a = (
            dest_store.read_metadata(FeatureA, current_only=False).collect().to_polars()
        )
        assert dest_data_a.height == 1

        # Verify FeatureB does not exist in destination
        assert not dest_store.has_feature(FeatureB)


def test_copy_metadata_with_snapshot_filter(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test copying metadata filtered by snapshot version."""
    FeatureA, FeatureB = sample_features

    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write data with different snapshot versions to source
    with source_store.open("write"):
        snapshot_1 = "snapshot_123"
        snapshot_2 = "snapshot_456"

        # Write metadata with different snapshot versions
        with allow_feature_version_override():
            # Data with snapshot 1
            data_snapshot_1 = pl.DataFrame(
                {
                    "sample_uid": ["s1", "s2"],
                    "field_a": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"field_a": "hash1"},
                        {"field_a": "hash2"},
                    ],
                    "metaxy_feature_version": [FeatureA.feature_version()] * 2,
                    "metaxy_snapshot_version": [snapshot_1] * 2,
                }
            )
            source_store.write_metadata(FeatureA, data_snapshot_1)

            # Data with snapshot 2
            data_snapshot_2 = pl.DataFrame(
                {
                    "sample_uid": ["s3", "s4", "s5"],
                    "field_a": [3, 4, 5],
                    "metaxy_provenance_by_field": [
                        {"field_a": "hash3"},
                        {"field_a": "hash4"},
                        {"field_a": "hash5"},
                    ],
                    "metaxy_feature_version": [FeatureA.feature_version()] * 3,
                    "metaxy_snapshot_version": [snapshot_2] * 3,
                }
            )
            source_store.write_metadata(FeatureA, data_snapshot_2)

        # Verify source has both snapshots
        all_source_data = (
            source_store.read_metadata(FeatureA, current_only=False)
            .collect()
            .to_polars()
        )
        assert all_source_data.height == 5

    # Copy only snapshot 2
    with dest_store.open("write"):
        stats = dest_store.copy_metadata(
            from_store=source_store,
            features=[FeatureA.spec().key],
            from_snapshot=snapshot_2,
        )

        # Verify only snapshot 2 data was copied with same snapshot_version
        assert stats["features_copied"] == 1
        assert stats["rows_copied"] == 3

        # Verify destination has only the copied data with preserved snapshot_version
        dest_data = (
            dest_store.read_metadata(FeatureA, current_only=False).collect().to_polars()
        )
        assert dest_data.height == 3
        assert set(dest_data["sample_uid"].to_list()) == {"s3", "s4", "s5"}
        assert all(sid == snapshot_2 for sid in dest_data["metaxy_snapshot_version"])


def test_copy_metadata_empty_source() -> None:
    """Test copying from store with no features."""
    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    with dest_store.open("write"):
        stats = dest_store.copy_metadata(
            from_store=source_store,
            features=None,
        )

        assert stats["features_copied"] == 0
        assert stats["rows_copied"] == 0


def test_copy_metadata_missing_feature(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test copying a feature that doesn't exist in source."""
    FeatureA, FeatureB = sample_features

    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write only FeatureA to source
    with source_store.open("write"):
        source_data_a = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "field_a": [1],
                "metaxy_provenance_by_field": [{"field_a": "hash1"}],
            }
        )
        source_store.write_metadata(FeatureA, source_data_a)

        # Get the snapshot_version that was written
        written_data = (
            source_store.read_metadata(FeatureA, current_only=False)
            .collect()
            .to_polars()
        )
        snapshot_version = written_data["metaxy_snapshot_version"][0]

    # Try to copy both features (FeatureB doesn't exist)
    with dest_store.open("write"):
        stats = dest_store.copy_metadata(
            from_store=source_store,
            features=[FeatureA.spec().key, FeatureB.spec().key],
            from_snapshot=snapshot_version,
        )

        # Should copy only FeatureA and skip FeatureB with warning
        assert stats["features_copied"] == 1
        assert stats["rows_copied"] == 1


def test_copy_metadata_preserves_feature_version(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test that feature_version is preserved during copy."""
    FeatureA, _ = sample_features

    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write data to source
    with source_store.open("write"):
        original_version = FeatureA.feature_version()
        source_data = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "field_a": [1],
                "metaxy_provenance_by_field": [{"field_a": "hash1"}],
            }
        )
        source_store.write_metadata(FeatureA, source_data)

        # Get the snapshot_version that was written
        written_data = (
            source_store.read_metadata(FeatureA, current_only=False)
            .collect()
            .to_polars()
        )
        snapshot_version = written_data["metaxy_snapshot_version"][0]

    # Copy to destination
    with dest_store.open("write"):
        dest_store.copy_metadata(
            from_store=source_store,
            features=[FeatureA.spec().key],
            from_snapshot=snapshot_version,
        )

        # Verify feature_version is preserved
        dest_data = (
            dest_store.read_metadata(FeatureA, current_only=False).collect().to_polars()
        )
        assert dest_data["metaxy_feature_version"][0] == original_version


def test_copy_metadata_store_not_open() -> None:
    """Test that copy fails if stores are not opened."""
    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Don't use context managers - stores not opened
    with pytest.raises(ValueError, match="must be opened"):
        dest_store.copy_metadata(from_store=source_store)


def test_copy_metadata_preserves_snapshot_version(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test that snapshot_version is preserved during copy."""
    FeatureA, _ = sample_features

    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write data to source
    with source_store.open("write"):
        source_data = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "field_a": [1],
                "metaxy_provenance_by_field": [{"field_a": "hash1"}],
            }
        )
        source_store.write_metadata(FeatureA, source_data)

        # Get the snapshot_version that was written
        written_data = (
            source_store.read_metadata(FeatureA, current_only=False)
            .collect()
            .to_polars()
        )
        original_snapshot = written_data["metaxy_snapshot_version"][0]

    # Copy - snapshot_version should be preserved
    with dest_store.open("write"):
        dest_store.copy_metadata(
            from_store=source_store,
            features=[FeatureA.spec().key],
            from_snapshot=original_snapshot,
        )

        # Verify snapshot_version was preserved
        dest_data = (
            dest_store.read_metadata(FeatureA, current_only=False).collect().to_polars()
        )
        assert dest_data["metaxy_snapshot_version"][0] == original_snapshot


def test_copy_metadata_no_rows_for_snapshot(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test copying when no rows match the specified snapshot."""
    FeatureA, _ = sample_features

    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write data to source
    with source_store.open("write"):
        source_data = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "field_a": [1],
                "metaxy_provenance_by_field": [{"field_a": "hash1"}],
            }
        )
        source_store.write_metadata(FeatureA, source_data)

    # Try to copy with non-existent snapshot
    with dest_store.open("write"):
        stats = dest_store.copy_metadata(
            from_store=source_store,
            features=[FeatureA.spec().key],
            from_snapshot="nonexistent_snapshot",
        )

        # Should skip feature with warning
        assert stats["features_copied"] == 0
        assert stats["rows_copied"] == 0


def test_copy_metadata_with_global_filters(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test copying with global filters applied to all features."""
    FeatureA, FeatureB = sample_features

    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write metadata with different sample_uids
    with source_store.open("write"):
        source_data_a = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3"],
                "field_a": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"field_a": "hash1"},
                    {"field_a": "hash2"},
                    {"field_a": "hash3"},
                ],
            }
        )
        source_store.write_metadata(FeatureA, source_data_a)

        source_data_b = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3", "s4"],
                "field_b": [10, 20, 30, 40],
                "metaxy_provenance_by_field": [
                    {"field_b": "hash10"},
                    {"field_b": "hash20"},
                    {"field_b": "hash30"},
                    {"field_b": "hash40"},
                ],
            }
        )
        source_store.write_metadata(FeatureB, source_data_b)

        # Get snapshot
        written_data = (
            source_store.read_metadata(FeatureA, current_only=False)
            .collect()
            .to_polars()
        )
        snapshot_version = written_data["metaxy_snapshot_version"][0]

    # Copy with global filter - only s1 and s2
    with dest_store.open("write"):
        stats = dest_store.copy_metadata(
            from_store=source_store,
            features=None,  # All features
            from_snapshot=snapshot_version,
            filters={
                "test/feature_a": [nw.col("sample_uid").is_in(["s1", "s2"])],
                "test/feature_b": [nw.col("sample_uid").is_in(["s1", "s2"])],
            },
        )

        # Verify both features were copied but filtered
        assert stats["features_copied"] == 2
        assert stats["rows_copied"] == 4  # 2 rows from A, 2 from B

        # Verify FeatureA has only s1, s2
        dest_data_a = (
            dest_store.read_metadata(FeatureA, current_only=False).collect().to_polars()
        )
        assert set(dest_data_a["sample_uid"]) == {"s1", "s2"}

        # Verify FeatureB has only s1, s2
        dest_data_b = (
            dest_store.read_metadata(FeatureB, current_only=False).collect().to_polars()
        )
        assert set(dest_data_b["sample_uid"]) == {"s1", "s2"}


def test_copy_metadata_with_per_feature_filters(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test copying with per-feature filters."""
    FeatureA, FeatureB = sample_features

    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write metadata
    with source_store.open("write"):
        source_data_a = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3"],
                "field_a": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"field_a": "hash1"},
                    {"field_a": "hash2"},
                    {"field_a": "hash3"},
                ],
            }
        )
        source_store.write_metadata(FeatureA, source_data_a)

        source_data_b = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3"],
                "field_b": [10, 20, 30],
                "metaxy_provenance_by_field": [
                    {"field_b": "hash10"},
                    {"field_b": "hash20"},
                    {"field_b": "hash30"},
                ],
            }
        )
        source_store.write_metadata(FeatureB, source_data_b)

        # Get snapshot
        written_data = (
            source_store.read_metadata(FeatureA, current_only=False)
            .collect()
            .to_polars()
        )
        snapshot_version = written_data["metaxy_snapshot_version"][0]

    # Copy with per-feature filters
    with dest_store.open("write"):
        stats = dest_store.copy_metadata(
            from_store=source_store,
            features=[FeatureA.spec().key, FeatureB.spec().key],
            from_snapshot=snapshot_version,
            filters={
                "test/feature_a": [
                    nw.col("field_a") > 1
                ],  # Only rows where field_a > 1
                "test/feature_b": [
                    nw.col("field_b") < 30
                ],  # Only rows where field_b < 30
            },
        )

        # Verify both features copied with their specific filters
        assert stats["features_copied"] == 2
        assert stats["rows_copied"] == 4  # 2 from A (s2, s3), 2 from B (s1, s2)

        # Verify FeatureA has only s2, s3 (field_a > 1)
        dest_data_a = (
            dest_store.read_metadata(FeatureA, current_only=False).collect().to_polars()
        )
        assert set(dest_data_a["sample_uid"]) == {"s2", "s3"}
        assert all(dest_data_a["field_a"] > 1)

        # Verify FeatureB has only s1, s2 (field_b < 30)
        dest_data_b = (
            dest_store.read_metadata(FeatureB, current_only=False).collect().to_polars()
        )
        assert set(dest_data_b["sample_uid"]) == {"s1", "s2"}
        assert all(dest_data_b["field_b"] < 30)


def test_copy_metadata_with_mixed_filters(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test copying with multiple filters combined."""
    FeatureA, FeatureB = sample_features

    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write metadata
    with source_store.open("write"):
        source_data_a = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3", "s4"],
                "field_a": [1, 2, 3, 4],
                "metaxy_provenance_by_field": [
                    {"field_a": "hash1"},
                    {"field_a": "hash2"},
                    {"field_a": "hash3"},
                    {"field_a": "hash4"},
                ],
            }
        )
        source_store.write_metadata(FeatureA, source_data_a)

        source_data_b = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3", "s4"],
                "field_b": [10, 20, 30, 40],
                "metaxy_provenance_by_field": [
                    {"field_b": "hash10"},
                    {"field_b": "hash20"},
                    {"field_b": "hash30"},
                    {"field_b": "hash40"},
                ],
            }
        )
        source_store.write_metadata(FeatureB, source_data_b)

        # Get snapshot
        written_data = (
            source_store.read_metadata(FeatureA, current_only=False)
            .collect()
            .to_polars()
        )
        snapshot_version = written_data["metaxy_snapshot_version"][0]

    # Copy with multiple filters combined for each feature
    with dest_store.open("write"):
        stats = dest_store.copy_metadata(
            from_store=source_store,
            features=[FeatureA.spec().key, FeatureB.spec().key],
            from_snapshot=snapshot_version,
            filters={
                "test/feature_a": [
                    nw.col("sample_uid").is_in(["s1", "s2", "s3"]),
                    nw.col("field_a") <= 3,
                ],
                "test/feature_b": [nw.col("sample_uid").is_in(["s1", "s2", "s3"])],
            },
        )

        # Verify results
        assert stats["features_copied"] == 2

        # FeatureA: both filters applied (s1,s2,s3) AND (field_a <= 3)
        # Result: s1, s2, s3
        dest_data_a = (
            dest_store.read_metadata(FeatureA, current_only=False).collect().to_polars()
        )
        assert set(dest_data_a["sample_uid"]) == {"s1", "s2", "s3"}
        assert all(dest_data_a["field_a"] <= 3)

        # FeatureB: only global filter (s1,s2,s3)
        # Result: s1, s2, s3
        dest_data_b = (
            dest_store.read_metadata(FeatureB, current_only=False).collect().to_polars()
        )
        assert set(dest_data_b["sample_uid"]) == {"s1", "s2", "s3"}


def test_copy_metadata_with_mixed_feature_types(
    sample_features: tuple[type[Feature], type[Feature]],
) -> None:
    """Test that we can use different filter configurations for different features."""
    FeatureA, FeatureB = sample_features

    source_store = InMemoryMetadataStore()
    dest_store = InMemoryMetadataStore()

    # Write metadata
    with source_store.open("write"):
        source_data_a = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2"],
                "field_a": [1, 2],
                "metaxy_provenance_by_field": [
                    {"field_a": "hash1"},
                    {"field_a": "hash2"},
                ],
            }
        )
        source_store.write_metadata(FeatureA, source_data_a)

        source_data_b = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2"],
                "field_b": [10, 20],
                "metaxy_provenance_by_field": [
                    {"field_b": "hash10"},
                    {"field_b": "hash20"},
                ],
            }
        )
        source_store.write_metadata(FeatureB, source_data_b)

        # Get snapshot
        written_data = (
            source_store.read_metadata(FeatureA, current_only=False)
            .collect()
            .to_polars()
        )
        snapshot_version = written_data["metaxy_snapshot_version"][0]

    # Apply filter to one feature but not the other
    with dest_store.open("write"):
        stats = dest_store.copy_metadata(
            from_store=source_store,
            features=[FeatureA.spec().key, FeatureB.spec().key],
            from_snapshot=snapshot_version,
            filters={
                "test/feature_a": [nw.col("field_a") > 1],
                # No filter for feature_b
            },
        )

        # Verify results
        assert stats["features_copied"] == 2
        assert stats["rows_copied"] == 3  # 1 from A, 2 from B

        # FeatureA filtered
        dest_data_a = (
            dest_store.read_metadata(FeatureA, current_only=False).collect().to_polars()
        )
        assert dest_data_a.height == 1
        assert dest_data_a["sample_uid"][0] == "s2"

        # FeatureB not filtered
        dest_data_b = (
            dest_store.read_metadata(FeatureB, current_only=False).collect().to_polars()
        )
        assert dest_data_b.height == 2
