import pickle

import polars as pl
import pytest
from pytest_cases import parametrize_with_cases
from syrupy.assertion import SnapshotAssertion

from metaxy import BaseFeature, FeatureKey, FeatureSpec
from metaxy.metadata_store import MetadataStore, StoreNotOpenError

from .conftest import AllStoresCases


@parametrize_with_cases("store", cases=AllStoresCases)
def test_repr(store: MetadataStore, snapshot: SnapshotAssertion) -> None:
    """Test that repr() contains display() for all store types."""
    # For unnamed stores, repr() equals display()
    # For named stores, repr() wraps display() with [name] prefix
    assert store.display() in repr(store)


@parametrize_with_cases("store", cases=AllStoresCases)
def test_store_is_pickleable(store: MetadataStore):
    """Test that metadata stores can be pickled and unpickled."""
    # Pickle the store
    pickled = pickle.dumps(store)

    # Unpickle the store
    unpickled_store = pickle.loads(pickled)

    # Verify it's the same type
    assert type(unpickled_store) is type(store)
    assert isinstance(unpickled_store, MetadataStore)


@parametrize_with_cases("store", cases=AllStoresCases)
def test_store_is_pickleable_while_open(store: MetadataStore):
    """Test that an open store can be pickled and the unpickled copy is closed."""
    key = FeatureKey(["test_pickle_open"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    with store.open("w"):
        # Pickle while the store is open
        pickled = pickle.dumps(store)
        unpickled_store = pickle.loads(pickled)

        # Unpickled store should NOT be open
        assert type(unpickled_store) is type(store)
        assert not unpickled_store._is_open

    # Reopen the unpickled store and verify it works
    with unpickled_store.open("w"):
        unpickled_store.write(
            key,
            pl.DataFrame([{"id": "1", "metaxy_provenance_by_field": {"default": "asd"}}]),
        )
        result = unpickled_store.read(key).collect().to_polars()
        assert result.shape[0] == 1


@parametrize_with_cases("store", cases=AllStoresCases)
def test_has_feature(store: MetadataStore):
    """Test that metadata stores can be pickled and unpickled."""
    key = FeatureKey(["test"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        """Root feature with no dependencies."""

        id: str

    with store.open("w"):
        assert not store.has_feature(key)
        assert not store.has_feature(MyFeature)

        store.write(
            key,
            pl.DataFrame([{"id": "1", "metaxy_provenance_by_field": {"default": "asd"}}]),
        )

        assert store.has_feature(key)
        assert store.has_feature(MyFeature)


@parametrize_with_cases("store", cases=AllStoresCases)
def test_read_raises_store_not_open(store: MetadataStore):
    """Test that read raises StoreNotOpenError when store is not open."""
    key = FeatureKey(["test_not_open"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    # Store is not open - should raise StoreNotOpenError
    with pytest.raises(StoreNotOpenError, match="must be opened before use"):
        store.read(key)


@parametrize_with_cases("store", cases=AllStoresCases)
def test_write_raises_store_not_open(store: MetadataStore):
    """Test that write raises StoreNotOpenError when store is not open."""
    key = FeatureKey(["test_not_open"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    metadata = pl.DataFrame([{"id": "1", "metaxy_provenance_by_field": {"default": "hash1"}}])

    # Store is not open - should raise StoreNotOpenError
    with pytest.raises(StoreNotOpenError, match="must be opened before use"):
        store.write(key, metadata)


@parametrize_with_cases("store", cases=AllStoresCases)
def test_has_feature_raises_store_not_open(store: MetadataStore):
    """Test that has_feature raises StoreNotOpenError when store is not open."""
    key = FeatureKey(["test_not_open"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    # Store is not open - should raise StoreNotOpenError
    with pytest.raises(StoreNotOpenError, match="must be opened before use"):
        store.has_feature(key)


@parametrize_with_cases("store", cases=AllStoresCases)
def test_read_filters_applied_after_deduplication(store: MetadataStore):
    """Test that user filters in read are applied AFTER timestamp-based deduplication.

    This tests the critical behavior where:
    1. Initial write has rows with a column as NULL
    2. Later write updates those rows with non-NULL values
    3. Filter by IS NULL should only match rows where the LATEST version has NULL

    Without this fix, filters were applied BEFORE deduplication, causing wrong results:
    - Old rows would match the filter even if they were superseded by newer versions.

    See: https://github.com/anam-org/metaxy/issues/XXX
    """
    import time

    import narwhals as nw

    key = FeatureKey(["test_filter_after_dedup"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str
        status: int | None

    with store.open("w"):
        # Step 1: Write initial data where all rows have status=NULL
        initial_data = pl.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e"],
                "status": pl.Series([None, None, None, None, None], dtype=pl.Int64),
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 6)],
            }
        )
        store.write(key, initial_data)

        # Step 2: Write updated data for rows a, b, c with status filled
        time.sleep(0.01)  # Ensure different metaxy_created_at timestamps
        updated_data = pl.DataFrame(
            {
                "id": ["a", "b", "c"],
                "status": [100, 200, 300],
                "metaxy_provenance_by_field": [{"default": f"hash{i}_v2"} for i in range(1, 4)],
            }
        )
        store.write(key, updated_data)

        # After deduplication:
        # - Rows a, b, c should have the latest version (status filled)
        # - Rows d, e should still have status=NULL
        # Total: 5 unique rows

        # Test 1: Read without filter - should get 5 rows
        result_all = store.read(key)
        assert result_all.collect().shape[0] == 5

        # Test 2: Read with filter "status IS NULL" - should get 2 rows (d, e)
        # This is the key test: filter must be applied AFTER deduplication
        result_null = store.read(key, filters=[nw.col("status").is_null()])
        result_null_df = result_null.collect()
        assert result_null_df.shape[0] == 2, (
            f"Expected 2 rows with status IS NULL (d, e), got {result_null_df.shape[0]}. "
            "Filter should be applied AFTER deduplication."
        )
        # Verify the correct rows are returned (use polars for consistent access)
        ids = set(result_null_df.to_polars()["id"].to_list())
        assert ids == {"d", "e"}, f"Expected rows d, e but got {ids}"

        # Test 3: Read with filter "status IS NOT NULL" - should get 3 rows (a, b, c)
        result_not_null = store.read(key, filters=[~nw.col("status").is_null()])
        result_not_null_df = result_not_null.collect()
        assert result_not_null_df.shape[0] == 3, (
            f"Expected 3 rows with status IS NOT NULL (a, b, c), got {result_not_null_df.shape[0]}. "
            "Filter should be applied AFTER deduplication."
        )
        # Verify the correct rows are returned (use polars for consistent access)
        ids = set(result_not_null_df.to_polars()["id"].to_list())
        assert ids == {"a", "b", "c"}, f"Expected rows a, b, c but got {ids}"


@parametrize_with_cases("store", cases=AllStoresCases)
def test_metaxy_updated_at_column(store: MetadataStore):
    """Test that metaxy_updated_at is populated on write and used for deduplication.

    This tests the behavior of the metaxy_updated_at column:
    1. It's automatically populated when metadata is written
    2. It's updated to a new timestamp on every write (unlike metaxy_created_at)
    3. It's used by keep_latest_by_group for deduplication

    See: https://github.com/anam-org/metaxy/issues/703
    """
    import time

    from metaxy.models.constants import METAXY_CREATED_AT, METAXY_UPDATED_AT

    key = FeatureKey(["test_updated_at"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str
        value: int

    with store.open("w"):
        # Step 1: Write initial data
        initial_data = pl.DataFrame(
            {
                "id": ["a"],
                "value": [1],
                "metaxy_provenance_by_field": [{"default": "hash1"}],
            }
        )
        store.write(key, initial_data)

        # Read back to check metaxy_updated_at was populated
        result1 = store.read(key, with_sample_history=True).collect().to_polars()
        assert METAXY_UPDATED_AT in result1.columns, "metaxy_updated_at should be present"
        assert METAXY_CREATED_AT in result1.columns, "metaxy_created_at should be present"
        updated_at_1 = result1[METAXY_UPDATED_AT][0]
        created_at_1 = result1[METAXY_CREATED_AT][0]
        assert updated_at_1 is not None, "metaxy_updated_at should not be null"
        assert created_at_1 is not None, "metaxy_created_at should not be null"

        # Step 2: Wait a bit and write updated data for the same row
        time.sleep(0.01)
        updated_data = pl.DataFrame(
            {
                "id": ["a"],
                "value": [2],
                "metaxy_provenance_by_field": [{"default": "hash1_v2"}],
            }
        )
        store.write(key, updated_data)

        # Read all versions (without deduplication) to verify both rows exist
        all_versions = store.read(key, with_sample_history=True).collect().to_polars()
        assert all_versions.shape[0] == 2, "Should have 2 versions of the row"

        # The newer row should have a later updated_at timestamp
        updated_ats = all_versions[METAXY_UPDATED_AT].sort()
        assert updated_ats[1] > updated_ats[0], "Second write should have later updated_at"

        # Read with deduplication (with_sample_history=False) - should get 1 row with value=2
        latest = store.read(key, with_sample_history=False).collect().to_polars()
        assert latest.shape[0] == 1, "Should get 1 row after deduplication"
        assert latest["value"][0] == 2, "Should get the latest value"


@parametrize_with_cases("store", cases=AllStoresCases)
def test_soft_delete_preserves_original_updated_at(store: MetadataStore):
    """Test that soft delete preserves the original metaxy_updated_at timestamp.

    When a row is soft-deleted, metaxy_deleted_at is set to the deletion time,
    but metaxy_updated_at should preserve the original value from when the row
    was last actually updated.
    """
    from metaxy.models.constants import METAXY_DELETED_AT, METAXY_UPDATED_AT

    key = FeatureKey(["test_soft_delete_timestamps"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str
        value: int

    with store.open("w"):
        # Write initial data
        initial_data = pl.DataFrame(
            {
                "id": ["a"],
                "value": [1],
                "metaxy_provenance_by_field": [{"default": "hash1"}],
            }
        )
        store.write(key, initial_data)

        # Read to get the original updated_at
        before_delete = store.read(key, with_sample_history=False).collect().to_polars()
        original_updated_at = before_delete[METAXY_UPDATED_AT][0]

        # Soft delete the row (filters=None deletes all rows)
        store.delete(key, filters=None, soft=True)

        # Read including soft-deleted rows
        result = store.read(key, include_soft_deleted=True, with_sample_history=False).collect().to_polars()

        assert result.shape[0] == 1, "Should have 1 row"
        deleted_at = result[METAXY_DELETED_AT][0]
        updated_at = result[METAXY_UPDATED_AT][0]

        assert deleted_at is not None, "metaxy_deleted_at should be set after soft delete"
        assert updated_at is not None, "metaxy_updated_at should be set"
        # updated_at should be preserved from before the delete, not changed to deleted_at
        assert updated_at == original_updated_at, (
            f"metaxy_updated_at should be preserved as {original_updated_at}, but got {updated_at}"
        )
        assert deleted_at > updated_at, "metaxy_deleted_at should be after metaxy_updated_at"


@parametrize_with_cases("store", cases=AllStoresCases)
def test_soft_delete_timestamps_consistency(store: MetadataStore):
    """Test timestamp consistency during soft delete operations.

    When soft deleting:
    - metaxy_deleted_at is set to the deletion time
    - metaxy_updated_at is preserved from the original row (reflects when data was last changed)
    - metaxy_created_at preserves the original row's creation time
    """
    from metaxy.models.constants import METAXY_CREATED_AT, METAXY_DELETED_AT, METAXY_UPDATED_AT

    key = FeatureKey(["test_soft_delete_all_timestamps"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str
        value: int

    with store.open("w"):
        # Write initial data
        initial_data = pl.DataFrame(
            {
                "id": ["a"],
                "value": [1],
                "metaxy_provenance_by_field": [{"default": "hash1"}],
            }
        )
        store.write(key, initial_data)

        # Read the original row to get its timestamps
        original_row = store.read(key, with_sample_history=False).collect().to_polars()
        original_created_at = original_row[METAXY_CREATED_AT][0]
        original_updated_at = original_row[METAXY_UPDATED_AT][0]

        # Soft delete the row
        store.delete(key, filters=None, soft=True)

        # Read all rows including soft-deleted, without deduplication
        all_rows = store.read(key, include_soft_deleted=True, with_sample_history=True).collect().to_polars()

        # Should have 2 rows: original and soft-deleted
        assert all_rows.shape[0] == 2, f"Expected 2 rows, got {all_rows.shape[0]}"

        # Find the soft-deleted row (has non-null deleted_at)
        soft_deleted_row = all_rows.filter(pl.col(METAXY_DELETED_AT).is_not_null())
        assert soft_deleted_row.shape[0] == 1, "Should have exactly 1 soft-deleted row"

        created_at = soft_deleted_row[METAXY_CREATED_AT][0]
        updated_at = soft_deleted_row[METAXY_UPDATED_AT][0]
        deleted_at = soft_deleted_row[METAXY_DELETED_AT][0]

        assert created_at is not None, "metaxy_created_at should be set"
        assert updated_at is not None, "metaxy_updated_at should be set"
        assert deleted_at is not None, "metaxy_deleted_at should be set"

        # updated_at should be preserved from the original row
        assert updated_at == original_updated_at, (
            f"metaxy_updated_at ({updated_at}) should be preserved from original ({original_updated_at})"
        )
        # deleted_at should be after updated_at (deletion happened after the last update)
        assert deleted_at > updated_at, (
            f"metaxy_deleted_at ({deleted_at}) should be after metaxy_updated_at ({updated_at})"
        )

        # The soft-deleted row preserves the original created_at (read from existing row)
        assert created_at == original_created_at, (
            f"Soft-deleted row should preserve original created_at ({original_created_at}), but got {created_at}"
        )

        # The deletion timestamps should be after the original created_at
        assert deleted_at > original_created_at, (
            f"Deletion timestamp ({deleted_at}) should be after original created_at ({original_created_at})"
        )
