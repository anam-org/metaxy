import pickle

import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from metaxy import BaseFeature, FeatureKey, FeatureSpec
from metaxy.metadata_store import MetadataStore, StoreNotOpenError

from .conftest import AllStoresCases


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
def test_has_feature(store: MetadataStore):
    """Test that metadata stores can be pickled and unpickled."""
    key = FeatureKey(["test"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        """Root feature with no dependencies."""

        id: str

    with store:
        assert not store.has_feature(key)
        assert not store.has_feature(MyFeature)

        store.write_metadata(
            key,
            pl.DataFrame([{"id": "1", "metaxy_provenance_by_field": {"default": "asd"}}]),
        )

        assert store.has_feature(key)
        assert store.has_feature(MyFeature)


@parametrize_with_cases("store", cases=AllStoresCases)
def test_read_metadata_raises_store_not_open(store: MetadataStore):
    """Test that read_metadata raises StoreNotOpenError when store is not open."""
    key = FeatureKey(["test_not_open"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    # Store is not open - should raise StoreNotOpenError
    with pytest.raises(StoreNotOpenError, match="must be opened before use"):
        store.read_metadata(key)


@parametrize_with_cases("store", cases=AllStoresCases)
def test_write_metadata_raises_store_not_open(store: MetadataStore):
    """Test that write_metadata raises StoreNotOpenError when store is not open."""
    key = FeatureKey(["test_not_open"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    metadata = pl.DataFrame([{"id": "1", "metaxy_provenance_by_field": {"default": "hash1"}}])

    # Store is not open - should raise StoreNotOpenError
    with pytest.raises(StoreNotOpenError, match="must be opened before use"):
        store.write_metadata(key, metadata)


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
def test_read_metadata_filters_applied_after_deduplication(store: MetadataStore):
    """Test that user filters in read_metadata are applied AFTER timestamp-based deduplication.

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

    with store.open("write"):
        # Step 1: Write initial data where all rows have status=NULL
        initial_data = pl.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e"],
                "status": pl.Series([None, None, None, None, None], dtype=pl.Int64),
                "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 6)],
            }
        )
        store.write_metadata(key, initial_data)

        # Step 2: Write updated data for rows a, b, c with status filled
        time.sleep(0.01)  # Ensure different metaxy_created_at timestamps
        updated_data = pl.DataFrame(
            {
                "id": ["a", "b", "c"],
                "status": [100, 200, 300],
                "metaxy_provenance_by_field": [{"default": f"hash{i}_v2"} for i in range(1, 4)],
            }
        )
        store.write_metadata(key, updated_data)

        # After deduplication:
        # - Rows a, b, c should have the latest version (status filled)
        # - Rows d, e should still have status=NULL
        # Total: 5 unique rows

        # Test 1: Read without filter - should get 5 rows
        result_all = store.read_metadata(key)
        assert result_all.collect().shape[0] == 5

        # Test 2: Read with filter "status IS NULL" - should get 2 rows (d, e)
        # This is the key test: filter must be applied AFTER deduplication
        result_null = store.read_metadata(key, filters=[nw.col("status").is_null()])
        result_null_df = result_null.collect()
        assert result_null_df.shape[0] == 2, (
            f"Expected 2 rows with status IS NULL (d, e), got {result_null_df.shape[0]}. "
            "Filter should be applied AFTER deduplication."
        )
        # Verify the correct rows are returned (use polars for consistent access)
        ids = set(result_null_df.to_polars()["id"].to_list())
        assert ids == {"d", "e"}, f"Expected rows d, e but got {ids}"

        # Test 3: Read with filter "status IS NOT NULL" - should get 3 rows (a, b, c)
        result_not_null = store.read_metadata(key, filters=[~nw.col("status").is_null()])
        result_not_null_df = result_not_null.collect()
        assert result_not_null_df.shape[0] == 3, (
            f"Expected 3 rows with status IS NOT NULL (a, b, c), got {result_not_null_df.shape[0]}. "
            "Filter should be applied AFTER deduplication."
        )
        # Verify the correct rows are returned (use polars for consistent access)
        ids = set(result_not_null_df.to_polars()["id"].to_list())
        assert ids == {"a", "b", "c"}, f"Expected rows a, b, c but got {ids}"
