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
            pl.DataFrame(
                [{"id": "1", "metaxy_provenance_by_field": {"default": "asd"}}]
            ),
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

    metadata = pl.DataFrame(
        [{"id": "1", "metaxy_provenance_by_field": {"default": "hash1"}}]
    )

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
