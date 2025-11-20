import pickle

import polars as pl
from pytest_cases import parametrize_with_cases

from metaxy import BaseFeature, FeatureKey, FeatureSpec
from metaxy.metadata_store import MetadataStore

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
