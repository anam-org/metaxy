"""Test that all metadata stores are pickleable."""

import pickle

from pytest_cases import parametrize_with_cases

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
