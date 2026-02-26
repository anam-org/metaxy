"""Tests for MetadataStore pickle roundtrip simulating Ray serialization."""

from __future__ import annotations

import pickle

import polars as pl
import pytest

import metaxy as mx
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

pytest.importorskip("ray")

from .conftest import FEATURE_KEY


def test_pickle_roundtrip_simulating_ray(
    ray_config: mx.MetaxyConfig,
    delta_store: DeltaMetadataStore,
):
    """Simulate Ray's pickle-based serialization of an open store.

    Ray pickles objects to send them to workers. The worker then reopens the
    store. This test verifies the full cycle without requiring a live Ray cluster.
    """
    mx.init(ray_config)

    # Open store and pickle it (simulates driver side)
    with delta_store.open("w"):
        pickled = pickle.dumps(delta_store)

    # Unpickle on "worker" side — store should be closed
    worker_store = pickle.loads(pickled)
    assert not worker_store._is_open

    # Worker reopens, writes, then reads back
    with worker_store.open("w"):
        worker_store.write(
            FEATURE_KEY,
            pl.DataFrame(
                [
                    {
                        "sample_uid": "x",
                        "value": 42,
                        "metaxy_provenance_by_field": {"default": "hash_x"},
                        "metaxy_provenance": "combined_hash_x",
                    }
                ]
            ),
        )
        result = worker_store.read(FEATURE_KEY).collect().to_polars()
        assert result.shape[0] == 1
        assert result["sample_uid"][0] == "x"
