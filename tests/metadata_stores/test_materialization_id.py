"""Tests for metaxy_materialization_id system column."""

import narwhals as nw
import polars as pl
from pytest_cases import parametrize_with_cases

from metaxy import BaseFeature, FeatureKey, FeatureSpec
from metaxy.metadata_store import MetadataStore
from metaxy.models.constants import (
    METAXY_MATERIALIZATION_ID,
    METAXY_PROVENANCE_BY_FIELD,
)

from .conftest import AllStoresCases


@parametrize_with_cases("store", cases=AllStoresCases)
def test_store_level_materialization_id(store: MetadataStore):
    """Test that store-level materialization_id is applied to all writes."""
    # Set the materialization_id on the existing store
    store._materialization_id = "test-run-123"

    key = FeatureKey(["test_mat_id"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    with store.open(mode="w"):
        store.write(
            key,
            pl.DataFrame([{"id": "1", METAXY_PROVENANCE_BY_FIELD: {"default": "abc"}}]),
        )

    # Read back and verify materialization_id
    with store:
        df = store.read(key)
        assert df is not None
        df_collected = df.collect()
        assert METAXY_MATERIALIZATION_ID in df_collected.columns
        assert df_collected[METAXY_MATERIALIZATION_ID][0] == "test-run-123"


@parametrize_with_cases("store", cases=AllStoresCases)
def test_write_level_materialization_id_override(store: MetadataStore):
    """Test that write-level materialization_id overrides store default."""
    # Set a default materialization_id on the store
    store._materialization_id = "default-run-123"

    key = FeatureKey(["test_override"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    with store.open(mode="w"):
        store.write(
            key,
            pl.DataFrame([{"id": "1", METAXY_PROVENANCE_BY_FIELD: {"default": "abc"}}]),
            materialization_id="override-run-456",
        )

    # Verify override was applied
    with store:
        df = store.read(key)
        assert df is not None
        df_collected = df.collect()
        assert df_collected[METAXY_MATERIALIZATION_ID][0] == "override-run-456"


@parametrize_with_cases("store", cases=AllStoresCases)
def test_nullable_materialization_id(store: MetadataStore):
    """Test that materialization_id can be null when not provided."""
    key = FeatureKey(["test_nullable"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    with store.open(mode="w"):
        store.write(
            key,
            pl.DataFrame([{"id": "1", METAXY_PROVENANCE_BY_FIELD: {"default": "abc"}}]),
        )

    # Verify column exists but is null
    with store:
        df = store.read(key)
        assert df is not None
        df_collected = df.collect()
        assert METAXY_MATERIALIZATION_ID in df_collected.columns
        assert df_collected[METAXY_MATERIALIZATION_ID][0] is None


@parametrize_with_cases("store", cases=AllStoresCases)
def test_filter_by_materialization_id(store: MetadataStore):
    """Test filtering reads by materialization_id."""
    key = FeatureKey(["test_filter"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    # Write multiple batches with different IDs
    with store.open(mode="w"):
        store.write(
            key,
            pl.DataFrame([{"id": "1", METAXY_PROVENANCE_BY_FIELD: {"default": "abc"}}]),
            materialization_id="run-1",
        )
        store.write(
            key,
            pl.DataFrame([{"id": "2", METAXY_PROVENANCE_BY_FIELD: {"default": "def"}}]),
            materialization_id="run-2",
        )
        store.write(
            key,
            pl.DataFrame([{"id": "3", METAXY_PROVENANCE_BY_FIELD: {"default": "ghi"}}]),
            materialization_id="run-1",
        )

    # Filter by run-1
    with store:
        df = store.read(key, filters=[nw.col(METAXY_MATERIALIZATION_ID) == "run-1"])
        assert df is not None
        df_collected = df.collect()
        assert len(df_collected) == 2
        assert set(df_collected["id"].to_list()) == {"1", "3"}

    # Filter by run-2
    with store:
        df = store.read(key, filters=[nw.col(METAXY_MATERIALIZATION_ID) == "run-2"])
        assert df is not None
        df_collected = df.collect()
        assert len(df_collected) == 1
        assert df_collected["id"][0] == "2"


@parametrize_with_cases("store", cases=AllStoresCases)
def test_materialization_id_multiple_writes(store: MetadataStore):
    """Test that different writes can have different materialization_ids."""
    key = FeatureKey(["test_multiple"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str

    # Write with different materialization_ids
    with store.open(mode="w"):
        for i in range(3):
            store.write(
                key,
                pl.DataFrame(
                    [
                        {
                            "id": f"sample_{i}",
                            METAXY_PROVENANCE_BY_FIELD: {"default": f"hash_{i}"},
                        }
                    ]
                ),
                materialization_id=f"run-{i}",
            )

    # Read all and verify each has its materialization_id
    with store:
        df = store.read(key, with_sample_history=True)
        assert df is not None
        df_collected = df.collect()
        assert len(df_collected) == 3

        # Check that each sample has the correct materialization_id
        for i in range(3):
            sample_row = df_collected.filter(nw.col("id") == f"sample_{i}")
            assert len(sample_row) == 1
            assert sample_row[METAXY_MATERIALIZATION_ID][0] == f"run-{i}"
