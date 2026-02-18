"""Dagster cleanup op tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import dagster as dg
import polars as pl
import pytest
from metaxy_testing.models import SampleFeatureSpec

import metaxy.ext.dagster as mxd
from metaxy import BaseFeature, FeatureDep, FeatureKey, FeatureSpec, FieldKey, FieldSpec
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD


@pytest.fixture
def feature_cls():
    class Logs(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["logs"]),
            fields=[FieldSpec(key=FieldKey(["level"]), code_version="1")],
        ),
    ):
        level: str | None = None

    return Logs


def test_delete_op_with_mock():
    """Test delete op with mocked store using simple feature key."""
    store = MagicMock()

    ctx = dg.build_op_context(
        resources={"metaxy_store": store},
        op_config={
            "feature_key": ["logs"],
            "filters": ["level = 'debug'"],
            "soft": True,
        },
    )

    mxd.delete(ctx)
    store.delete.assert_called_once()
    args, kwargs = store.delete.call_args
    assert args[0] == FeatureKey(["logs"])
    assert kwargs["soft"] is True


def test_delete_op_with_slashed_key():
    """Test delete op with slashed feature key format."""
    store = MagicMock()

    ctx = dg.build_op_context(
        resources={"metaxy_store": store},
        op_config={
            "feature_key": ["customer", "segment"],
            "filters": ["status = 'inactive'"],
            "soft": True,
        },
    )

    mxd.delete(ctx)
    store.delete.assert_called_once()
    args, kwargs = store.delete.call_args
    assert args[0] == FeatureKey(["customer", "segment"])
    assert kwargs["soft"] is True


def test_delete_op_hard_delete_with_mock():
    """Test delete op with hard delete."""
    store = MagicMock()

    ctx = dg.build_op_context(
        resources={"metaxy_store": store},
        op_config={
            "feature_key": ["logs"],
            "filters": ["level = 'warn'"],
            "soft": False,
        },
    )

    mxd.delete(ctx)
    store.delete.assert_called_once()
    args, kwargs = store.delete.call_args
    assert args[0] == FeatureKey(["logs"])
    assert kwargs["soft"] is False


def test_delete_integration(feature_cls, tmp_path):
    """Integration test: create job, write data, run delete, verify results."""

    # Create a delta store
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")

    # Define a job with the delete op
    @dg.job(resource_defs={"metaxy_store": dg.ResourceDefinition.hardcoded_resource(store)})
    def cleanup_job():
        mxd.delete()

    # Write some test data
    test_data = pl.DataFrame(
        {
            "sample_uid": ["s1", "s2", "s3"],
            "level": ["debug", "info", "warn"],
            METAXY_PROVENANCE_BY_FIELD: [
                {"level": "p1"},
                {"level": "p2"},
                {"level": "p3"},
            ],
        }
    )

    with store.open("w"):
        store.write(feature_cls, test_data)

    # Verify data was written
    with store:
        initial_data = store.read(feature_cls).collect().to_polars()
        assert initial_data.height == 3

    # Execute the job with config to delete debug logs
    result = cleanup_job.execute_in_process(
        run_config={
            "ops": {
                "delete": {
                    "config": {
                        "feature_key": ["logs"],
                        "filters": ["level = 'debug'"],
                        "soft": True,
                    }
                }
            }
        }
    )

    # Verify the job succeeded
    assert result.success

    # Verify the data was soft-deleted (only 2 rows remain)
    with store:
        remaining_data = store.read(feature_cls).collect().to_polars()
        assert remaining_data.height == 2
        assert set(remaining_data["level"]) == {"info", "warn"}

        # Verify soft-deleted row is still in store when including deleted
        all_data = store.read(feature_cls, include_soft_deleted=True).collect().to_polars()
        assert all_data.height == 3


def test_delete_integration_hard_delete(feature_cls, tmp_path):
    """Integration test for hard delete: data is physically removed."""

    # Create a delta store
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")

    # Define a job with the delete op
    @dg.job(resource_defs={"metaxy_store": dg.ResourceDefinition.hardcoded_resource(store)})
    def hard_cleanup_job():
        mxd.delete()

    # Write some test data
    test_data = pl.DataFrame(
        {
            "sample_uid": ["s1", "s2", "s3"],
            "level": ["debug", "info", "warn"],
            METAXY_PROVENANCE_BY_FIELD: [
                {"level": "p1"},
                {"level": "p2"},
                {"level": "p3"},
            ],
        }
    )

    with store.open("w"):
        store.write(feature_cls, test_data)

    # Execute the job with hard delete
    result = hard_cleanup_job.execute_in_process(
        run_config={
            "ops": {
                "delete": {
                    "config": {
                        "feature_key": ["logs"],
                        "filters": ["level = 'debug'"],
                        "soft": False,  # Hard delete
                    }
                }
            }
        }
    )

    # Verify the job succeeded
    assert result.success

    # Verify the data was physically removed (only 2 rows remain)
    with store:
        remaining_data = store.read(feature_cls).collect().to_polars()
        assert remaining_data.height == 2
        assert set(remaining_data["level"]) == {"info", "warn"}

        # Verify hard-deleted row is NOT in store even when including deleted
        all_data = store.read(feature_cls, include_soft_deleted=True).collect().to_polars()
        assert all_data.height == 2  # Hard delete means it's gone


def test_delete_metadata_cascade_downstream(tmp_path):
    """Test cascade deletion downstream in Dagster op."""

    class VideoRaw(
        BaseFeature,
        spec=FeatureSpec(
            key=["video", "raw"],
            id_columns=["video_id"],
            fields=[FieldSpec(key=FieldKey(["frames"]), code_version="1")],
        ),
    ):
        video_id: str
        frames: str | None = None

    class VideoChunk(
        BaseFeature,
        spec=FeatureSpec(
            key=["video", "chunk"],
            id_columns=["chunk_id"],
            fields=[FieldSpec(key=FieldKey(["frames"]), code_version="1")],
            deps=[FeatureDep(feature=VideoRaw)],
        ),
    ):
        video_id: str
        chunk_id: str
        frames: str | None = None

    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")

    @dg.job(resource_defs={"metaxy_store": dg.ResourceDefinition.hardcoded_resource(store)})
    def cascade_cleanup_job():
        mxd.delete()

    raw_data = pl.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "frames": ["f1", "f2"],
            METAXY_PROVENANCE_BY_FIELD: [
                {"frames": "p1"},
                {"frames": "p2"},
            ],
        }
    )

    chunk_data = pl.DataFrame(
        {
            "video_id": ["v1", "v1", "v2"],  # Add required video_id column
            "chunk_id": ["c1", "c2", "c3"],
            "frames": ["cf1", "cf2", "cf3"],
            METAXY_PROVENANCE_BY_FIELD: [
                {"frames": "p1"},
                {"frames": "p2"},
                {"frames": "p3"},
            ],
        }
    )

    with store.open("w"):
        store.write(VideoRaw, raw_data)
        store.write(VideoChunk, chunk_data)

    with store:
        assert store.read(VideoRaw).collect().to_polars().height == 2
        assert store.read(VideoChunk).collect().to_polars().height == 3

    result = cascade_cleanup_job.execute_in_process(
        run_config={
            "ops": {
                "delete": {
                    "config": {
                        "feature_key": ["video", "raw"],
                        "filters": [],
                        "soft": True,
                        "cascade": "DOWNSTREAM",
                    }
                }
            }
        }
    )

    assert result.success

    with store:
        raw_remaining = store.read(VideoRaw).collect().to_polars()
        assert raw_remaining.height == 0

        chunk_remaining = store.read(VideoChunk).collect().to_polars()
        assert chunk_remaining.height == 0

        raw_all = store.read(VideoRaw, include_soft_deleted=True).collect().to_polars()
        assert raw_all.height == 2

        chunk_all = store.read(VideoChunk, include_soft_deleted=True).collect().to_polars()
        assert chunk_all.height == 3
