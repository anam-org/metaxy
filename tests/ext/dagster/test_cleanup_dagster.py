"""Dagster cleanup op tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import dagster as dg
import polars as pl
import pytest

import metaxy.ext.dagster as mxd
from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store.delta import DeltaMetadataStore
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


def test_delete_metadata_op_with_mock():
    """Test delete_metadata op with mocked store using simple feature key."""
    store = MagicMock()

    ctx = dg.build_op_context(
        resources={"metaxy_store": store},
        op_config={
            "feature_key": ["logs"],
            "filters": ["level = 'debug'"],
            "soft": True,
        },
    )

    mxd.delete_metadata(ctx)
    store.delete_metadata.assert_called_once()
    args, kwargs = store.delete_metadata.call_args
    assert args[0] == FeatureKey(["logs"])
    assert kwargs["soft"] is True


def test_delete_metadata_op_with_slashed_key():
    """Test delete_metadata op with slashed feature key format."""
    store = MagicMock()

    ctx = dg.build_op_context(
        resources={"metaxy_store": store},
        op_config={
            "feature_key": ["customer", "segment"],
            "filters": ["status = 'inactive'"],
            "soft": True,
        },
    )

    mxd.delete_metadata(ctx)
    store.delete_metadata.assert_called_once()
    args, kwargs = store.delete_metadata.call_args
    assert args[0] == FeatureKey(["customer", "segment"])
    assert kwargs["soft"] is True


def test_delete_metadata_op_hard_delete_with_mock():
    """Test delete_metadata op with hard delete."""
    store = MagicMock()

    ctx = dg.build_op_context(
        resources={"metaxy_store": store},
        op_config={
            "feature_key": ["logs"],
            "filters": ["level = 'warn'"],
            "soft": False,
        },
    )

    mxd.delete_metadata(ctx)
    store.delete_metadata.assert_called_once()
    args, kwargs = store.delete_metadata.call_args
    assert args[0] == FeatureKey(["logs"])
    assert kwargs["soft"] is False


def test_delete_metadata_integration(feature_cls, tmp_path):
    """Integration test: create job, write data, run delete, verify results."""

    # Create a delta store
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")

    # Define a job with the delete op
    @dg.job(resource_defs={"metaxy_store": dg.ResourceDefinition.hardcoded_resource(store)})
    def cleanup_job():
        mxd.delete_metadata()

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

    with store.open("write"):
        store.write_metadata(feature_cls, test_data)

    # Verify data was written
    with store:
        initial_data = store.read_metadata(feature_cls).collect().to_polars()
        assert initial_data.height == 3

    # Execute the job with config to delete debug logs
    result = cleanup_job.execute_in_process(
        run_config={
            "ops": {
                "delete_metadata": {
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
        remaining_data = store.read_metadata(feature_cls).collect().to_polars()
        assert remaining_data.height == 2
        assert set(remaining_data["level"]) == {"info", "warn"}

        # Verify soft-deleted row is still in store when including deleted
        all_data = store.read_metadata(feature_cls, include_soft_deleted=True).collect().to_polars()
        assert all_data.height == 3


def test_delete_metadata_integration_hard_delete(feature_cls, tmp_path):
    """Integration test for hard delete: data is physically removed."""

    # Create a delta store
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")

    # Define a job with the delete op
    @dg.job(resource_defs={"metaxy_store": dg.ResourceDefinition.hardcoded_resource(store)})
    def hard_cleanup_job():
        mxd.delete_metadata()

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

    with store.open("write"):
        store.write_metadata(feature_cls, test_data)

    # Execute the job with hard delete
    result = hard_cleanup_job.execute_in_process(
        run_config={
            "ops": {
                "delete_metadata": {
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
        remaining_data = store.read_metadata(feature_cls).collect().to_polars()
        assert remaining_data.height == 2
        assert set(remaining_data["level"]) == {"info", "warn"}

        # Verify hard-deleted row is NOT in store even when including deleted
        all_data = store.read_metadata(feature_cls, include_soft_deleted=True).collect().to_polars()
        assert all_data.height == 2  # Hard delete means it's gone
