"""Delta Lake-specific tests.

Most Delta store functionality is tested via parametrized tests in StoreCases.
This module tests Delta-specific features:
- Object storage integration (S3) with storage_options
"""

from __future__ import annotations

import polars as pl

from metaxy.metadata_store.delta import DeltaMetadataStore


def test_delta_s3_storage_options_passed(s3_bucket, test_features) -> None:
    """Verify storage_options are passed to Delta operations with S3.

    This ensures object store credentials are correctly forwarded to delta-rs.
    """
    store_path = f"s3://{s3_bucket['bucket_name']}/delta_store"
    feature_cls = test_features["UpstreamFeatureA"]
    storage_options = s3_bucket["storage_options"]

    with DeltaMetadataStore(store_path, storage_options=storage_options) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(feature_cls, metadata)

        # Deadlock here, lets manually check with boto3
        # assert store.has_feature(feature_cls, check_fallback=False)

    s3_resource = s3_bucket["s3_resource"]
    bucket_name = s3_bucket["bucket_name"]
    bucket = s3_resource.Bucket(bucket_name)

    objects_in_bucket = [obj.key for obj in bucket.objects.all()]

    log_file_found = any(
        "_delta_log/00000000000000000000.json" in key for key in objects_in_bucket
    )
    assert log_file_found, "The Delta transaction log file was not created."

    data_file_found = any(key.endswith(".parquet") for key in objects_in_bucket)
    assert data_file_found, "A Parquet data file was not created."


def test_delta_hanging(s3_bucket, test_features) -> None:
    """this one is hanging on assert"""
    store_path = f"s3://{s3_bucket['bucket_name']}/delta_store"
    feature_cls = test_features["UpstreamFeatureA"]
    storage_options = s3_bucket["storage_options"]

    with DeltaMetadataStore(store_path, storage_options=storage_options) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(feature_cls, metadata)

        # Deadlock here, lets manually check with boto3
        assert store.has_feature(feature_cls, check_fallback=False)
