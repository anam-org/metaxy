"""Delta Lake-specific tests.

Most Delta store functionality is tested via parametrized tests in StoreCases.
This module tests Delta-specific features:
- Object storage integration (S3) with storage_options
"""

from __future__ import annotations

import polars as pl

from metaxy.metadata_store.delta import DeltaMetadataStore


def test_delta_s3_storage_options_passed(
    s3_bucket_and_storage_options, test_features
) -> None:
    """Verify storage_options are passed to Delta operations with S3.

    This ensures object store credentials are correctly forwarded to delta-rs.
    """
    bucket_name, storage_options = s3_bucket_and_storage_options
    store_path = f"s3://{bucket_name}/delta_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with DeltaMetadataStore(store_path, storage_options=storage_options) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(feature_cls, metadata)
        assert store.has_feature(feature_cls, check_fallback=False)
