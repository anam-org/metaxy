"""Delta Lake-specific tests.

Most Delta store functionality is tested via parametrized tests in StoreCases.
This module tests Delta-specific features:
- Object storage integration (S3) with storage_options
"""

from __future__ import annotations

import polars as pl

from metaxy.metadata_store.delta import DeltaMetadataStore


def test_delta_s3_storage_options_passed(moto_s3_bucket, test_features) -> None:
    """Verify storage_options are passed to Delta operations with S3.

    This ensures object store credentials are correctly forwarded to delta-rs.
    Uses moto server running as subprocess with proper threading support.
    """
    store_path = f"s3://{moto_s3_bucket['bucket']}/delta_store"
    feature_cls = test_features["UpstreamFeatureA"]
    storage_options = moto_s3_bucket["storage_options"]

    # Simple write to verify storage_options work
    with DeltaMetadataStore(store_path, storage_options=storage_options) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(feature_cls, metadata)

        # Verify the write succeeded by checking feature exists
        assert store.has_feature(feature_cls, check_fallback=False)
