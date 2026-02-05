"""LanceDB S3 integration tests using moto."""

from __future__ import annotations

import polars as pl

from metaxy._utils import collect_to_polars
from metaxy.ext.metadata_stores.lancedb import LanceDBMetadataStore


def test_lancedb_s3_roundtrip_with_moto(s3_bucket_and_storage_options, test_features) -> None:
    """Ensure LanceDB works end-to-end against moto-backed S3."""
    bucket_name, storage_options = s3_bucket_and_storage_options
    store_uri = f"s3://{bucket_name}/lancedb_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with LanceDBMetadataStore(store_uri, connect_kwargs={"storage_options": storage_options}) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        store.write(feature_cls, metadata)

        assert store.has_feature(feature_cls, check_fallback=False)

        result = collect_to_polars(store.read(feature_cls))
        assert result.shape[0] == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}
