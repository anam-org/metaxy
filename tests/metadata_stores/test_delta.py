"""Delta Lake-specific tests.

Most Delta store functionality is tested via parametrized tests in StoreCases.
This module tests Delta-specific features:
- Object storage integration (S3) with storage_options
- Local absolute path handling
"""

from __future__ import annotations

import polars as pl

from metaxy.metadata_store.delta import DeltaMetadataStore


def test_delta_local_absolute_path(tmp_path, test_features) -> None:
    """Verify DeltaMetadataStore works with local absolute paths like /tmp/rofl.

    This test ensures the store correctly constructs table paths under the root path
    rather than writing to the current directory.
    """
    store_path = tmp_path / "delta_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with DeltaMetadataStore(store_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(feature_cls, metadata)

        # Verify the table was created under the store path, not current directory
        expected_path = store_path / "test_stores" / "upstream_a.delta"
        assert expected_path.exists(), (
            f"Expected table at {expected_path}, but it doesn't exist. "
            f"Store root: {store._root_uri}"
        )

        # Verify we can read the data back
        result = store.read_metadata(feature_cls)
        assert result is not None
        assert result.collect().to_native().height == 1


def test_delta_feature_uri_with_leading_slash_key(tmp_path) -> None:
    """Verify _feature_uri handles feature keys with leading slashes correctly.

    When a FeatureKey like '/feature' is created, it splits into parts ('', 'feature').
    The empty part must be filtered out to prevent creating an absolute path that
    would cause os.path.join to discard the root_uri.

    This was a bug where DeltaMetadataStore('/tmp/rofl') with feature key '/test'
    would write to '/test.delta' instead of '/tmp/rofl/test.delta'.
    """
    from metaxy.models.types import FeatureKey

    store_path = tmp_path / "delta_store"
    store = DeltaMetadataStore(store_path)

    # Keys with leading slashes create empty first parts when split
    test_cases = [
        ("/feature", "feature.delta"),
        ("//double", "double.delta"),
        ("a/b/c", "a/b/c.delta"),
        ("normal", "normal.delta"),
    ]

    for key_str, expected_suffix in test_cases:
        key = FeatureKey(key_str)
        uri = store._feature_uri(key)
        expected_uri = f"{store._root_uri}/{expected_suffix}"
        assert uri == expected_uri, (
            f"For key '{key_str}' (parts={key.parts}), "
            f"expected '{expected_uri}', got '{uri}'"
        )


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
