"""Delta Lake-specific tests.

Most Delta store functionality is tested via parametrized tests in StoreCases.
This module tests Delta-specific features:
- Object storage integration (S3) with storage_options
- Local absolute path handling
"""

from __future__ import annotations

import polars as pl

from metaxy.ext.metadata_stores.delta import DeltaMetadataStore


def test_delta_local_absolute_path(tmp_path, test_features) -> None:
    """Verify DeltaMetadataStore works with local absolute paths like /tmp/rofl.

    This test ensures the store correctly constructs table paths under the root path
    rather than writing to the current directory.
    """
    store_path = tmp_path / "delta_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with DeltaMetadataStore(store_path).open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(feature_cls, metadata)

        # Verify the table was created under the store path, not current directory
        expected_path = store_path / "test_stores" / "upstream_a.delta"
        assert expected_path.exists(), (
            f"Expected table at {expected_path}, but it doesn't exist. Store root: {store._root_uri}"
        )

        # Verify we can read the data back
        result = store.read(feature_cls)
        assert result is not None
        assert result.collect().to_native().height == 1


def test_delta_feature_uri_generation(tmp_path) -> None:
    """Verify _feature_uri generates correct URIs for valid feature keys."""
    from metaxy.models.types import FeatureKey

    store_path = tmp_path / "delta_store"
    store = DeltaMetadataStore(store_path)

    # Valid feature keys and their expected URI suffixes
    test_cases = [
        ("feature", "feature.delta"),
        ("a/b/c", "a/b/c.delta"),
        ("normal", "normal.delta"),
        ("my_feature/v1", "my_feature/v1.delta"),
    ]

    for key_str, expected_suffix in test_cases:
        key = FeatureKey(key_str)
        uri = store._feature_uri(key)
        expected_uri = f"{store._root_uri}/{expected_suffix}"
        assert uri == expected_uri, f"For key '{key_str}' (parts={key.parts}), expected '{expected_uri}', got '{uri}'"


def test_delta_feature_key_validation_rejects_empty_parts(tmp_path) -> None:
    """Ensure FeatureKey rejects invalid keys with empty parts.

    Leading slashes like '/feature' would create empty parts ('', 'feature')
    which are now rejected at validation time to prevent path bugs.
    """
    import pytest

    from metaxy.models.types import FeatureKey

    # Keys with leading slashes or empty parts should be rejected
    invalid_keys = [
        "/feature",  # Leading slash creates empty first part
        "//double",  # Double slash creates empty parts
        "a//b",  # Empty part in the middle
    ]

    for key_str in invalid_keys:
        with pytest.raises(Exception):  # ValidationError
            FeatureKey(key_str)


def test_delta_s3_storage_options_passed(s3_bucket_and_storage_options, test_features) -> None:
    """Verify storage_options are passed to Delta operations with S3.

    This ensures object store credentials are correctly forwarded to delta-rs.
    """
    bucket_name, storage_options = s3_bucket_and_storage_options
    store_path = f"s3://{bucket_name}/delta_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with DeltaMetadataStore(store_path, storage_options=storage_options).open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(feature_cls, metadata)
        assert store.has_feature(feature_cls, check_fallback=False)


def test_delta_sink_lazyframe_local(tmp_path, test_features) -> None:
    """Verify LazyFrame.sink_delta works with local storage.

    With Polars >= 1.37, lazy frames should be sinked directly without materialization.
    """
    store_path = tmp_path / "delta_store"
    feature_cls = test_features["UpstreamFeatureA"]

    # Create a lazy frame (native Polars)
    metadata_lazy = pl.LazyFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field": [
                {"frames": "h1", "audio": "h1"},
                {"frames": "h2", "audio": "h2"},
                {"frames": "h3", "audio": "h3"},
            ],
        }
    )

    with DeltaMetadataStore(store_path).open("w") as store:
        store.write(feature_cls, metadata_lazy)

        # Verify the table was created
        expected_path = store_path / "test_stores" / "upstream_a.delta"
        assert expected_path.exists()

        # Verify we can read the data back
        result = store.read(feature_cls)
        assert result is not None
        assert result.collect().to_native().height == 3


def test_delta_sink_lazyframe_s3(s3_bucket_and_storage_options, test_features) -> None:
    """Verify LazyFrame.sink_delta works with S3 storage.

    With Polars >= 1.37, lazy frames should be sinked directly without materialization.
    """
    bucket_name, storage_options = s3_bucket_and_storage_options
    store_path = f"s3://{bucket_name}/delta_store"
    feature_cls = test_features["UpstreamFeatureA"]

    # Create a lazy frame (native Polars)
    metadata_lazy = pl.LazyFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field": [
                {"frames": "h1", "audio": "h1"},
                {"frames": "h2", "audio": "h2"},
                {"frames": "h3", "audio": "h3"},
            ],
        }
    )

    with DeltaMetadataStore(store_path, storage_options=storage_options).open("w") as store:
        store.write(feature_cls, metadata_lazy)

        # Verify the feature exists
        assert store.has_feature(feature_cls, check_fallback=False)

        # Verify we can read the data back
        result = store.read(feature_cls)
        assert result is not None
        assert result.collect().to_native().height == 3
