"""Delta metadata store tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from metaxy import HashAlgorithm
from metaxy.ext.polars.handlers.delta import DeltaMetadataStore
from metaxy.metadata_store import MetadataStore
from metaxy.models.feature_definition import FeatureDefinition
from metaxy.models.types import FeatureKey
from tests.metadata_stores.shared import (
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    MapDtypeTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
)


@pytest.fixture
def delta_store(tmp_path: Path) -> DeltaMetadataStore:
    """A DeltaMetadataStore backed by a local temporary directory."""
    return DeltaMetadataStore(root_path=tmp_path / "delta_store")


@pytest.fixture
def delta_s3_store(s3_bucket_and_storage_options: tuple[str, dict[str, str]]) -> DeltaMetadataStore:
    """A DeltaMetadataStore backed by an S3 bucket."""
    bucket_name, storage_options = s3_bucket_and_storage_options
    return DeltaMetadataStore(
        root_path=f"s3://{bucket_name}/delta_store",
        storage_options=storage_options,
    )


@pytest.mark.delta
@pytest.mark.polars
class TestDelta(
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    MapDtypeTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
):
    @pytest.fixture
    def store(self, tmp_path: Path) -> MetadataStore:
        return DeltaMetadataStore(
            root_path=tmp_path / "delta_store",
            hash_algorithm=HashAlgorithm.XXHASH64,
        )

    def cleanup_feature(self, store: MetadataStore, feature: FeatureDefinition) -> None:
        """Remove the Delta table directory to allow schema changes between iterations.

        Delta's soft-delete retains the table schema, so writing a Map column with
        a different key/value type would silently coerce data to the old schema.
        """
        assert isinstance(store, DeltaMetadataStore)
        table_path = Path(store._feature_uri(feature.spec.key))
        if table_path.exists():
            shutil.rmtree(table_path)


def test_delta_local_absolute_path(
    delta_store: DeltaMetadataStore, test_features: dict[str, FeatureDefinition]
) -> None:
    """Verify DeltaMetadataStore works with local absolute paths like /tmp/rofl.

    This test ensures the store correctly constructs table paths under the root path
    rather than writing to the current directory.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    store_path = Path(delta_store._root_uri)

    with delta_store.open("w") as store:
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


def test_delta_feature_uri_generation(delta_store: DeltaMetadataStore) -> None:
    """Verify _feature_uri generates correct URIs for valid feature keys."""
    # Valid feature keys and their expected URI suffixes
    test_cases = [
        ("feature", "feature.delta"),
        ("a/b/c", "a/b/c.delta"),
        ("normal", "normal.delta"),
        ("my_feature/v1", "my_feature/v1.delta"),
    ]

    for key_str, expected_suffix in test_cases:
        key = FeatureKey(key_str)
        uri = delta_store._feature_uri(key)
        expected_uri = f"{delta_store._root_uri}/{expected_suffix}"
        assert uri == expected_uri, f"For key '{key_str}' (parts={key.parts}), expected '{expected_uri}', got '{uri}'"


def test_delta_feature_key_validation_rejects_empty_parts() -> None:
    """Ensure FeatureKey rejects invalid keys with empty parts.

    Leading slashes like '/feature' would create empty parts ('', 'feature')
    which are now rejected at validation time to prevent path bugs.
    """
    # Keys with leading slashes or empty parts should be rejected
    invalid_keys = [
        "/feature",  # Leading slash creates empty first part
        "//double",  # Double slash creates empty parts
        "a//b",  # Empty part in the middle
    ]

    for key_str in invalid_keys:
        with pytest.raises(Exception):  # ValidationError
            FeatureKey(key_str)


def test_delta_s3_storage_options_passed(
    delta_s3_store: DeltaMetadataStore, test_features: dict[str, FeatureDefinition]
) -> None:
    """Verify storage_options are passed to Delta operations with S3.

    This ensures object store credentials are correctly forwarded to delta-rs.
    """
    feature_cls = test_features["UpstreamFeatureA"]

    with delta_s3_store.open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(feature_cls, metadata)
        assert store.has_feature(feature_cls, check_fallback=False)


def test_delta_sink_lazyframe_local(
    delta_store: DeltaMetadataStore, test_features: dict[str, FeatureDefinition]
) -> None:
    """Verify LazyFrame.sink_delta works with local storage.

    With Polars >= 1.37, lazy frames should be sinked directly without materialization.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    store_path = Path(delta_store._root_uri)

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

    with delta_store.open("w") as store:
        store.write(feature_cls, metadata_lazy)

        # Verify the table was created
        expected_path = store_path / "test_stores" / "upstream_a.delta"
        assert expected_path.exists()

        # Verify we can read the data back
        result = store.read(feature_cls)
        assert result is not None
        assert result.collect().to_native().height == 3


def test_delta_sink_lazyframe_s3(
    delta_s3_store: DeltaMetadataStore, test_features: dict[str, FeatureDefinition]
) -> None:
    """Verify LazyFrame.sink_delta works with S3 storage.

    With Polars >= 1.37, lazy frames should be sinked directly without materialization.
    """
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

    with delta_s3_store.open("w") as store:
        store.write(feature_cls, metadata_lazy)

        # Verify the feature exists
        assert store.has_feature(feature_cls, check_fallback=False)

        # Verify we can read the data back
        result = store.read(feature_cls)
        assert result is not None
        assert result.collect().to_native().height == 3
