"""Vortex-specific tests.

Core metadata store functionality (provenance, deduplication, resolve_update) is tested
via the golden reference tests in test_provenance_golden_reference.py where Vortex
is included in the `any_store` fixture.

This module tests Vortex-specific features only:
- Storage layout (nested vs flat directory structure)
- Feature URI generation (path construction edge cases)
- Drop feature (file deletion)
"""

from __future__ import annotations

import sys

import polars as pl
import pytest

from metaxy.metadata_store.vortex import VortexMetadataStore

pytest.importorskip("vortex", reason="vortex-data not installed")

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="VortexMetadataStore requires Python 3.11+",
)


def test_vortex_feature_uri_generation(tmp_path) -> None:
    """Verify _feature_uri generates correct URIs for various feature key formats."""
    from metaxy.models.types import FeatureKey

    store_path = tmp_path / "vortex_store"
    store = VortexMetadataStore(store_path)

    test_cases = [
        ("feature", "feature.vortex"),
        ("a/b/c", "a/b/c.vortex"),
        ("my_feature/v1", "my_feature/v1.vortex"),
    ]

    for key_str, expected_suffix in test_cases:
        key = FeatureKey(key_str)
        uri = store._feature_uri(key)
        expected_uri = f"{store._root_uri}/{expected_suffix}"
        assert uri == expected_uri, (
            f"For key '{key_str}' (parts={key.parts}), "
            f"expected '{expected_uri}', got '{uri}'"
        )


def test_vortex_flat_layout(tmp_path, test_features) -> None:
    """Test that flat layout creates files with __ separator."""
    store_path = tmp_path / "vortex_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with VortexMetadataStore(store_path, layout="flat") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(feature_cls, metadata)

        expected_path = store_path / "test_stores__upstream_a.vortex"
        assert expected_path.exists(), f"Expected flat layout file at {expected_path}"


def test_vortex_nested_layout(tmp_path, test_features) -> None:
    """Test that nested layout creates directory hierarchy."""
    store_path = tmp_path / "vortex_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with VortexMetadataStore(store_path, layout="nested") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(feature_cls, metadata)

        expected_path = store_path / "test_stores" / "upstream_a.vortex"
        assert expected_path.exists(), f"Expected nested layout file at {expected_path}"
        assert expected_path.parent.is_dir()


def test_vortex_drop_feature(tmp_path, test_features) -> None:
    """Test that drop_feature_metadata removes the Vortex file."""
    store_path = tmp_path / "vortex_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with VortexMetadataStore(store_path) as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(feature_cls, metadata)
        assert store.has_feature(feature_cls)

        store.drop_feature_metadata(feature_cls)
        assert not store.has_feature(feature_cls)


def test_vortex_invalid_layout(tmp_path) -> None:
    """Test that invalid layout raises ValueError."""
    store_path = tmp_path / "vortex_store"

    with pytest.raises(ValueError, match="Invalid layout.*Must be 'flat' or 'nested'"):
        VortexMetadataStore(store_path, layout="invalid")  # pyright: ignore[reportArgumentType]


def test_vortex_remote_path_raises_not_implemented() -> None:
    """Test that remote paths raise NotImplementedError with clear message."""
    with pytest.raises(
        NotImplementedError, match="Remote storage paths are not yet supported"
    ):
        VortexMetadataStore("s3://bucket/path")

    with pytest.raises(
        NotImplementedError, match="Remote storage paths are not yet supported"
    ):
        VortexMetadataStore("gs://bucket/path")

    with pytest.raises(
        NotImplementedError, match="Remote storage paths are not yet supported"
    ):
        VortexMetadataStore("az://container/path")
