"""Test resolve_update for root features (features with no upstream dependencies).

Root features are special because:
1. They have no upstream dependencies to calculate provenance_by_field from
2. Users provide provenance_by_field directly in their samples
3. resolve_update should compare user-provided provenance_by_field with stored metadata
"""

from typing import Any

import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from metaxy._testing import add_metaxy_provenance_column
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import FeatureGraph, TestingFeature
from metaxy.models.feature_spec import SampleFeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey
from tests.metadata_stores.conftest import (
    BasicStoreCases,  # pyrefly: ignore[import-error]
)


# Define a root feature (no upstream dependencies)
class VideoEmbeddingsFeature(
    TestingFeature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["test_root", "video_embeddings"]),
        # No upstream dependencies
        fields=[
            FieldSpec(
                key=FieldKey(["embedding"]),
                code_version="1",
            ),
        ],
    ),
):
    """Root feature - video embeddings with no upstream dependencies.

    Users provide provenance_by_field directly based on their video files.
    """

    pass


class TestResolveUpdateRootFeatures:
    """Test resolve_update behavior for root features."""

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_root_feature_requires_samples(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test that resolve_update raises ValueError without samples parameter.

        Root features (no upstream dependencies) require user-provided samples
        with provenance_by_field column.
        """
        store_type, config = store_config
        store = store_type(**config)  # type: ignore[abstract]

        graph = FeatureGraph()
        graph.add_feature(VideoEmbeddingsFeature)

        with graph.use(), store:
            # Calling resolve_update on a root feature without samples should raise ValueError
            with pytest.raises(ValueError, match="root feature"):
                store.resolve_update(VideoEmbeddingsFeature)

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_root_feature_with_samples_no_existing_metadata(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test resolve_update with samples for root feature (no existing metadata).

        When store is empty, all user samples should be 'added'.
        """
        store_type, config = store_config
        store = store_type(**config)  # type: ignore[abstract]

        graph = FeatureGraph()
        graph.add_feature(VideoEmbeddingsFeature)

        with graph.use(), store:
            # User provides samples
            import narwhals as nw

            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "metaxy_provenance_by_field": [
                        {"embedding": "hash1"},
                        {"embedding": "hash2"},
                        {"embedding": "hash3"},
                    ],
                }
            )
            user_samples = add_metaxy_provenance_column(
                user_samples, VideoEmbeddingsFeature
            )

            result = store.resolve_update(
                VideoEmbeddingsFeature, samples=nw.from_native(user_samples)
            )

            # All samples should be added
            assert len(result.added) == 3
            assert sorted(result.added["sample_uid"].to_list()) == [1, 2, 3]
            assert len(result.changed) == 0
            assert len(result.removed) == 0

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_root_feature_with_samples_and_changes(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test resolve_update with samples parameter detects all changes correctly.

        Tests added, changed, and removed samples.
        """
        store_type, config = store_config
        store = store_type(**config)  # type: ignore[abstract]

        graph = FeatureGraph()
        graph.add_feature(VideoEmbeddingsFeature)

        with graph.use(), store:
            # Write initial metadata
            initial_metadata = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "metaxy_provenance_by_field": [
                        {"embedding": "hash1"},
                        {"embedding": "hash2"},
                        {"embedding": "hash3"},
                    ],
                }
            )
            initial_metadata = add_metaxy_provenance_column(
                initial_metadata, VideoEmbeddingsFeature
            )
            store.write_metadata(VideoEmbeddingsFeature, initial_metadata)

            # User provides updated samples
            import narwhals as nw

            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 4],
                    "metaxy_provenance_by_field": [
                        {"embedding": "hash1"},  # unchanged
                        {"embedding": "hash2_updated"},  # changed
                        {"embedding": "hash4"},  # new
                    ],
                }
            )
            user_samples = add_metaxy_provenance_column(
                user_samples, VideoEmbeddingsFeature
            )

            result = store.resolve_update(
                VideoEmbeddingsFeature, samples=nw.from_native(user_samples)
            )

            # Should detect changes correctly
            assert len(result.added) == 1
            assert result.added["sample_uid"].to_list() == [4]

            assert len(result.changed) == 1
            assert result.changed["sample_uid"].to_list() == [2]

            assert len(result.removed) == 1
            assert result.removed["sample_uid"].to_list() == [3]
