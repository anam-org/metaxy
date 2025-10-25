"""Test resolve_update for root features (features with no upstream dependencies).

Root features are special because:
1. They have no upstream dependencies to calculate data_version from
2. Users provide data_version directly in their samples
3. resolve_update should compare user-provided data_version with stored metadata
"""

from typing import Any

import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey
from tests.metadata_stores.conftest import StoreCases  # pyrefly: ignore[import-error]


# Define a root feature (no upstream dependencies)
class VideoEmbeddingsFeature(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["test_root", "video_embeddings"]),
        deps=None,  # No upstream dependencies
        fields=[
            FieldSpec(
                key=FieldKey(["embedding"]),
                code_version=1,
            ),
        ],
    ),
):
    """Root feature - video embeddings with no upstream dependencies.

    Users provide data_version directly based on their video files.
    """

    pass


class TestResolveUpdateRootFeatures:
    """Test resolve_update behavior for root features."""

    @parametrize_with_cases("store_config", cases=StoreCases)
    def test_resolve_update_root_feature_requires_samples(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test that resolve_update raises ValueError without samples parameter.

        Root features (no upstream dependencies) require user-provided samples
        with data_version column.
        """
        store_type, config = store_config
        store = store_type(**config)  # type: ignore[abstract]

        graph = FeatureGraph()
        graph.add_feature(VideoEmbeddingsFeature)

        with graph.use(), store:
            # Calling resolve_update on a root feature without samples should raise ValueError
            with pytest.raises(ValueError, match="root feature"):
                store.resolve_update(VideoEmbeddingsFeature)

    @parametrize_with_cases("store_config", cases=StoreCases)
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
                    "data_version": [
                        {"embedding": "hash1"},
                        {"embedding": "hash2"},
                        {"embedding": "hash3"},
                    ],
                }
            )

            result = store.resolve_update(
                VideoEmbeddingsFeature, samples=nw.from_native(user_samples)
            )

            # All samples should be added
            assert len(result.added) == 3
            assert sorted(result.added["sample_uid"].to_list()) == [1, 2, 3]
            assert len(result.changed) == 0
            assert len(result.removed) == 0

    @parametrize_with_cases("store_config", cases=StoreCases)
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
                    "data_version": [
                        {"embedding": "hash1"},
                        {"embedding": "hash2"},
                        {"embedding": "hash3"},
                    ],
                }
            )
            store.write_metadata(VideoEmbeddingsFeature, initial_metadata)

            # User provides updated samples
            import narwhals as nw

            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 4],
                    "data_version": [
                        {"embedding": "hash1"},  # unchanged
                        {"embedding": "hash2_updated"},  # changed
                        {"embedding": "hash4"},  # new
                    ],
                }
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
