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
from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import FeatureGraph
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey
from tests.metadata_stores.conftest import (
    BasicStoreCases,  # pyrefly: ignore[import-error]
)


def create_video_embeddings_feature(graph: FeatureGraph) -> type[SampleFeature]:
    """Create the test root feature within the provided graph context."""
    with graph.use():

        class VideoEmbeddingsFeature(
            SampleFeature,
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

        return VideoEmbeddingsFeature


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
        video_feature = create_video_embeddings_feature(graph)

        with graph.use(), store:
            # Calling resolve_update on a root feature without samples should raise ValueError
            with pytest.raises(ValueError, match="root feature"):
                store.resolve_update(video_feature)

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
        video_feature = create_video_embeddings_feature(graph)

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
            user_samples = add_metaxy_provenance_column(user_samples, video_feature)

            result = store.resolve_update(
                video_feature, samples=nw.from_native(user_samples)
            )

            # All samples should be added
            assert len(result.added) == 3
            assert sorted(result.added["sample_uid"].to_list()) == [1, 2, 3]
            assert len(result.changed) == 0
            assert len(result.removed) == 0

    @pytest.mark.skip(
        reason="Requires groupby id_columns+feature_version and taking latest sample by created_at. "
        "Support for changing provenance values without changing code versions will be added later."
    )
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
        video_feature = create_video_embeddings_feature(graph)

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
                initial_metadata, video_feature
            )
            store.write_metadata(video_feature, initial_metadata)

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
            user_samples = add_metaxy_provenance_column(user_samples, video_feature)

            result = store.resolve_update(
                video_feature, samples=nw.from_native(user_samples)
            )

            # Should detect changes correctly
            assert len(result.added) == 1
            assert result.added["sample_uid"].to_list() == [4]

            assert len(result.changed) == 1
            assert result.changed["sample_uid"].to_list() == [2]

            assert len(result.removed) == 1
            assert result.removed["sample_uid"].to_list() == [3]
