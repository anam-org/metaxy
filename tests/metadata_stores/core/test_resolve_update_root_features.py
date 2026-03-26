"""Test resolve_update for root features (features with no upstream dependencies).

Root features are special because:
1. They have no upstream dependencies to calculate provenance_by_field from
2. Users provide provenance_by_field directly in their samples
3. resolve_update should compare user-provided provenance_by_field with stored metadata
"""

from typing import Any

import polars as pl
import pytest
from metaxy_testing import add_metaxy_provenance_column
from metaxy_testing.models import SampleFeature, SampleFeatureSpec
from pytest_cases import parametrize_with_cases

from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import FeatureGraph
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey
from tests.metadata_stores.conftest import (
    BasicStoreCases,
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
        store = store_type(**config)

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
        store = store_type(**config)

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

            result = store.resolve_update(video_feature, samples=nw.from_native(user_samples))

            # All samples should be added
            assert len(result.new) == 3
            assert sorted(result.new["sample_uid"].to_list()) == [1, 2, 3]
            assert len(result.stale) == 0
            assert len(result.orphaned) == 0

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
        store = store_type(**config)

        graph = FeatureGraph()
        video_feature = create_video_embeddings_feature(graph)

        with graph.use(), store.open("w"):
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
            initial_metadata = add_metaxy_provenance_column(initial_metadata, video_feature)
            store.write(video_feature, initial_metadata)

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

            result = store.resolve_update(video_feature, samples=nw.from_native(user_samples))

            # Should detect changes correctly
            assert len(result.new) == 1
            assert result.new["sample_uid"].to_list() == [4]

            assert len(result.stale) == 1
            assert result.stale["sample_uid"].to_list() == [2]

            assert len(result.orphaned) == 1
            assert result.orphaned["sample_uid"].to_list() == [3]

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_root_feature_adds_data_version_columns(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test that resolve_update adds metaxy_data_version columns for root features.

        For root features (no upstream computation), data_version columns should
        equal provenance columns by default.
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()
        video_feature = create_video_embeddings_feature(graph)

        with graph.use(), store:
            import narwhals as nw

            from metaxy.models.constants import (
                METAXY_DATA_VERSION,
                METAXY_DATA_VERSION_BY_FIELD,
                METAXY_PROVENANCE,
                METAXY_PROVENANCE_BY_FIELD,
            )

            # User provides samples with only provenance columns
            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"embedding": "hash1"},
                        {"embedding": "hash2"},
                        {"embedding": "hash3"},
                    ],
                }
            )
            user_samples = add_metaxy_provenance_column(user_samples, video_feature)

            result = store.resolve_update(video_feature, samples=nw.from_native(user_samples))

            # Verify data_version columns are present in added samples
            assert METAXY_DATA_VERSION in result.new.columns
            assert METAXY_DATA_VERSION_BY_FIELD in result.new.columns

            # Verify data_version values equal provenance values
            added_df = result.new.to_polars()
            for row in added_df.iter_rows(named=True):
                assert row[METAXY_DATA_VERSION] == row[METAXY_PROVENANCE]
                assert row[METAXY_DATA_VERSION_BY_FIELD] == row[METAXY_PROVENANCE_BY_FIELD]

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_root_feature_preserves_custom_data_version(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test that resolve_update preserves custom metaxy_data_version columns for root features.

        When users provide custom data_version columns, they should be preserved.
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()
        video_feature = create_video_embeddings_feature(graph)

        with graph.use(), store:
            import narwhals as nw

            from metaxy.models.constants import (
                METAXY_DATA_VERSION,
                METAXY_DATA_VERSION_BY_FIELD,
                METAXY_PROVENANCE,
                METAXY_PROVENANCE_BY_FIELD,
            )

            # User provides samples with both provenance and custom data_version columns
            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"embedding": "prov_hash1"},
                        {"embedding": "prov_hash2"},
                    ],
                    # Custom data_version that differs from provenance
                    METAXY_DATA_VERSION_BY_FIELD: [
                        {"embedding": "custom_v1"},
                        {"embedding": "custom_v1"},  # Same version for both
                    ],
                }
            )
            user_samples = add_metaxy_provenance_column(user_samples, video_feature)
            # Add data_version hash column
            user_samples = user_samples.with_columns(pl.concat_str([pl.lit("custom_v1")]).alias(METAXY_DATA_VERSION))

            result = store.resolve_update(video_feature, samples=nw.from_native(user_samples))

            # Verify custom data_version columns are preserved
            assert METAXY_DATA_VERSION in result.new.columns
            assert METAXY_DATA_VERSION_BY_FIELD in result.new.columns

            # Verify custom data_version values are preserved (different from provenance)
            added_df = result.new.to_polars()
            for row in added_df.iter_rows(named=True):
                # Data version should be the custom value, not provenance
                assert row[METAXY_DATA_VERSION] == "custom_v1"
                assert row[METAXY_DATA_VERSION_BY_FIELD] == {"embedding": "custom_v1"}
                # Provenance should be the original hash values
                assert row[METAXY_DATA_VERSION] != row[METAXY_PROVENANCE]

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_root_feature_skip_comparison_adds_data_version(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test that resolve_update with skip_comparison=True adds data_version columns for root features."""
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()
        video_feature = create_video_embeddings_feature(graph)

        with graph.use(), store:
            import narwhals as nw

            from metaxy.models.constants import (
                METAXY_DATA_VERSION,
                METAXY_DATA_VERSION_BY_FIELD,
                METAXY_PROVENANCE,
                METAXY_PROVENANCE_BY_FIELD,
            )

            # User provides samples without data_version columns
            user_samples = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"embedding": "hash1"},
                        {"embedding": "hash2"},
                    ],
                }
            )
            user_samples = add_metaxy_provenance_column(user_samples, video_feature)

            # Use skip_comparison=True for backfilling scenarios
            result = store.resolve_update(
                video_feature,
                samples=nw.from_native(user_samples),
                skip_comparison=True,
            )

            # All samples should be in added (skip_comparison returns all as added)
            assert len(result.new) == 2

            # Verify data_version columns are present
            assert METAXY_DATA_VERSION in result.new.columns
            assert METAXY_DATA_VERSION_BY_FIELD in result.new.columns

            # Verify data_version values equal provenance values
            added_df = result.new.to_polars()
            for row in added_df.iter_rows(named=True):
                assert row[METAXY_DATA_VERSION] == row[METAXY_PROVENANCE]
                assert row[METAXY_DATA_VERSION_BY_FIELD] == row[METAXY_PROVENANCE_BY_FIELD]

    @parametrize_with_cases("store_config", cases=BasicStoreCases)
    def test_resolve_update_accepts_polars_frame_directly(
        self,
        store_config: tuple[type[MetadataStore], dict[str, Any]],
    ):
        """Test that resolve_update accepts a Polars DataFrame directly without nw.from_native().

        The samples parameter now accepts IntoFrame, so users can pass Polars frames directly.
        """
        store_type, config = store_config
        store = store_type(**config)

        graph = FeatureGraph()
        video_feature = create_video_embeddings_feature(graph)

        with graph.use(), store:
            # Pass a Polars DataFrame directly without wrapping with nw.from_native()
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

            # Pass Polars frame directly - should work without nw.from_native()
            result = store.resolve_update(video_feature, samples=user_samples)

            # All samples should be added
            assert len(result.new) == 3
            assert sorted(result.new["sample_uid"].to_list()) == [1, 2, 3]
            assert len(result.stale) == 0
            assert len(result.orphaned) == 0
