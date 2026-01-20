"""Tests for 1:N ExpansionRelationship functionality."""

from pathlib import Path

import narwhals as nw
import polars as pl

from metaxy import BaseFeature
from metaxy._testing import add_metaxy_provenance_column
from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.models.feature_spec import FeatureDep
from metaxy.models.field import FieldSpec
from metaxy.models.lineage import LineageRelationship


class TestExpansionRelationships:
    """Test suite for 1:N expansion relationship functionality."""

    def test_expansion_with_explicit_on_parameter(self, graph, tmp_path: Path):
        """Test ExpansionRelationship with explicit 'on' parameter."""

        class Video(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="video/source",
                id_columns=["video_id"],
                fields=[FieldSpec(key="duration", code_version="1")],
            ),
        ):
            pass

        class VideoFrames(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="video/frames",
                id_columns=["video_id", "frame_id"],
                fields=[FieldSpec(key="frame_data", code_version="1")],
                deps=[
                    FeatureDep(
                        feature="video/source",
                        lineage=LineageRelationship.expansion(
                            on=["video_id"],  # Explicit parent ID
                            id_generation_pattern="sequential",
                        ),
                        # No rename needed - video_id column is the same
                        # frame_id is generated, not mapped
                    )
                ],
            ),
        ):
            pass

        # Test lineage configuration (now on dep, not spec)
        dep = VideoFrames.spec().deps[0]
        lineage = dep.lineage
        assert lineage is not None
        aggregation_cols = lineage.get_aggregation_columns(["video_id", "frame_id"])
        assert aggregation_cols is None, (
            f"Expected None for expansion relationships, got {aggregation_cols}"
        )

        # Test with metadata store
        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Write parent metadata
            video_data = pl.DataFrame(
                {
                    "video_id": ["v1", "v2"],
                    "duration": [30.0, 45.0],
                    "metaxy_provenance_by_field": [
                        {"duration": "hash_v1"},
                        {"duration": "hash_v2"},
                    ],
                }
            )
            video_data = add_metaxy_provenance_column(video_data, Video)
            store.write_metadata(Video, nw.from_native(video_data))

            # Write child metadata (multiple frames per video)
            frames_data = pl.DataFrame(
                {
                    "video_id": ["v1", "v1", "v1", "v2", "v2", "v2"],
                    "frame_id": [0, 1, 2, 0, 1, 2],
                    "frame_data": ["f1", "f2", "f3", "f4", "f5", "f6"],
                    "metaxy_provenance_by_field": [
                        {"frame_data": "old_f1"},
                        {"frame_data": "old_f2"},
                        {"frame_data": "old_f3"},
                        {"frame_data": "old_f4"},
                        {"frame_data": "old_f5"},
                        {"frame_data": "old_f6"},
                    ],
                }
            )
            frames_data = add_metaxy_provenance_column(frames_data, VideoFrames)
            store.write_metadata(VideoFrames, nw.from_native(frames_data))

            # Resolve update
            diff = store.resolve_update(VideoFrames)

            # Should detect changes (parent provenance vs child provenance)
            assert diff.changed is not None
            changed_df = diff.changed.to_polars()
            # Should group by parent ID, so 2 parent changes detected
            assert changed_df.shape[0] == 2

    def test_expansion_with_metadata_store_diff(self, graph, tmp_path: Path):
        """Test ExpansionRelationship diff resolution with metadata store."""

        class Article(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="article",
                id_columns=["article_id"],
                fields=[FieldSpec(key="text", code_version="1")],
            ),
        ):
            pass

        class ArticleParagraphs(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="paragraphs",
                id_columns=["article_id", "para_id"],
                fields=[FieldSpec(key="para_text", code_version="1")],
                deps=[
                    FeatureDep(
                        feature="article",
                        lineage=LineageRelationship.expansion(on=["article_id"]),
                        # No rename needed - article_id column is the same
                    )
                ],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Parent metadata
            article_data = pl.DataFrame(
                {
                    "article_id": ["a1", "a2"],
                    "text": ["Article 1 text", "Article 2 text"],
                    "metaxy_provenance_by_field": [
                        {"text": "hash_a1_v2"},  # Changed
                        {"text": "hash_a2"},  # Unchanged
                    ],
                }
            )
            article_data = add_metaxy_provenance_column(article_data, Article)
            store.write_metadata(Article, nw.from_native(article_data))

            # Child metadata (existing)
            para_data = pl.DataFrame(
                {
                    "article_id": ["a1", "a1", "a1", "a2", "a2"],
                    "para_id": [1, 2, 3, 1, 2],
                    "para_text": ["p1", "p2", "p3", "p4", "p5"],
                    "metaxy_provenance_by_field": [
                        {"para_text": "old_p1"},
                        {"para_text": "old_p2"},
                        {"para_text": "old_p3"},
                        {"para_text": "old_p4"},
                        {"para_text": "old_p5"},
                    ],
                }
            )
            para_data = add_metaxy_provenance_column(para_data, ArticleParagraphs)
            store.write_metadata(ArticleParagraphs, nw.from_native(para_data))

            # Resolve update
            diff = store.resolve_update(ArticleParagraphs)

            # Should detect changes based on parent provenance
            assert diff.changed is not None
            changed_df = diff.changed.to_polars()

            # Should have 2 groups (one per parent)
            assert changed_df.shape[0] == 2

            # Verify parent ID is in the result
            assert "article_id" in changed_df.columns

    def test_expansion_relationship_get_aggregation_columns(self, graph):
        """Test get_aggregation_columns method behavior."""
        from metaxy.models.lineage import ExpansionRelationship

        # With explicit 'on' parameter
        rel1 = ExpansionRelationship(
            on=["video_id"], id_generation_pattern="sequential"
        )
        result1 = rel1.get_aggregation_columns(["video_id", "frame_id"])
        assert result1 is None  # Expansion relationships don't aggregate during join

        # With 'on' parameter for different ID structure
        rel2 = ExpansionRelationship(on=["doc_id"], id_generation_pattern="hash")
        result2 = rel2.get_aggregation_columns(["doc_id", "chunk_id"])
        assert result2 is None  # Expansion relationships don't aggregate during join

        # Empty 'on' parameter
        rel3 = ExpansionRelationship(on=[], id_generation_pattern="custom")
        result3 = rel3.get_aggregation_columns(["id1", "id2"])
        assert result3 is None  # Expansion relationships don't aggregate during join
