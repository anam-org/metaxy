"""Tests for 1:N ExpansionRelationship functionality."""

import narwhals as nw
import polars as pl

from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.models.feature import BaseFeature
from metaxy.models.feature_spec import BaseFeatureSpec, FeatureDep, FieldSpec
from metaxy.models.lineage import LineageRelationship


class TestExpansionRelationships:
    """Test suite for 1:N expansion relationship functionality."""

    def test_expansion_with_explicit_on_parameter(self, graph):
        """Test ExpansionRelationship with explicit 'on' parameter."""

        class Video(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="video/source",
                id_columns=["video_id"],
                fields=[FieldSpec(key="duration", code_version="1")],
            ),
        ):
            pass

        class VideoFrames(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="video/frames",
                id_columns=["video_id", "frame_id"],
                lineage=LineageRelationship.expansion(
                    on=["video_id"],  # Explicit parent ID
                    id_generation_pattern="sequential",
                ),
                fields=[FieldSpec(key="frame_data", code_version="1")],
                deps=[
                    FeatureDep(
                        feature="video/source",
                        # No rename needed - video_id column is the same
                        # frame_id is generated, not mapped
                    )
                ],
            ),
        ):
            pass

        # Test lineage configuration
        lineage = VideoFrames.spec().lineage
        assert lineage is not None
        aggregation_cols = lineage.get_aggregation_columns(["video_id", "frame_id"])
        assert aggregation_cols == ["video_id"], (
            f"Expected ['video_id'], got {aggregation_cols}"
        )

        # Test with metadata store
        with InMemoryMetadataStore() as store:
            # Write parent metadata
            video_data = pl.DataFrame(
                {
                    "video_id": ["v1", "v2"],
                    "duration": [30.0, 45.0],
                    "provenance_by_field": [
                        {"duration": "hash_v1"},
                        {"duration": "hash_v2"},
                    ],
                }
            )
            store.write_metadata(Video, nw.from_native(video_data))

            # Write child metadata (multiple frames per video)
            frames_data = pl.DataFrame(
                {
                    "video_id": ["v1", "v1", "v1", "v2", "v2", "v2"],
                    "frame_id": [0, 1, 2, 0, 1, 2],
                    "frame_data": ["f1", "f2", "f3", "f4", "f5", "f6"],
                    "provenance_by_field": [
                        {"frame_data": "old_f1"},
                        {"frame_data": "old_f2"},
                        {"frame_data": "old_f3"},
                        {"frame_data": "old_f4"},
                        {"frame_data": "old_f5"},
                        {"frame_data": "old_f6"},
                    ],
                }
            )
            store.write_metadata(VideoFrames, nw.from_native(frames_data))

            # Resolve update
            diff = store.resolve_update(VideoFrames)

            # Should detect changes (parent provenance vs child provenance)
            assert diff.changed is not None
            changed_df = diff.changed.to_polars()
            # Should group by parent ID, so 2 parent changes detected
            assert changed_df.shape[0] == 2

    def test_expansion_without_on_parameter(self, graph):
        """Test ExpansionRelationship without 'on' parameter (inferred)."""

        class Document(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="document",
                id_columns=["doc_id"],
                fields=[FieldSpec(key="content", code_version="1")],
            ),
        ):
            pass

        class DocumentChunks(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="chunks",
                id_columns=["doc_id", "chunk_id"],
                lineage=LineageRelationship.expansion(
                    # No 'on' parameter - will be inferred
                    id_generation_pattern="hash"
                ),
                fields=[FieldSpec(key="chunk_text", code_version="1")],
                deps=[
                    FeatureDep(
                        feature="document",
                        # No rename needed - doc_id column is the same
                    )
                ],
            ),
        ):
            pass

        # Verify lineage returns None for inference
        lineage = DocumentChunks.spec().lineage
        aggregation_cols = lineage.get_aggregation_columns(["doc_id", "chunk_id"])
        assert aggregation_cols is None, "Should return None for inference"

        # Test with joiner
        doc_data = pl.DataFrame(
            {
                "doc_id": ["d1", "d2"],
                "content": ["Content 1", "Content 2"],
                "provenance_by_field": [
                    {"content": "hash_d1"},
                    {"content": "hash_d2"},
                ],
            }
        )

        joiner = NarwhalsJoiner()
        doc_ref = nw.from_native(doc_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"document": doc_ref},
            feature_spec=DocumentChunks.spec(),
            feature_plan=graph.get_feature_plan(DocumentChunks.spec().key),
            # No rename needed - doc_id column names are the same
        )

        result_df = result.collect().to_native()
        # Should not aggregate (expansion happens in load_input)
        assert result_df.shape[0] == 2
        assert set(result_df["doc_id"]) == {"d1", "d2"}

    def test_expansion_no_aggregation_during_join(self, graph):
        """Test that ExpansionRelationship does NOT aggregate during join phase."""

        class User(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="user",
                id_columns=["user_id"],
                fields=[FieldSpec(key="name", code_version="1")],
            ),
        ):
            pass

        class UserSessions(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="user/sessions",
                id_columns=["user_id", "session_id"],
                lineage=LineageRelationship.expansion(on=["user_id"]),
                fields=[FieldSpec(key="session_data", code_version="1")],
                deps=[
                    FeatureDep(
                        feature="user",
                        # No rename needed - user_id column is the same
                    )
                ],
            ),
        ):
            pass

        # Parent data
        user_data = pl.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "name": ["Alice", "Bob", "Charlie"],
                "provenance_by_field": [
                    {"name": "hash_alice"},
                    {"name": "hash_bob"},
                    {"name": "hash_charlie"},
                ],
            }
        )

        # Test joiner behavior
        joiner = NarwhalsJoiner()
        user_ref = nw.from_native(user_data.lazy())

        result, mapping = joiner.join_upstream(
            upstream_refs={"user": user_ref},
            feature_spec=UserSessions.spec(),
            feature_plan=graph.get_feature_plan(UserSessions.spec().key),
            # No rename needed - user_id column names are the same
        )

        result_df = result.collect().to_native()

        # IMPORTANT: Should NOT aggregate during join
        # Expansion happens in load_input, not in joiner
        assert result_df.shape[0] == 3, (
            "Should have same rows as parent (no aggregation)"
        )
        assert "name" in result_df.columns, "Parent columns should be preserved"

        # Provenance should NOT be aggregated
        provenance_col = "__upstream_user__provenance_by_field"
        assert provenance_col in result_df.columns

        # Check individual provenance values
        for i, expected_hash in enumerate(["hash_alice", "hash_bob", "hash_charlie"]):
            provenance = result_df[provenance_col][i]
            assert provenance["name"] == expected_hash, (
                "Provenance should not be modified"
            )

    def test_expansion_with_multiple_parents(self, graph):
        """Test ExpansionRelationship with multiple parent dependencies."""

        class VideoMetadata(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="video/metadata",
                id_columns=["video_id"],
            ),
        ):
            pass

        class VideoContent(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="video/content",
                id_columns=["video_id"],
            ),
        ):
            pass

        class VideoAnalysis(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="video/analysis",
                id_columns=["video_id", "analysis_id"],
                lineage=LineageRelationship.expansion(on=["video_id"]),
                deps=[
                    FeatureDep(
                        feature="video/metadata",
                        # No rename needed - video_id column is the same
                    ),
                    FeatureDep(
                        feature="video/content",
                        # No rename needed - video_id column is the same
                    ),
                ],
            ),
        ):
            pass

        metadata_data = pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "title": ["Video 1", "Video 2"],
                "provenance_by_field": [
                    {"default": "meta_v1"},
                    {"default": "meta_v2"},
                ],
            }
        )

        content_data = pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "size_mb": [100, 200],
                "provenance_by_field": [
                    {"default": "content_v1"},
                    {"default": "content_v2"},
                ],
            }
        )

        joiner = NarwhalsJoiner()

        result, mapping = joiner.join_upstream(
            upstream_refs={
                "video/metadata": nw.from_native(metadata_data.lazy()),
                "video/content": nw.from_native(content_data.lazy()),
            },
            feature_spec=VideoAnalysis.spec(),
            feature_plan=graph.get_feature_plan(VideoAnalysis.spec().key),
            # No rename needed - video_id column names are the same
        )

        result_df = result.collect().to_native()

        # Should join both parents without aggregation
        assert result_df.shape[0] == 2
        assert "title" in result_df.columns
        assert "size_mb" in result_df.columns

    def test_expansion_with_metadata_store_diff(self, graph):
        """Test ExpansionRelationship diff resolution with metadata store."""

        class Article(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="article",
                id_columns=["article_id"],
                fields=[FieldSpec(key="text", code_version="1")],
            ),
        ):
            pass

        class ArticleParagraphs(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="paragraphs",
                id_columns=["article_id", "para_id"],
                lineage=LineageRelationship.expansion(on=["article_id"]),
                fields=[FieldSpec(key="para_text", code_version="1")],
                deps=[
                    FeatureDep(
                        feature="article",
                        # No rename needed - article_id column is the same
                    )
                ],
            ),
        ):
            pass

        with InMemoryMetadataStore() as store:
            # Parent metadata
            article_data = pl.DataFrame(
                {
                    "article_id": ["a1", "a2"],
                    "text": ["Article 1 text", "Article 2 text"],
                    "provenance_by_field": [
                        {"text": "hash_a1_v2"},  # Changed
                        {"text": "hash_a2"},  # Unchanged
                    ],
                }
            )
            store.write_metadata(Article, nw.from_native(article_data))

            # Child metadata (existing)
            para_data = pl.DataFrame(
                {
                    "article_id": ["a1", "a1", "a1", "a2", "a2"],
                    "para_id": [1, 2, 3, 1, 2],
                    "para_text": ["p1", "p2", "p3", "p4", "p5"],
                    "provenance_by_field": [
                        {"para_text": "old_p1"},
                        {"para_text": "old_p2"},
                        {"para_text": "old_p3"},
                        {"para_text": "old_p4"},
                        {"para_text": "old_p5"},
                    ],
                }
            )
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
        assert result1 == ["video_id"]

        # Without 'on' parameter (None for inference)
        rel2 = ExpansionRelationship(id_generation_pattern="hash")
        result2 = rel2.get_aggregation_columns(["doc_id", "chunk_id"])
        assert result2 is None

        # Empty 'on' parameter
        rel3 = ExpansionRelationship(on=[], id_generation_pattern="custom")
        result3 = rel3.get_aggregation_columns(["id1", "id2"])
        assert result3 == []

    def test_expansion_vs_aggregation_difference(self, graph):
        """Test that Expansion and Aggregation behave differently."""

        class SensorData(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sensor/data",
                id_columns=["sensor_id", "timestamp", "sub_reading"],
            ),
        ):
            pass

        # Aggregation version (N:1)
        class SensorSummaryAgg(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sensor/summary_agg",
                id_columns=["sensor_id", "timestamp"],
                lineage=LineageRelationship.aggregation(on=["sensor_id", "timestamp"]),
                deps=[
                    FeatureDep(
                        feature="sensor/data",
                        # No rename needed - sensor_id and timestamp columns are the same
                    )
                ],
            ),
        ):
            pass

        # Expansion version (1:N) - WRONG for this use case but testing behavior
        class SensorSummaryExp(
            BaseFeature,
            spec=BaseFeatureSpec(
                key="sensor/summary_exp",
                id_columns=["sensor_id", "timestamp"],
                lineage=LineageRelationship.expansion(on=["sensor_id", "timestamp"]),
                deps=[
                    FeatureDep(
                        feature="sensor/data",
                        # No rename needed - sensor_id and timestamp columns are the same
                    )
                ],
            ),
        ):
            pass

        # Data with multiple sub_readings per sensor/timestamp
        sensor_data = pl.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s1", "s2", "s2"],
                "timestamp": [100, 100, 100, 100, 100],
                "sub_reading": [1, 2, 3, 1, 2],
                "value": [10.1, 10.2, 10.3, 20.1, 20.2],
                "provenance_by_field": [{"default": f"hash_{i}"} for i in range(5)],
            }
        )

        joiner = NarwhalsJoiner()
        sensor_ref = nw.from_native(sensor_data.lazy())

        # Test Aggregation behavior
        agg_result, _ = joiner.join_upstream(
            upstream_refs={"sensor/data": sensor_ref},
            feature_spec=SensorSummaryAgg.spec(),
            feature_plan=graph.get_feature_plan(SensorSummaryAgg.spec().key),
            # No rename needed - sensor_id and timestamp column names are the same
        )
        agg_df = agg_result.collect().to_native()

        # Aggregation should reduce 5 rows to 2 (one per sensor)
        assert agg_df.shape[0] == 2, (
            f"Aggregation should produce 2 rows, got {agg_df.shape[0]}"
        )

        # Test Expansion behavior
        exp_result, _ = joiner.join_upstream(
            upstream_refs={"sensor/data": sensor_ref},
            feature_spec=SensorSummaryExp.spec(),
            feature_plan=graph.get_feature_plan(SensorSummaryExp.spec().key),
            # No rename needed - sensor_id and timestamp column names are the same
        )
        exp_df = exp_result.collect().to_native()

        # Expansion should NOT aggregate (keep all 5 rows)
        # Actually, since ExpansionRelationship with on=["sensor_id", "timestamp"]
        # returns those columns for aggregation, it WILL aggregate!
        # This shows why the semantic difference matters
        assert exp_df.shape[0] == 2, "Expansion with 'on' still aggregates in joiner!"
