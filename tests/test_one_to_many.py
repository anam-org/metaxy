"""Tests for one-to-many relationship handling."""

import narwhals as nw
import polars as pl

from metaxy import (
    Feature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
)
from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner
from metaxy.utils.one_to_many import (
    expand_to_children,
    generate_child_sample_uid,
)


class TestOneToManyUtilities:
    """Test utility functions for one-to-many relationships."""

    def test_generate_child_sample_uid_deterministic(self):
        """Test that child UIDs are deterministic."""
        parent_uid = 12345
        child_index = 0
        namespace = "test"

        # Generate multiple times - should be same
        uid1 = generate_child_sample_uid(parent_uid, child_index, namespace=namespace)
        uid2 = generate_child_sample_uid(parent_uid, child_index, namespace=namespace)

        assert uid1 == uid2
        assert uid1 > 0  # Should be positive

    def test_generate_child_sample_uid_unique(self):
        """Test that different inputs produce different UIDs."""
        base_parent = 12345

        # Different indices should produce different UIDs
        uid1 = generate_child_sample_uid(base_parent, 0, namespace="test")
        uid2 = generate_child_sample_uid(base_parent, 1, namespace="test")
        assert uid1 != uid2

        # Different parents should produce different UIDs
        uid3 = generate_child_sample_uid(54321, 0, namespace="test")
        assert uid1 != uid3

        # Different namespaces should produce different UIDs
        uid4 = generate_child_sample_uid(base_parent, 0, namespace="other")
        assert uid1 != uid4

    def test_expand_to_children_fixed_count(self):
        """Test expansion with fixed number of children per parent."""
        # Create parent data
        parent_data = pl.DataFrame({
            "sample_uid": [100, 200, 300],
            "data_version": [
                {"default": "v1"},
                {"default": "v2"},
                {"default": "v3"},
            ],
            "custom_field": ["a", "b", "c"],
        })

        parent_lazy = nw.from_native(parent_data.lazy(), eager_only=False)

        # Expand to 3 children per parent
        expanded = expand_to_children(
            parent_lazy,
            num_children_per_parent=3,
            parent_ref_column="parent_id",
            namespace="test_chunk",
        )

        result = expanded.collect().to_native()

        # Should have 3 parents × 3 children = 9 rows
        assert len(result) == 9

        # Check that each parent produced 3 children
        for parent_uid in [100, 200, 300]:
            parent_children = result.filter(pl.col("parent_id") == parent_uid)
            assert len(parent_children) == 3
            # Check child indices
            assert sorted(parent_children["child_index"].to_list()) == [0, 1, 2]
            # Check parent data is preserved
            parent_row = parent_data.filter(pl.col("sample_uid") == parent_uid)
            assert all(parent_children["custom_field"] == parent_row["custom_field"][0])

    def test_expand_to_children_variable_count(self):
        """Test expansion with variable number of children per parent."""
        parent_data = pl.DataFrame({
            "sample_uid": [100, 200, 300],
            "data_version": [
                {"default": "v1"},
                {"default": "v2"},
                {"default": "v3"},
            ],
            "duration": [50, 120, 30],  # Different durations
        })

        parent_lazy = nw.from_native(parent_data.lazy(), eager_only=False)

        # Variable children based on duration (1 child per 10 seconds)
        children_counts = {
            100: 5,   # 50 seconds → 5 chunks
            200: 12,  # 120 seconds → 12 chunks
            300: 3,   # 30 seconds → 3 chunks
        }

        expanded = expand_to_children(
            parent_lazy,
            num_children_per_parent=children_counts,
            parent_ref_column="parent_video_id",
            namespace="chunk",
        )

        result = expanded.collect().to_native()

        # Total should be 5 + 12 + 3 = 20
        assert len(result) == 20

        # Verify each parent has correct number of children
        for parent_uid, expected_count in children_counts.items():
            parent_children = result.filter(pl.col("parent_video_id") == parent_uid)
            assert len(parent_children) == expected_count

    def test_expand_to_children_empty(self):
        """Test expansion with empty parent DataFrame."""
        empty_data = pl.DataFrame({
            "sample_uid": [],
            "data_version": [],
        }).with_columns([
            pl.col("sample_uid").cast(pl.Int64),
        ])

        empty_lazy = nw.from_native(empty_data.lazy(), eager_only=False)

        expanded = expand_to_children(
            empty_lazy,
            num_children_per_parent=5,
            parent_ref_column="parent_id",
        )

        result = expanded.collect().to_native()

        # Should still be empty
        assert len(result) == 0
        # But should have the expected columns
        assert "sample_uid" in result.columns
        assert "child_index" in result.columns
        assert "parent_id" in result.columns


class TestOneToManyFeatures:
    """Test one-to-many relationships in features."""

    def test_video_chunk_expansion(self):
        """Test basic video to chunk expansion."""
        test_graph = FeatureGraph()

        with test_graph.use():
            # Define features
            class Video(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "video"]),
                    deps=None,
                    fields=[
                        FieldSpec(key=FieldKey(["frames"]), code_version=1),
                    ],
                ),
            ):
                pass

            class VideoChunk(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "video", "chunk"]),
                    deps=[FeatureDep(key=FeatureKey(["test", "video"]))],
                    fields=[
                        FieldSpec(
                            key=FieldKey(["frames"]),
                            code_version=1,
                            deps=[
                                FieldDep(
                                    feature_key=FeatureKey(["test", "video"]),
                                    fields=[FieldKey(["frames"])],
                                ),
                            ],
                        ),
                    ],
                ),
            ):
                @classmethod
                def load_input(cls, joiner, upstream_refs):
                    """Custom load_input for one-to-many expansion."""
                    video_key = "test/video"  # Video.key.to_string()
                    video_ref = upstream_refs[video_key]

                    # Expand each video to 5 chunks
                    expanded = expand_to_children(
                        video_ref,
                        num_children_per_parent=5,
                        parent_ref_column="parent_video_id",
                        namespace="chunk",
                    )

                    # Update refs
                    expanded_refs = dict(upstream_refs)
                    expanded_refs[video_key] = expanded

                    # Use standard joiner with expanded data
                    return joiner.join_upstream(
                        expanded_refs,
                        cls.spec,
                        cls.graph.get_feature_plan(cls.spec.key),
                    )

            # Create test data
            video_data = pl.DataFrame({
                "sample_uid": [1000, 2000],
                "data_version": [
                    {"frames": "hash1"},
                    {"frames": "hash2"},
                ],
                "video_path": ["/video1.mp4", "/video2.mp4"],
            })

            # Test expansion
            joiner = NarwhalsJoiner()
            upstream_refs = {
                "test/video": nw.from_native(video_data.lazy(), eager_only=False)
            }

            joined, mapping = VideoChunk.load_input(joiner, upstream_refs)
            result = joined.collect().to_native()

            # Should have 2 videos × 5 chunks = 10 rows
            assert len(result) == 10

            # Check parent references are maintained
            assert "parent_video_id" in result.columns
            video1_chunks = result.filter(pl.col("parent_video_id") == 1000)
            assert len(video1_chunks) == 5
            assert all(video1_chunks["video_path"] == "/video1.mp4")

            # Check chunk indices
            assert sorted(video1_chunks["child_index"].to_list()) == [0, 1, 2, 3, 4]

            # Check data_version column is renamed correctly
            assert "__upstream_test/video__data_version" in result.columns

    def test_multi_dependency_expansion(self):
        """Test expansion with multiple dependencies."""
        test_graph = FeatureGraph()

        with test_graph.use():
            class Video(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "video"]),
                    deps=None,
                ),
            ):
                pass

            class ChunkConfig(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "chunk", "config"]),
                    deps=[FeatureDep(key=FeatureKey(["test", "video"]))],
                ),
            ):
                pass

            class VideoChunk(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "video", "chunk"]),
                    deps=[
                        FeatureDep(key=FeatureKey(["test", "video"])),
                        FeatureDep(key=FeatureKey(["test", "chunk", "config"])),
                    ],
                ),
            ):
                @classmethod
                def load_input(cls, joiner, upstream_refs):
                    """Handle multiple dependencies with expansion."""
                    video_key = "test/video"  # Video.key.to_string()
                    config_key = "test/chunk/config"  # ChunkConfig.key.to_string()

                    video_ref = upstream_refs[video_key]
                    config_ref = upstream_refs[config_key]

                    # First join video and config
                    joined = video_ref.join(
                        config_ref,
                        on="sample_uid",
                        how="inner",
                    )

                    # Then expand based on config
                    expanded = expand_to_children(
                        joined,
                        num_children_per_parent=3,  # Simplified for test
                        parent_ref_column="parent_video_id",
                        namespace="configured_chunk",
                    )

                    # Create refs for standard joiner
                    # Since we've already joined, just return the expanded data
                    # with proper column naming
                    expanded_with_renames = expanded.with_columns([
                        nw.col("data_version").alias(f"__upstream_{video_key}__data_version"),
                    ])

                    column_mapping = {
                        video_key: f"__upstream_{video_key}__data_version",
                        config_key: f"__upstream_{config_key}__data_version",
                    }

                    return expanded_with_renames, column_mapping

            # Create test data
            video_data = pl.DataFrame({
                "sample_uid": [1000],
                "data_version": [{"default": "video_hash"}],
                "path": ["/video.mp4"],
            })

            config_data = pl.DataFrame({
                "sample_uid": [1000],
                "data_version": [{"default": "config_hash"}],
                "chunk_size": [10],
            })

            # Test multi-dependency expansion
            joiner = NarwhalsJoiner()
            upstream_refs = {
                "test/video": nw.from_native(video_data.lazy(), eager_only=False),
                "test/chunk/config": nw.from_native(config_data.lazy(), eager_only=False),
            }

            joined, mapping = VideoChunk.load_input(joiner, upstream_refs)
            result = joined.collect().to_native()

            # Should have 3 chunks
            assert len(result) == 3

            # Should have data from both upstreams
            assert all(result["path"] == "/video.mp4")
            assert all(result["chunk_size"] == 10)

            # Check parent reference
            assert all(result["parent_video_id"] == 1000)

    def test_aggregation_many_to_one(self):
        """Test aggregating chunks back to video level."""
        test_graph = FeatureGraph()

        with test_graph.use():
            class VideoChunk(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "chunk"]),
                    deps=None,
                    id_columns=["chunk_id"],
                ),
            ):
                pass

            class VideoSummary(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["test", "summary"]),
                    deps=[FeatureDep(key=FeatureKey(["test", "chunk"]))],
                    id_columns=["video_id"],
                ),
            ):
                @classmethod
                def load_input(cls, joiner, upstream_refs):
                    """Aggregate chunks back to video level."""
                    chunk_key = "test/chunk"  # VideoChunk.key.to_string()
                    chunks_ref = upstream_refs[chunk_key]

                    # Aggregate by parent video
                    aggregated = (
                        chunks_ref
                        .group_by("parent_video_id")
                        .agg([
                            nw.col("score").mean().alias("avg_score"),
                            nw.col("score").max().alias("max_score"),
                            nw.col("chunk_id").count().alias("num_chunks"),
                        ])
                        .rename({"parent_video_id": "video_id"})
                    )

                    # Add data_version column
                    aggregated = aggregated.with_columns([
                        nw.lit({"default": "aggregated"}).alias(
                            f"__upstream_{chunk_key}__data_version"
                        )
                    ])

                    column_mapping = {
                        chunk_key: f"__upstream_{chunk_key}__data_version",
                    }

                    return aggregated, column_mapping

            # Create chunk data
            chunk_data = pl.DataFrame({
                "chunk_id": [1, 2, 3, 4, 5, 6],
                "parent_video_id": [100, 100, 100, 200, 200, 200],
                "score": [0.5, 0.7, 0.9, 0.3, 0.6, 0.8],
                "data_version": [{"default": f"chunk_{i}"} for i in range(6)],
            })

            # Test aggregation
            joiner = NarwhalsJoiner()
            upstream_refs = {
                "test/chunk": nw.from_native(
                    chunk_data.lazy(), eager_only=False
                )
            }

            aggregated, mapping = VideoSummary.load_input(joiner, upstream_refs)
            result = aggregated.collect().to_native()

            # Should have 2 videos
            assert len(result) == 2

            # Check aggregations
            video1 = result.filter(pl.col("video_id") == 100)
            assert len(video1) == 1
            assert video1["num_chunks"][0] == 3
            assert abs(video1["avg_score"][0] - 0.7) < 0.01  # (0.5+0.7+0.9)/3
            assert video1["max_score"][0] == 0.9

            video2 = result.filter(pl.col("video_id") == 200)
            assert len(video2) == 1
            assert video2["num_chunks"][0] == 3
            assert abs(video2["avg_score"][0] - 0.567) < 0.01  # (0.3+0.6+0.8)/3
            assert video2["max_score"][0] == 0.8