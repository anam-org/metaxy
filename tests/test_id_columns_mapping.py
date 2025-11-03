"""Tests for id_columns_mapping in FeatureDep for flexible dependency joins."""

import narwhals as nw
import polars as pl
import pytest

from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.models.feature_spec import BaseFeatureSpec, FeatureDep, IDColumns
from metaxy.models.types import FeatureKey


class TestIdColumnsMapping:
    """Test suite for id_columns_mapping functionality."""

    def test_basic_id_columns_mapping(self):
        """Test basic ID column mapping between features with different column names."""
        # Create isolated graph for testing
        test_graph = FeatureGraph()

        with test_graph.use():
            # Define parent feature with different ID column name
            class ParentFeature(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(key="test/parent", id_columns=["parent_id"]),
            ):
                pass

            # Define child feature that maps to parent's ID column
            class ChildFeature(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(
                    key="test/child",
                    id_columns=["child_parent_id"],
                    deps=[
                        FeatureDep(
                            feature="test/parent",
                            id_columns_mapping={
                                "child_parent_id": "parent_id"  # Map child's ID to parent's ID
                            },
                        )
                    ],
                ),
            ):
                pass

            # Create test data
            parent_data = pl.DataFrame(
                {
                    "parent_id": [1, 2, 3],
                    "parent_value": ["a", "b", "c"],
                    "provenance_by_field": [
                        {"default": "hash1"},
                        {"default": "hash2"},
                        {"default": "hash3"},
                    ],
                }
            )

            # Test joining with mapped columns
            joiner = NarwhalsJoiner()
            parent_ref = nw.from_native(parent_data.lazy())

            result, mapping = joiner.join_upstream(
                upstream_refs={"test/parent": parent_ref},
                feature_spec=ChildFeature.spec(),
                feature_plan=test_graph.get_feature_plan(ChildFeature.spec().key),
                upstream_id_mappings={"test/parent": {"child_parent_id": "parent_id"}},
            )

            # Verify result has correct columns
            result_df = result.collect().to_native()
            assert (
                "child_parent_id" in result_df.columns
            )  # Renamed to match child's ID column
            assert "parent_value" in result_df.columns
            assert len(result_df) == 3

    def test_one_to_many_video_chunks(self):
        """Test one-to-many relationship: video -> chunks with dynamic chunk generation."""
        # Create isolated graph for testing
        test_graph = FeatureGraph()

        with test_graph.use():
            # Define video feature (parent)
            class Video(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(key="video", id_columns=["video_id"]),
            ):
                pass

            # Define video chunks feature (child)
            class VideoChunks(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(
                    key="video/chunks",
                    id_columns=["video_id", "chunk_id"],  # chunk_id is child-specific
                    deps=[
                        FeatureDep(
                            feature="video",
                            id_columns_mapping={
                                "video_id": "video_id",  # Map video_id to parent's video_id
                                # chunk_id is not mapped - it's generated dynamically
                            },
                        )
                    ],
                ),
            ):
                pass

            # Create test data - parent video
            video_data = pl.DataFrame(
                {
                    "video_id": ["vid1", "vid2"],
                    "video_file": ["file1.mp4", "file2.mp4"],
                    "provenance_by_field": [
                        {"default": "hash_vid1"},
                        {"default": "hash_vid2"},
                    ],
                }
            )

            # Create test data - child chunks (multiple per video)
            pl.DataFrame(
                {
                    "video_id": ["vid1", "vid1", "vid1", "vid2", "vid2"],
                    "chunk_id": ["chunk_0", "chunk_1", "chunk_2", "chunk_0", "chunk_1"],
                    "chunk_data": ["c1_0", "c1_1", "c1_2", "c2_0", "c2_1"],
                }
            )

            # Test joining
            joiner = NarwhalsJoiner()
            video_ref = nw.from_native(video_data.lazy())

            result, mapping = joiner.join_upstream(
                upstream_refs={"video": video_ref},
                feature_spec=VideoChunks.spec(),
                feature_plan=test_graph.get_feature_plan(VideoChunks.spec().key),
                upstream_id_mappings={"video": {"video_id": "video_id"}},
            )

            # Verify result
            result_df = result.collect().to_native()
            assert "video_id" in result_df.columns
            assert "video_file" in result_df.columns
            # Note: chunk_id is not in upstream, so not in joined result
            # It would be added during transform

    def test_validation_invalid_target_columns(self):
        """Test validation error when mapping references invalid target columns."""
        test_graph = FeatureGraph()

        with test_graph.use():

            class ParentFeature(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(key="test/parent", id_columns=["parent_id"]),
            ):
                pass

            # This should raise an error - invalid_id is not in target's id_columns
            with pytest.raises(
                ValueError, match="Invalid target ID columns in id_columns_mapping"
            ):

                class InvalidChild(
                    BaseFeature[IDColumns],
                    spec=BaseFeatureSpec(
                        key="test/invalid_child",
                        id_columns=["child_id"],
                        deps=[
                            FeatureDep(
                                feature="test/parent",
                                id_columns_mapping={
                                    "invalid_id": "parent_id"  # invalid_id is not in id_columns
                                },
                            )
                        ],
                    ),
                ):
                    pass

    def test_validation_missing_upstream_columns(self):
        """Test validation error when mapping references non-existent upstream columns."""
        test_graph = FeatureGraph()

        with test_graph.use():

            class ParentFeature(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(key="test/parent", id_columns=["parent_id"]),
            ):
                pass

            class ChildFeature(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(
                    key="test/child",
                    id_columns=["child_id"],
                    deps=[
                        FeatureDep(
                            feature="test/parent",
                            id_columns_mapping={
                                "child_id": "nonexistent_id"  # This column doesn't exist in parent
                            },
                        )
                    ],
                ),
            ):
                pass

            # Create test data without the mapped column
            parent_data = pl.DataFrame(
                {
                    "parent_id": [1, 2, 3],
                    "parent_value": ["a", "b", "c"],
                    "provenance_by_field": [
                        {"default": "hash1"},
                        {"default": "hash2"},
                        {"default": "hash3"},
                    ],
                }
            )

            # This should raise an error during join
            joiner = NarwhalsJoiner()
            parent_ref = nw.from_native(parent_data.lazy())

            with pytest.raises(ValueError, match="is missing mapped ID columns"):
                result, mapping = joiner.join_upstream(
                    upstream_refs={"test/parent": parent_ref},
                    feature_spec=ChildFeature.spec(),
                    feature_plan=test_graph.get_feature_plan(ChildFeature.spec().key),
                    upstream_id_mappings={
                        "test/parent": {"child_id": "nonexistent_id"}
                    },
                )

    def test_multiple_upstreams_mixed_mappings(self):
        """Test multiple upstream dependencies with mix of mapped and unmapped."""
        test_graph = FeatureGraph()

        with test_graph.use():
            # First upstream with standard ID column
            class UpstreamA(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(key="upstream/a", id_columns=["sample_uid"]),
            ):
                pass

            # Second upstream with different ID column
            class UpstreamB(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(key="upstream/b", id_columns=["item_id"]),
            ):
                pass

            # Target feature with mixed dependencies
            class TargetFeature(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(
                    key="target",
                    id_columns=["sample_uid", "sequence_id"],
                    deps=[
                        FeatureDep(
                            feature="upstream/a"
                            # No id_columns_mapping - expects all ID columns to match
                        ),
                        FeatureDep(
                            feature="upstream/b",
                            id_columns_mapping={
                                "sample_uid": "item_id"  # Map target's sample_uid to upstream's item_id
                            },
                        ),
                    ],
                ),
            ):
                pass

            # Verify the feature was created successfully
            assert TargetFeature.spec().key == FeatureKey(["target"])
            deps = TargetFeature.spec().deps
            assert deps is not None
            assert len(deps) == 2

    def test_column_renaming_with_id_mapping(self):
        """Test that column renaming works correctly with id_columns_mapping."""
        test_graph = FeatureGraph()

        with test_graph.use():

            class ParentFeature(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(key="test/parent", id_columns=["parent_id"]),
            ):
                pass

            class ChildFeature(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(
                    key="test/child",
                    id_columns=["child_id"],
                    deps=[
                        FeatureDep(
                            feature="test/parent",
                            columns=("parent_value",),  # Select specific column
                            rename={"parent_value": "renamed_value"},  # Rename it
                            id_columns_mapping={
                                "child_id": "parent_id"  # Map IDs
                            },
                        )
                    ],
                ),
            ):
                pass

            # Create test data
            parent_data = pl.DataFrame(
                {
                    "parent_id": [1, 2, 3],
                    "parent_value": ["a", "b", "c"],
                    "other_value": ["x", "y", "z"],
                    "provenance_by_field": [
                        {"default": "hash1"},
                        {"default": "hash2"},
                        {"default": "hash3"},
                    ],
                }
            )

            # Test joining with both renaming and mapping
            joiner = NarwhalsJoiner()
            parent_ref = nw.from_native(parent_data.lazy())

            result, mapping = joiner.join_upstream(
                upstream_refs={"test/parent": parent_ref},
                feature_spec=ChildFeature.spec(),
                feature_plan=test_graph.get_feature_plan(ChildFeature.spec().key),
                upstream_columns={"test/parent": ("parent_value",)},
                upstream_renames={"test/parent": {"parent_value": "renamed_value"}},
                upstream_id_mappings={"test/parent": {"child_id": "parent_id"}},
            )

            # Verify result
            result_df = result.collect().to_native()
            assert "child_id" in result_df.columns  # Renamed from parent_id
            assert "renamed_value" in result_df.columns  # Renamed from parent_value
            assert "other_value" not in result_df.columns  # Not selected

    def test_empty_id_mapping_dict(self):
        """Test that empty id_columns_mapping dict behaves like None."""
        test_graph = FeatureGraph()

        with test_graph.use():

            class ParentFeature(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(key="test/parent", id_columns=["sample_uid"]),
            ):
                pass

            # Empty mapping should require all ID columns to match
            class ChildFeature(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(
                    key="test/child",
                    id_columns=["sample_uid"],
                    deps=[
                        FeatureDep(
                            feature="test/parent",
                            id_columns_mapping={},  # Empty mapping
                        )
                    ],
                ),
            ):
                pass

            # Create test data
            parent_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "value": ["a", "b", "c"],
                    "provenance_by_field": [
                        {"default": "hash1"},
                        {"default": "hash2"},
                        {"default": "hash3"},
                    ],
                }
            )

            # Test joining - should work since all ID columns match
            joiner = NarwhalsJoiner()
            parent_ref = nw.from_native(parent_data.lazy())

            result, mapping = joiner.join_upstream(
                upstream_refs={"test/parent": parent_ref},
                feature_spec=ChildFeature.spec(),
                feature_plan=test_graph.get_feature_plan(ChildFeature.spec().key),
                upstream_id_mappings={"test/parent": {}},  # Empty mapping
            )

            # Verify result
            result_df = result.collect().to_native()
            assert "sample_uid" in result_df.columns
            assert "value" in result_df.columns

    def test_integration_with_metadata_store(self):
        """Test id_columns_mapping integration with metadata store operations."""
        test_graph = FeatureGraph()

        with test_graph.use():
            # Define video feature
            class Video(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(key="video", id_columns=["video_id"]),
            ):
                pass

            # Define video chunks feature with mapping
            class VideoChunks(
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(
                    key="video/chunks",
                    id_columns=["video_id", "chunk_id"],
                    deps=[
                        FeatureDep(
                            feature="video", id_columns_mapping={"video_id": "video_id"}
                        )
                    ],
                ),
            ):
                pass

            # Test with metadata store
            with InMemoryMetadataStore() as store:
                # Write parent metadata
                video_metadata = pl.DataFrame(
                    {
                        "video_id": ["vid1", "vid2"],
                        "video_file": ["file1.mp4", "file2.mp4"],
                        "feature_version": ["v1", "v1"],
                        "provenance_by_field": [
                            {"default": "hash_vid1"},
                            {"default": "hash_vid2"},
                        ],
                    }
                )
                store.write_metadata(Video, nw.from_native(video_metadata))

                # Verify we can read it back
                read_back = store.read_metadata(Video)
                assert read_back.collect().shape[0] == 2
