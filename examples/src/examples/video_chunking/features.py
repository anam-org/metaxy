"""Video processing features demonstrating one-to-many relationships.

This module demonstrates how to handle one-to-many relationships in Metaxy,
where a single video produces multiple chunks, and each chunk can be processed
independently by downstream features.

Key concepts demonstrated:
1. Parent-child relationships via custom load_input()
2. Field-level dependencies (e.g., DetectedFaces depends on VideoChunk.frames)
3. Sample UID management in fan-out patterns
4. Proper type hints with Narwhals DataFrames
"""

from typing import TYPE_CHECKING, Any

from pydantic import Field

from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
)

if TYPE_CHECKING:
    import narwhals as nw

    from metaxy.data_versioning.joiners import UpstreamJoiner


class Video(
    Feature,
    spec=FeatureSpec(
        id_columns=["video_id"],
        key=FeatureKey(["video", "raw"]),
        deps=None,  # Root feature - no dependencies
        fields=[
            FieldSpec(
                key=FieldKey(["frames"]),
                code_version=1,
                deps=None,
            ),
            FieldSpec(
                key=FieldKey(["audio"]),
                code_version=1,
                deps=None,
            ),
        ],
    ),
):
    """Root feature representing raw video data.

    This is the source feature that contains:
    - frames: Raw video frames data
    - audio: Raw audio track data

    Each video has a unique sample_uid identifying it.
    """

    path: str = Field(description="Path to the video file")
    duration: float = Field(description="Duration of the video in seconds")


class VideoChunkingComputation(
    Feature,
    spec=FeatureSpec(
        id_columns=["video_id"],
        key=FeatureKey(["video", "chunking", "state"]),
        deps=[
            FeatureDep(key=Video.key),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["frames"]),
                code_version=1,
                deps=FieldDep(
                    feature_key=FeatureKey(["video", "raw"]),
                    fields=[FieldKey(["frames"])],
                ),
            ),
            FieldSpec(
                key=FieldKey(["audio"]),
                code_version=1,
                deps=FieldDep(
                    feature_key=FeatureKey(["video", "raw"]),
                    fields=[FieldKey(["audio"])],  # Fixed: audio should depend on audio, not frames
                ),
            ),
        ],
    ),
):
    """A **logical feature** that represents computation.

    It doesn't hold any data itself. It will be materialized together with VideoChunk.
    """

    pass


class VideoChunk(
    Feature,
    spec=FeatureSpec(
        id_columns=["chunk_id", "parent_video_id"],
        key=FeatureKey(["video", "chunk"]),
        deps=[
            FeatureDep(key=Video.key),
            FeatureDep(key=VideoChunkingComputation.key),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["frames"]),
                code_version=1,
                deps=FieldDep(
                    feature_key=FeatureKey(["video", "raw"]),
                    fields=[FieldKey(["frames"])],
                ),
            ),
            FieldSpec(
                key=FieldKey(["audio"]),
                code_version=1,
                deps=FieldDep(
                    feature_key=FeatureKey(["video", "raw"]),
                    fields=[FieldKey(["audio"])],  # Fixed: audio should depend on audio, not frames
                ),
            ),
        ],
    ),
):
    """Individual video chunk extracted from the parent video.

    This feature demonstrates ONE-TO-MANY relationships:
    - One video produces multiple chunks
    - Each chunk has its own unique identifier: chunk_id
    - Chunks maintain reference to parent video via parent_video_id

    Fields:
    - frames: Frames for this specific chunk
    - audio: Audio for this specific chunk

    The custom load_input() method below handles the fan-out logic.
    """

    # Configuration for video chunking
    chunk_size_seconds: float = Field(default=10.0, description="Size of each chunk in seconds")
    chunk_overlap_seconds: float = Field(default=1.0, description="Overlap between chunks in seconds")

    @classmethod
    def load_input(
        cls,
        joiner: "UpstreamJoiner",
        upstream_refs: dict[str, "nw.LazyFrame[Any]"],
    ) -> tuple["nw.LazyFrame[Any]", dict[str, str]]:
        """Custom load_input for one-to-many relationship.

        This method performs the fan-out from Video to VideoChunk, where one video
        produces multiple chunks. Each chunk gets a unique sample_uid but maintains
        a reference to its parent video.

        Args:
            joiner: UpstreamJoiner from MetadataStore
            upstream_refs: References to upstream feature metadata

        Returns:
            Tuple of (expanded DataFrame with chunks, column mapping)
        """
        from metaxy.utils.one_to_many import expand_to_children

        # Since we have two dependencies (Video and VideoChunkingComputation),
        # but only want to expand Video, we need to handle this carefully.
        # VideoChunkingComputation is a logical feature that doesn't add data.

        # First, let's get the Video reference (the one to expand)
        video_key = "video/raw"
        chunking_key = "video/chunking/state"

        if video_key not in upstream_refs:
            raise ValueError(f"Expected upstream feature '{video_key}' not found")

        video_ref = upstream_refs[video_key]

        # For demonstration, we'll expand each video to a fixed number of chunks
        # In a real scenario, you might calculate this based on video duration
        # For now, let's assume 10 chunks per video
        NUM_CHUNKS_PER_VIDEO = 10

        # Expand the video reference to create chunks
        expanded_video = expand_to_children(
            video_ref,
            num_children_per_parent=NUM_CHUNKS_PER_VIDEO,
            parent_id_column="sample_uid",  # Video's sample_uid
            child_id_column="sample_uid",    # Chunk's sample_uid (replaces parent)
            child_index_column="chunk_id",   # Add chunk_id column
            parent_ref_column="parent_video_id",  # Keep reference to parent video
            namespace="video_chunk",  # Namespace to ensure unique chunk UIDs
        )

        # Update the upstream_refs with expanded video
        expanded_refs = dict(upstream_refs)
        expanded_refs[video_key] = expanded_video

        # Now we need to handle the VideoChunkingComputation dependency
        # Since it's per-video, we need to join it with the expanded chunks
        # based on the parent_video_id matching the original sample_uid
        if chunking_key in upstream_refs:
            # The chunking state needs to be joined differently:
            # - Video chunks join on parent_video_id = chunking's sample_uid
            # This requires custom logic since standard joiner expects same ID columns

            # For now, we'll use the standard joiner but note that in production,
            # you'd want to handle this join specially based on parent_video_id
            pass

        # Use the standard joiner with our expanded references
        # This will join all upstreams on sample_uid (the chunk's sample_uid)
        joined, mapping = joiner.join_upstream(
            expanded_refs,
            cls.spec,
            cls.graph.get_feature_plan(cls.spec.key),
        )

        return joined, mapping


class DetectedFaces(
    Feature,
    spec=FeatureSpec(
        id_columns=["chunk_id"],
        key=FeatureKey(["video", "faces"]),
        deps=[
            # Only depend on VideoChunk, not the raw video
            FeatureDep(key=FeatureKey(["video", "chunk"])),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["faces"]),
                code_version=1,
                deps=[
                    # Field-level dependency on VideoChunk.frames only
                    FieldDep(
                        feature_key=FeatureKey(["video", "chunk"]),
                        fields=[
                            FieldKey(["frames"])
                        ],  # Only depends on frames, not audio
                    ),
                ],
            ),
        ],
    ),
):
    """Detected faces in video chunks.

    This feature demonstrates:
    - Processing data from a one-to-many relationship
    - Field-level dependencies (only depends on frames, not audio)
    - Maintaining the same composite key structure as parent

    Each chunk can have different numbers of detected faces, showing
    how downstream features handle the fan-out from VideoChunk.
    """

    num_faces: int = Field(description="Number of faces detected in the chunk")
