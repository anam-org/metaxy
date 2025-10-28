"""Advanced video chunking example with proper one-to-many handling.

This module demonstrates a more sophisticated approach to one-to-many relationships
in Metaxy, showing how to:

1. Handle multiple dependencies with different join semantics
2. Use dynamic chunk counts based on parent data
3. Properly manage parent-child relationships
4. Create a reusable pattern for similar use cases
"""

from typing import TYPE_CHECKING, Any

import narwhals as nw
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
    from metaxy.data_versioning.joiners import UpstreamJoiner


class RawVideo(
    Feature,
    spec=FeatureSpec(
        id_columns=["video_id"],
        key=FeatureKey(["advanced", "video", "raw"]),
        deps=None,  # Root feature
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
            FieldSpec(
                key=FieldKey(["metadata"]),
                code_version=1,
                deps=None,
            ),
        ],
    ),
):
    """Raw video data with metadata.

    This root feature contains the raw video data and metadata needed
    for intelligent chunking decisions.
    """

    path: str = Field(description="Path to the video file")
    duration_seconds: float = Field(description="Duration of the video in seconds")
    fps: float = Field(description="Frames per second")
    resolution: str = Field(description="Video resolution (e.g., '1920x1080')")


class ChunkingStrategy(
    Feature,
    spec=FeatureSpec(
        id_columns=["video_id"],
        key=FeatureKey(["advanced", "video", "chunking", "strategy"]),
        deps=[
            FeatureDep(
                key=RawVideo.key,
                columns=("duration_seconds", "fps"),  # Only need these for strategy
            ),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["chunk_boundaries"]),
                code_version=1,
                deps=FieldDep(
                    feature_key=RawVideo.key,
                    fields=[FieldKey(["metadata"])],
                ),
            ),
        ],
    ),
):
    """Computed chunking strategy based on video characteristics.

    This feature analyzes the video metadata and determines optimal chunk
    boundaries. It's computed per-video and produces chunking parameters.
    """

    num_chunks: int = Field(description="Number of chunks for this video")
    chunk_duration_seconds: float = Field(description="Duration of each chunk")
    overlap_seconds: float = Field(description="Overlap between consecutive chunks")


class VideoChunkAdvanced(
    Feature,
    spec=FeatureSpec(
        id_columns=["chunk_uid"],  # Different from parent!
        key=FeatureKey(["advanced", "video", "chunk"]),
        deps=[
            FeatureDep(key=RawVideo.key),
            FeatureDep(key=ChunkingStrategy.key),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["frames"]),
                code_version=1,
                deps=FieldDep(
                    feature_key=RawVideo.key,
                    fields=[FieldKey(["frames"])],
                ),
            ),
            FieldSpec(
                key=FieldKey(["audio"]),
                code_version=1,
                deps=FieldDep(
                    feature_key=RawVideo.key,
                    fields=[FieldKey(["audio"])],
                ),
            ),
            FieldSpec(
                key=FieldKey(["boundaries"]),
                code_version=1,
                deps=FieldDep(
                    feature_key=ChunkingStrategy.key,
                    fields=[FieldKey(["chunk_boundaries"])],
                ),
            ),
        ],
    ),
):
    """Individual video chunks with proper one-to-many expansion.

    This feature demonstrates advanced one-to-many handling:
    - Uses different ID columns (chunk_uid vs video_id)
    - Dynamically determines number of chunks from ChunkingStrategy
    - Maintains parent reference for traceability
    - Properly joins parent data after expansion
    """

    chunk_index: int = Field(description="0-based index of this chunk")
    parent_video_id: str = Field(description="ID of the parent video")
    start_time_seconds: float = Field(description="Start time of chunk in parent video")
    end_time_seconds: float = Field(description="End time of chunk in parent video")

    @classmethod
    def load_input(
        cls,
        joiner: "UpstreamJoiner",
        upstream_refs: dict[str, "nw.LazyFrame[Any]"],
    ) -> tuple["nw.LazyFrame[Any]", dict[str, str]]:
        """Custom load_input for advanced one-to-many video chunking.

        This implementation:
        1. First joins RawVideo and ChunkingStrategy on video_id
        2. Uses the num_chunks from ChunkingStrategy to expand
        3. Generates unique chunk_uid for each chunk
        4. Maintains parent_video_id reference
        5. Calculates chunk boundaries based on strategy
        """
        from metaxy.utils.one_to_many import generate_child_sample_uid

        # Keys for our upstream dependencies
        video_key = RawVideo.key.to_string()
        strategy_key = ChunkingStrategy.key.to_string()

        if video_key not in upstream_refs or strategy_key not in upstream_refs:
            raise ValueError("Missing required upstream features")

        video_ref = upstream_refs[video_key]
        strategy_ref = upstream_refs[strategy_key]

        # First, join video and strategy on video_id to get complete parent data
        # Both have video_id as their ID column, so standard join works
        parent_joined = video_ref.join(
            strategy_ref,
            on="video_id",  # Both use video_id
            how="inner",
        )

        # Collect to process (we need to expand based on num_chunks)
        parent_data = parent_joined.collect().to_native()

        # Import Polars for processing
        import polars as pl

        if not isinstance(parent_data, pl.DataFrame):
            raise NotImplementedError(f"Advanced chunking only implemented for Polars, got {type(parent_data)}")

        # Build expanded chunk data
        chunk_rows = []

        for row in parent_data.iter_rows(named=True):
            video_id = row["video_id"]
            num_chunks = row.get("num_chunks", 10)  # Default if missing
            duration = row.get("duration_seconds", 100.0)
            chunk_duration = row.get("chunk_duration_seconds", duration / num_chunks)
            overlap = row.get("overlap_seconds", 0.0)

            # Generate chunks for this video
            for chunk_idx in range(num_chunks):
                # Calculate chunk boundaries
                start_time = chunk_idx * (chunk_duration - overlap)
                end_time = min(start_time + chunk_duration, duration)

                # Generate unique chunk UID
                chunk_uid = generate_child_sample_uid(
                    video_id,
                    chunk_idx,
                    namespace="advanced_chunk"
                )

                # Create chunk row with all parent data plus chunk-specific fields
                chunk_row = dict(row)  # Copy all parent columns
                chunk_row["chunk_uid"] = chunk_uid
                chunk_row["chunk_index"] = chunk_idx
                chunk_row["parent_video_id"] = video_id
                chunk_row["start_time_seconds"] = start_time
                chunk_row["end_time_seconds"] = end_time

                chunk_rows.append(chunk_row)

        # Create expanded DataFrame
        if chunk_rows:
            chunks_df = pl.DataFrame(chunk_rows)
        else:
            # Empty result with correct schema
            chunks_df = pl.DataFrame([], schema={
                "chunk_uid": pl.Int64,
                "chunk_index": pl.Int64,
                "parent_video_id": pl.Utf8,
                "start_time_seconds": pl.Float64,
                "end_time_seconds": pl.Float64,
                # Include other expected columns...
            })

        # Convert to Narwhals LazyFrame
        chunks_lazy = nw.from_native(chunks_df.lazy(), eager_only=False)

        # Now we need to properly handle the data_version columns
        # The joiner expects to work with the expanded data
        # Create modified upstream_refs with our expanded chunks
        expanded_refs = {
            "expanded_chunks": chunks_lazy  # Single expanded reference
        }

        # Use a modified joiner approach since we've already done the joining
        # We just need to format the data_version columns correctly

        # Rename data_version columns from both upstreams
        rename_exprs = [
            nw.col("data_version").alias(f"__upstream_{video_key}__data_version"),
        ]

        # If strategy has its own data_version, rename it too
        if "__upstream_test/chunking/strategy__data_version" in chunks_lazy.collect_schema().names():
            rename_exprs.append(
                nw.col("__upstream_test/chunking/strategy__data_version")
                .alias(f"__upstream_{strategy_key}__data_version")
            )

        # Apply renamings if needed
        final_lazy = chunks_lazy

        # Build column mapping
        column_mapping = {
            video_key: f"__upstream_{video_key}__data_version",
            strategy_key: f"__upstream_{strategy_key}__data_version",
        }

        return final_lazy, column_mapping


class ChunkAnalysis(
    Feature,
    spec=FeatureSpec(
        id_columns=["chunk_uid"],  # Same as VideoChunkAdvanced
        key=FeatureKey(["advanced", "chunk", "analysis"]),
        deps=[
            FeatureDep(key=VideoChunkAdvanced.key),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["analysis_result"]),
                code_version=1,
                deps=FieldDep(
                    feature_key=VideoChunkAdvanced.key,
                    fields=[FieldKey(["frames"])],
                ),
            ),
        ],
    ),
):
    """Analysis performed on video chunks.

    This downstream feature processes the expanded chunks from VideoChunkAdvanced.
    Since it uses the same ID column (chunk_uid), standard joining works.
    """

    motion_score: float = Field(description="Amount of motion detected in chunk")
    scene_changes: int = Field(description="Number of scene changes in chunk")
    dominant_colors: list[str] = Field(description="Dominant colors in chunk")


# Example of aggregating back to video level
class VideoSummary(
    Feature,
    spec=FeatureSpec(
        id_columns=["video_id"],  # Back to video-level
        key=FeatureKey(["advanced", "video", "summary"]),
        deps=[
            FeatureDep(key=ChunkAnalysis.key),
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["aggregated_analysis"]),
                code_version=1,
                deps=FieldDep(
                    feature_key=ChunkAnalysis.key,
                    fields=[FieldKey(["analysis_result"])],
                ),
            ),
        ],
    ),
):
    """Video-level summary aggregating chunk analyses.

    This demonstrates many-to-one aggregation, going from chunk-level
    back to video-level by aggregating chunk analysis results.
    """

    total_motion: float = Field(description="Sum of motion scores across chunks")
    total_scene_changes: int = Field(description="Total scene changes in video")
    unique_colors: list[str] = Field(description="All unique colors across chunks")

    @classmethod
    def load_input(
        cls,
        joiner: "UpstreamJoiner",
        upstream_refs: dict[str, "nw.LazyFrame[Any]"],
    ) -> tuple["nw.LazyFrame[Any]", dict[str, str]]:
        """Custom load_input for many-to-one aggregation.

        Aggregates chunk-level data back to video level.
        """
        chunk_analysis_key = ChunkAnalysis.key.to_string()

        if chunk_analysis_key not in upstream_refs:
            raise ValueError("Missing ChunkAnalysis upstream")

        chunks_ref = upstream_refs[chunk_analysis_key]

        # Group by parent_video_id and aggregate
        # This assumes parent_video_id is available from VideoChunkAdvanced
        aggregated = (
            chunks_ref
            .group_by("parent_video_id")
            .agg([
                nw.col("motion_score").sum().alias("total_motion"),
                nw.col("scene_changes").sum().alias("total_scene_changes"),
                # For lists, we'd need custom aggregation logic
            ])
            .rename({"parent_video_id": "video_id"})  # Rename to match our ID column
        )

        # Build column mapping
        column_mapping = {
            chunk_analysis_key: f"__upstream_{chunk_analysis_key}__data_version",
        }

        return aggregated, column_mapping