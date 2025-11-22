"""Partitioned (category) example with Dagster + Metaxy.

This example demonstrates how to process one category end-to-end
through the entire feature graph using Dagster partitions. Each partition maps to
one category that owns multiple videos, and each video has multiple rows (frames) —
not a 1:1 row-to-partition mapping.

Key Concepts:
1. **Partitions**: Each partition represents one category spanning multiple videos and many rows
2. **Root Features**: Must manually filter samples by partition
3. **Downstream Features**: Automatically filtered by IOManager
4. **End-to-End**: One video flows through the entire pipeline in isolation

This pattern is ideal for:
- Video processing pipelines (one video at a time)
- Document processing (one document at a time)
- Incremental processing of large datasets
- Parallel processing with Dagster's execution model

Before materializing the asset ensure to create partitions in Dagster and run a backfill for these categories (each category contains multiple videos, each with multiple frames):

- video_category_1
- video_category_2

Example
```bash
source .venv/bin/activate
cd example-integration-dagster/
rm -f /tmp/metaxy_dagster*.duckdb
dagster dev -f src/example_integration_dagster/partitioned.py
```
"""

from typing import Any

import dagster as dg
import narwhals as nw
import polars as pl
from dagster import AssetExecutionContext

import metaxy as mx
import metaxy.ext.dagster as mxd
from metaxy.versioning.types import Increment

# ============= Setup =============

# Configure Metaxy store resource
store_resource = mxd.MetaxyMetadataStoreResource.from_config(
    store_name="dev",
)

# Configure IOManager for partitioned pattern
# The partition_key_column tells the IOManager to automatically filter
# downstream features to the current partition/category
metaxy_io_manager = mxd.MetaxyIOManager.from_store(
    store_resource,
    partition_key_column="video_category",  # This enables automatic partition filtering
)

# Define partitions (one per category; each covers many videos and frames/rows)
# This can be populated dynamically as new categories arrive
video_partitions = mxd.create_video_partitions_def(
    name="videos",
    initial_partition_keys=[
        "video_category_1",
        "video_category_2",
    ],  # Add initial partitions
)


# ============= Define Features =============


class RawVideo(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/raw",
        id_columns=["video_category", "video_id", "frame_id"],
        fields=[
            mx.FieldSpec(key="raw_pixels", code_version="1"),
            mx.FieldSpec(key="audio", code_version="1"),
        ],
        metadata={
            "description": "Raw video frames per category with audio and timestamps.",
            "tags": {"owner": "video-team", "tier": "bronze"},
        },
    ),
):
    """Raw video feature with frame-level data."""

    video_id: str
    frame_id: int
    s3_path: str
    frame_timestamp: float


class CleanVideo(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/clean",
        id_columns=["video_category", "video_id", "frame_id"],
        deps=[mx.FeatureDep(feature=mx.FeatureKey(["video", "raw"]))],
        fields=[
            mx.FieldSpec(key="clean_pixels", code_version="1"),
            mx.FieldSpec(key="clean_audio", code_version="1"),
        ],
        metadata={
            "description": "Cleaned video frames derived from raw frames.",
            "tags": {"owner": "video-team", "tier": "silver"},
        },
    ),
):
    """Cleaned video feature derived from raw video."""

    video_id: str
    frame_id: int
    s3_path: str
    frame_timestamp: float
    width: int
    height: int


class VideoEmbeddings(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/embeddings",
        id_columns=["video_category", "video_id", "frame_id"],
        deps=[mx.FeatureDep(feature=mx.FeatureKey(["video", "clean"]))],
        fields=[
            mx.FieldSpec(key="visual_embedding", code_version="1"),
            mx.FieldSpec(key="audio_embedding", code_version="1"),
        ],
        metadata={
            "description": "Embeddings generated from cleaned video frames.",
            "tags": {"owner": "video-team", "tier": "gold"},
        },
    ),
):
    """Video embeddings from clean video."""

    video_id: str
    frame_id: int
    visual_embedding: list[float]
    audio_embedding: list[float]


# ============= Helper Functions =============


def _generate_all_raw_video_samples() -> nw.DataFrame[Any]:
    """Simulate generating samples for all videos across categories.

    Each category owns multiple videos; each video has multiple frames.
    """
    categories: dict[str, list[str]] = {
        "video_category_1": ["video_1", "video_2"],
        "video_category_2": ["video_3", "video_4"],
    }
    data = []
    for category, videos in categories.items():
        for video_id in videos:
            for frame_num in range(10):
                data.append(
                    {
                        "video_category": category,
                        "video_id": video_id,
                        "frame_id": frame_num,
                        "s3_path": f"s3://bucket/raw/{video_id}/frame_{frame_num}.jpg",
                        "frame_timestamp": frame_num * 0.033,  # ~30fps
                        # Provenance for root feature fields
                        "metaxy_provenance_by_field": {
                            "raw_pixels": f"hash_{video_id}_{frame_num}_pixels",
                            "audio": f"hash_{video_id}_{frame_num}_audio",
                        },
                    }
                )

    return nw.from_native(pl.DataFrame(data))


# ============= Define Assets =============


@mxd.asset(feature=RawVideo, partitions_def=video_partitions)
def raw_video(
    context: AssetExecutionContext,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Root asset: Extract raw video frames.

    This is a ROOT FEATURE (no dependencies), so we must:
    1. Generate/load ALL samples
    2. Filter to current partition using filter_samples_by_partition()
    3. Pass filtered samples to resolve_update()

    The IOManager cannot filter samples for root features because it doesn't
    know what samples exist. Only downstream features get automatic filtering.
    """
    video_category = context.partition_key
    context.log.info(f"Processing category: {video_category}")

    # Step 1: Generate ALL samples (simulated)
    # In reality, this might query S3, a database, or a file system
    all_samples = _generate_all_raw_video_samples()

    # Step 2: Filter to current partition (THIS CATEGORY ONLY)
    # This is the key helper function for root features!
    samples = mxd.filter_samples_by_partition(
        samples=all_samples,
        partition_column="video_category",
        partition_key=video_category,
    )

    context.log.info(f"Filtered to {len(samples)} frames for {video_category}")

    # Step 3: Resolve increment with filtered samples
    with store:
        increment = store.resolve_update(RawVideo, samples=samples)

        context.log.info(
            f"RawVideo increment: {len(increment.added)} added, "
            f"{len(increment.changed)} changed, {len(increment.removed)} removed"
        )

        # Write metadata for new/changed frames
        if len(increment.added) > 0:
            store.write_metadata(RawVideo, increment.added)

    return RawVideo.spec()


@mxd.asset(feature=CleanVideo, partitions_def=video_partitions)
def clean_video(
    context: AssetExecutionContext,
    diff: Increment,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Downstream asset: Clean video frames.

    This is a DOWNSTREAM FEATURE (depends on RawVideo), so:
    1. IOManager automatically filters increment to current partition
    2. We receive the filtered Increment via the io_manager input
    3. Process and write results to the store

    The partition_key_column in IOManager config handles the filtering!
    """
    video_category = context.partition_key
    context.log.info(f"Cleaning category: {video_category}")

    context.log.info(
        f"CleanVideo increment: {len(diff.added)} added, "
        f"{len(diff.changed)} changed, {len(diff.removed)} removed"
    )

    if len(diff.added) == 0:
        context.log.info("No new videos to clean")
        return CleanVideo.spec()

    # IOManager already filtered to this category
    with store:
        for chunk in mxd.iter_dataframe_with_progress(
            diff.added,
            chunk_size=100,
            desc="clean_video",
            log_fn=context.log,
            log_level="debug",
            failed_count=0,
            echo_to_stderr=True,
        ):
            # Use Narwhals operations (works with any backend)
            cleaned = chunk.with_columns(
                nw.lit(f"s3://bucket/clean/{video_category}/").alias("s3_path"),
                nw.lit(1920).alias("width"),
                nw.lit(1080).alias("height"),
            )

            # Write cleaned metadata
            store.write_metadata(CleanVideo, cleaned)

    return CleanVideo.spec()


@mxd.asset(feature=VideoEmbeddings, partitions_def=video_partitions)
def video_embeddings(
    context: AssetExecutionContext,
    diff: Increment,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Downstream asset: Generate video embeddings.

    Another DOWNSTREAM FEATURE, demonstrating multi-level dependencies.
    Same pattern as clean_video: automatic filtering by IOManager.
    """
    video_category = context.partition_key
    context.log.info(f"Generating embeddings for category: {video_category}")

    # IOManager automatically filters to this category
    context.log.info(
        f"VideoEmbeddings increment: {len(diff.added)} added, "
        f"{len(diff.changed)} changed, {len(diff.removed)} removed"
    )

    if len(diff.added) == 0:
        context.log.info("No new videos to embed")
        return VideoEmbeddings.spec()

    # Generate embeddings with progress logging
    with store:
        for chunk in mxd.iter_dataframe_with_progress(
            diff.added,
            chunk_size=100,
            desc="video_embeddings",
            log_fn=context.log,
            log_level="debug",
            failed_count=0,
            echo_to_stderr=True,
        ):
            # For list columns, we need to use native operations temporarily
            # as nw.lit() doesn't support list types
            native_df = chunk.to_native()
            if hasattr(native_df, "with_columns"):  # Polars
                import polars as pl

                with_embeddings_native = native_df.with_columns(
                    [
                        pl.lit([0.1] * 512).alias("visual_embedding"),
                        pl.lit([0.2] * 128).alias("audio_embedding"),
                    ]
                )
            else:  # PyArrow or other
                import pyarrow as pa

                # PyArrow doesn't have with_columns, need to add columns differently
                visual_emb = [[0.1] * 512] * len(native_df)
                audio_emb = [[0.2] * 128] * len(native_df)
                with_embeddings_native = native_df.append_column(
                    pa.field("visual_embedding", pa.list_(pa.float64())),
                    pa.array(visual_emb, type=pa.list_(pa.float64())),
                ).append_column(
                    pa.field("audio_embedding", pa.list_(pa.float64())),
                    pa.array(audio_emb, type=pa.list_(pa.float64())),
                )

            with_embeddings = nw.from_native(with_embeddings_native)

            store.write_metadata(VideoEmbeddings, with_embeddings)

    return VideoEmbeddings.spec()


# ============= Dagster Definitions =============

defs = dg.Definitions(
    assets=[raw_video, clean_video, video_embeddings],
    resources={
        "store": store_resource,
        "io_manager": metaxy_io_manager,
    },
)
