"""Non-partitioned (data-parallel) example with Dagster + Metaxy.

This example processes all records of a feature at once (no partitions).
It is the simplest processing strategy.

Example
```bash
source .venv/bin/activate
cd example-integration-dagster/
rm -f /tmp/metaxy_dagster*.duckdb
dagster dev -f src/example_integration_dagster/non_partitioned.py
```
"""

import dagster as dg
import narwhals as nw
import polars as pl

import metaxy as mx
import metaxy.ext.dagster as mxd

# ============= Setup =============

# Configure Metaxy store resource
store_resource = mxd.MetaxyMetadataStoreResource.from_config(
    store_name="dev",
)

# Configure IOManager for data-parallel (default)
metaxy_io_manager = mxd.MetaxyIOManager.from_store(store_resource)


# ============= Define Features =============


class RawVideo(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/raw",
        id_columns=["video_id"],
        fields=["frames", "audio"],
        metadata={
            "description": "Raw video feature with frames and audio.",
            "tags": {"owner": "video-team", "priority": "medium"},
        },
    ),
):
    """Raw video feature with frames and audio."""

    video_id: str
    s3_path: str


class CleanVideo(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/clean",
        id_columns=["video_id"],
        deps=[mx.FeatureDep(feature=mx.FeatureKey(["video", "raw"]))],
        fields=[
            mx.FieldSpec(key="frames", code_version="1"),
            mx.FieldSpec(key="audio", code_version="1"),
        ],
        metadata={
            "description": "Clean video feature derived from raw video.",
            "tags": {"owner": "video-team", "priority": "medium"},
        },
    ),
):
    """Cleaned video feature derived from raw video."""

    video_id: str
    s3_path: str
    duration: float
    width: int
    height: int


# ============= Define Assets =============


@mxd.asset(feature=RawVideo)
def raw_video(
    context: dg.AssetExecutionContext,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Root asset that generates raw video metadata.

    Since this is a root feature (no dependencies), we:
    1. Generate samples with provenance
    2. Call resolve_update with samples parameter
    3. Write metadata to store
    """
    context.log.info("Generating raw video samples...")

    # Simulate generating raw video samples
    samples_df = pl.DataFrame(
        {
            "video_id": ["v1", "v2", "v3"],
            "s3_path": [
                "s3://bucket/raw/v1.mp4",
                "s3://bucket/raw/v2.mp4",
                "s3://bucket/raw/v3.mp4",
            ],
            # User must provide provenance for root features
            "metaxy_provenance_by_field": [
                {"frames": "hash_f1", "audio": "hash_a1"},
                {"frames": "hash_f2", "audio": "hash_a2"},
                {"frames": "hash_f3", "audio": "hash_a3"},
            ],
        }
    )

    # Resolve increment (requires samples for root features)
    with store:
        increment = store.resolve_update(RawVideo, samples=nw.from_native(samples_df))

        context.log.info(
            f"RawVideo increment: {increment.added.shape[0]} added, "
            f"{increment.changed.shape[0]} changed, {increment.removed.shape[0]} removed"
        )

        # Write only new/changed metadata
        if increment.added.shape[0] > 0:
            store.write_metadata(RawVideo, increment.added)
        else:
            context.log.info("No new raw videos to write")

    return RawVideo.spec()


@mxd.asset(feature=CleanVideo)
def clean_video(
    context: dg.AssetExecutionContext,
    diff: mxd.Increment,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Downstream asset that cleans raw video.

    The @mxd.asset decorator automatically injects the 'diff' Increment parameter
    for downstream features with dependencies.
    """
    context.log.info("Processing clean video from raw video...")

    context.log.info(
        f"CleanVideo increment: {diff.added.shape[0]} added, "
        f"{diff.changed.shape[0]} changed, {diff.removed.shape[0]} removed"
    )

    if diff.added.shape[0] == 0:
        context.log.info("No new videos to clean")
        return CleanVideo.spec()

    # Simulate processing - add cleaning metadata with progress logging
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
                # Build s3_path by concatenating strings
                nw.concat_str(
                    [
                        nw.lit("s3://bucket/clean/"),
                        nw.col("video_id"),
                        nw.lit(".mp4"),
                    ]
                ).alias("s3_path"),
                # Add other cleaning metadata
                nw.lit(120.0).alias("duration"),
                nw.lit(1920).alias("width"),
                nw.lit(1080).alias("height"),
            )

            # Write cleaned metadata (provenance is inherited automatically)
            store.write_metadata(CleanVideo, cleaned)

    return CleanVideo.spec()


# ============= Define Dagster Definitions =============

defs = dg.Definitions(
    assets=[raw_video, clean_video],
    resources={
        "store": store_resource,
        "io_manager": metaxy_io_manager,
    },
)
