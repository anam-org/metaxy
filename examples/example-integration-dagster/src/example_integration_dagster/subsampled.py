"""Subsampled + key-filtered Dagster example with Metaxy.

This example demonstrates two concerns:
1) **Sampling**: The root asset reads a subsample from a fallback store. If the
   fallback store is empty, we seed it with a tiny demo subset.
2) **Key-based execution**: The IOManager is configured to only pass specific
   video_ids downstream. You can change the keys via Dagster run config on the
   IOManager.

Usage:
    dagster dev -m example_integration_dagster.subsampled
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import dagster as dg
import narwhals as nw
import polars as pl

import metaxy as mx
import metaxy.ext.dagster as mxd
from metaxy.versioning.types import Increment

# ============= Setup =============

# Configure Metaxy store resource (uses DuckDB from metaxy.toml)
store_resource = mxd.MetaxyMetadataStoreResource.from_config(store_name="dev")

# Configure IOManager to filter to specific video IDs
# Override `target_keys` via run config to change which IDs flow through.
metaxy_io_manager = mxd.MetaxyIOManager.from_store(
    store_resource,
    target_key_column="video_id",
    target_keys=["video_1", "video_2"],
)


# ============= Define Features =============


class RawVideo(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/raw",
        id_columns=["video_id"],
        fields=[
            mx.FieldSpec(key="frames", code_version="1"),
            mx.FieldSpec(key="audio", code_version="1"),
        ],
    ),
):
    """Raw video feature containing media pointers."""

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
            mx.FieldSpec(key="duration", code_version="1"),
        ],
    ),
):
    """Cleaned video feature derived from raw video."""

    video_id: str
    s3_path: str
    duration: float


# ============= Sampling helpers =============

_SUBSAMPLED_RAW: nw.DataFrame[Any] = nw.from_native(
    pl.DataFrame(
        {
            "video_id": ["video_1", "video_2", "video_3"],
            "s3_path": [
                "s3://bucket/raw/video_1.mp4",
                "s3://bucket/raw/video_2.mp4",
                "s3://bucket/raw/video_3.mp4",
            ],
            "metaxy_provenance_by_field": [
                {"frames": "hash_v1_frames", "audio": "hash_v1_audio"},
                {"frames": "hash_v2_frames", "audio": "hash_v2_audio"},
                {"frames": "hash_v3_frames", "audio": "hash_v3_audio"},
            ],
        }
    )
)


def _load_subsample_from_fallback(
    store: mxd.MetaxyMetadataStoreResource, log_fn: Callable[[str], Any]
) -> nw.DataFrame[Any]:
    """Return subsampled rows from the first configured fallback store."""
    # Access fallback store (ConfigurableResource proxies to underlying MetadataStore)
    fallback: mx.MetadataStore | None = (
        cast(list[mx.MetadataStore], store.fallback_stores)[0]
        if store.fallback_stores
        else None
    )
    if fallback is None:
        log_fn("No fallback store configured; using in-memory demo subsample.")
        return _SUBSAMPLED_RAW

    with fallback:
        if not fallback.has_feature(RawVideo, check_fallback=False):
            log_fn("Seeding fallback store with demo subsample for RawVideo.")
            fallback.write_metadata(RawVideo, _SUBSAMPLED_RAW)

        log_fn("Using subsample from fallback store.")
        # read_metadata returns a LazyFrame; collect to eager DataFrame
        return fallback.read_metadata(RawVideo, allow_fallback=False).collect()


# ============= Define Assets =============


@mxd.asset(feature=RawVideo)
def raw_video(
    context,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Root asset that pulls a subsample from the fallback store."""
    samples = _load_subsample_from_fallback(store, context.log.info)
    context.log.info(f"Loaded {len(samples)} raw samples from fallback for subsampling")

    with store:
        increment = store.resolve_update(RawVideo, samples=samples)
        if len(increment.added) > 0:
            store.write_metadata(RawVideo, increment.added)

    return RawVideo.spec()


@mxd.asset(feature=CleanVideo)
def clean_video(
    context,
    diff: Increment,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Downstream asset that only processes the configured video IDs."""
    context.log.info(
        f"CleanVideo increment after key filter: {len(diff.added)} added, "
        f"{len(diff.changed)} changed, {len(diff.removed)} removed"
    )

    if len(diff.added) > 0:
        # Use Narwhals operations (works with any backend)
        cleaned = diff.added.with_columns(
            nw.lit(120.0).alias("duration"),
            nw.concat_str(
                [
                    nw.lit("s3://bucket/clean/"),
                    nw.col("video_id"),
                    nw.lit(".mp4"),
                ]
            ).alias("s3_path"),
        )
        with store:
            store.write_metadata(CleanVideo, cleaned)

    return CleanVideo.spec()


# ============= Dagster Definitions =============

defs = dg.Definitions(
    assets=[raw_video, clean_video],
    resources={
        "store": store_resource,
        "io_manager": metaxy_io_manager,
    },
)
