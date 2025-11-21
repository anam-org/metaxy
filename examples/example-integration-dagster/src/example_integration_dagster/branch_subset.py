"""Branching example that reuses production metadata but only processes a subset.

This pattern is useful for testing or hotfixing a small set of entities:

- The branch store reads the root feature from a production fallback
- Downstream assets are filtered to specific keys via the IOManager
- The branch run writes its own metadata without touching production
"""

from typing import Any

import dagster as dg
import narwhals as nw
import polars as pl

import metaxy as mx
import metaxy.ext.dagster as mxd
from metaxy.versioning.types import Increment

# Configure the branch store with production as a fallback
store_resource = mxd.MetaxyMetadataStoreResource.from_config(
    store_name="branch",
    fallback_stores=["production"],
)

# Only process the listed keys in the branch run
metaxy_io_manager = mxd.MetaxyIOManager.from_store(
    store_resource,
    target_key_column="video_id",
    target_keys=["video_123", "video_456"],
)


class RawVideo(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/raw",
        id_columns=["video_id"],
        fields=[mx.FieldSpec(key="payload", code_version="1")],
    ),
):
    video_id: str
    payload: str


class CleanVideo(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/clean",
        deps=[mx.FeatureDep(feature=mx.FeatureKey(["video", "raw"]))],
        id_columns=["video_id"],
        fields=[mx.FieldSpec(key="clean_payload", code_version="1")],
    ),
):
    video_id: str
    clean_payload: str


def _generate_default_samples() -> nw.DataFrame[Any]:
    """Fallback samples when prod doesn't have the root feature yet."""
    return nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["video_123"],
                "payload": ["example data"],
                # Provenance for root fields so resolve_update can diff
                "metaxy_provenance_by_field": [{"payload": "hash_video_123"}],
            }
        )
    )


@mxd.asset(feature=RawVideo)
def raw_video(
    context: dg.AssetExecutionContext, store: mxd.MetaxyMetadataStoreResource
) -> mx.FeatureSpec:
    """Root asset: read from production fallback if available, write to branch."""
    context.log.info("Loading RawVideo from production fallback (if present)")
    samples: nw.DataFrame[Any] = _generate_default_samples()

    # Get the underlying MetadataStore to access fallback_stores
    actual_store = store.get_store()
    fallback: mx.MetadataStore | None = (
        actual_store.fallback_stores[0] if actual_store.fallback_stores else None
    )
    if fallback is not None:
        with fallback:
            try:
                prod_samples = fallback.read_metadata(RawVideo, allow_fallback=False)
                prod_samples_any: Any = prod_samples
                if hasattr(prod_samples_any, "collect"):
                    prod_samples_eager = prod_samples_any.collect()
                else:
                    prod_samples_eager = nw.from_native(prod_samples_any)

                if len(prod_samples_eager) > 0:
                    samples = prod_samples_eager
                    context.log.info(
                        "Using %d rows from production fallback", len(samples)
                    )
            except Exception as exc:  # pragma: no cover - example resilience
                context.log.warning("Fallback read failed, using defaults: %s", exc)

    with store:
        increment = store.resolve_update(RawVideo, samples=samples)
        context.log.info(
            "RawVideo increment in branch: %d added, %d changed, %d removed",
            len(increment.added),
            len(increment.changed),
            len(increment.removed),
        )
        if len(increment.added) > 0:
            store.write_metadata(RawVideo, increment.added)

    return RawVideo.spec()


@mxd.asset(feature=CleanVideo)
def clean_video(
    context: dg.AssetExecutionContext,
    diff: Increment,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Downstream asset: IOManager filters to target keys before execution."""
    context.log.info(
        "CleanVideo diff after target-key filtering: %d added, %d changed, %d removed",
        len(diff.added),
        len(diff.changed),
        len(diff.removed),
    )

    if len(diff.added) > 0:
        # Use Narwhals operations (works with any backend)
        cleaned = diff.added.with_columns(
            nw.col("payload").str.replace("example", "cleaned").alias("clean_payload")
        )
        with store:
            store.write_metadata(CleanVideo, cleaned)

    return CleanVideo.spec()


defs = dg.Definitions(
    assets=[raw_video, clean_video],
    resources={
        "store": store_resource,
        "io_manager": metaxy_io_manager,
    },
)
