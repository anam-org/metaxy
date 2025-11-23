"""Branch + subsample Dagster example with Metaxy.

This example combines two patterns:
1) **Branching**: Write to a branch store while reading the root feature from a production fallback.
2) **Subsampling**: During each materialization, choose how to sample the root data:
   - Random: take a random sample of N rows
   - Keys: keep only specified video_ids

Select the sampling mode per materialization via Dagster run config.

Example run config (random sample of 2):
```yaml
assets:
  raw_video:
    config:
      sample_mode: random
      sample_size: 2
```

Example run config (explicit keys):
```yaml
assets:
  raw_video:
    config:
      sample_mode: keys
      sample_keys: ["video_123", "video_456"]
```

Example
```bash
source .venv/bin/activate
cd example-integration-dagster/
rm -f /tmp/metaxy_dagster*.duckdb
dagster dev -f src/example_integration_dagster/branch_subsampled.py
```
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

# IOManager can also be given target keys via Dagster run config if desired
metaxy_io_manager = mxd.MetaxyIOManager.from_store(
    store_resource,
    target_key_column="video_id",
)


class RawVideo(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/raw",
        id_columns=["video_id"],
        fields=[mx.FieldSpec(key="payload", code_version="1")],
        metadata={
            "description": "Raw video payloads ingested from production or sample data.",
            "tags": {"owner": "video-team", "priority": "medium"},
        },
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
        metadata={
            "description": "Cleaned video payloads derived from raw inputs.",
            "tags": {"owner": "video-team", "priority": "medium"},
        },
    ),
):
    video_id: str
    clean_payload: str


def _generate_default_samples() -> nw.DataFrame[Any]:
    """Fallback samples when prod doesn't have the root feature yet."""
    return nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["video_123", "video_456", "video_789"],
                "payload": ["example data a", "example data b", "example data c"],
                # Provenance for root fields so resolve_update can diff
                "metaxy_provenance_by_field": [
                    {"payload": "hash_video_123"},
                    {"payload": "hash_video_456"},
                    {"payload": "hash_video_789"},
                ],
            }
        )
    )


def _get_prod_samples(
    store: mxd.MetaxyMetadataStoreResource, log_fn
) -> nw.DataFrame[Any]:
    """Read samples from production fallback, seeding defaults if absent."""
    samples: nw.DataFrame[Any] = _generate_default_samples()

    actual_store = store.get_store()
    fallback: mx.MetadataStore | None = (
        actual_store.fallback_stores[0] if actual_store.fallback_stores else None
    )
    if fallback is None:
        log_fn("No production fallback configured; using default samples.")
        return samples

    with fallback:
        try:
            if not fallback.has_feature(RawVideo, check_fallback=False):
                log_fn("Seeding production fallback with default samples for RawVideo.")
                fallback.write_metadata(RawVideo, samples)

            prod_samples = fallback.read_metadata(RawVideo, allow_fallback=False)
            prod_samples_any: Any = prod_samples
            if hasattr(prod_samples_any, "collect"):
                prod_samples_eager = prod_samples_any.collect()
            else:
                prod_samples_eager = nw.from_native(prod_samples_any)

            if len(prod_samples_eager) > 0:
                log_fn(f"Using {len(prod_samples_eager)} rows from production fallback")
                samples = prod_samples_eager
        except Exception as exc:  # pragma: no cover - example resilience
            log_fn(f"Fallback read failed, using defaults: {exc}")

    return samples


@mxd.asset(
    feature=RawVideo,
    config_schema=mxd.sampling_config_schema(default_size=2),
)
def raw_video(
    context: dg.AssetExecutionContext,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Root asset: read from production fallback, then subsample per config."""
    config = context.op_config or {}
    sample_mode = config.get("sample_mode")
    sample_size = config.get("sample_size")
    sample_keys = config.get("sample_keys")

    samples = _get_prod_samples(store, context.log.info)
    sampled = mxd.apply_sampling(
        samples,
        sample_mode=sample_mode,
        sample_keys=sample_keys,
        sample_size=sample_size,
        log_fn=context.log.info,
    )

    context.log.info(
        "Sampling mode=%s size=%s keys=%s => %d rows",
        sample_mode,
        sample_size,
        sample_keys,
        len(sampled),
    )

    with store:
        increment = store.resolve_update(RawVideo, samples=sampled)
        context.log.info(
            "RawVideo increment in branch: %d added, %d changed, %d removed",
            len(increment.added),
            len(increment.changed),
            len(increment.removed),
        )
        if len(increment.added) > 0:
            store.write_metadata(RawVideo, increment.added)
        else:
            context.log.info("No new raw samples to write after sampling")

    return RawVideo.spec()


@mxd.asset(feature=CleanVideo)
def clean_video(
    context: dg.AssetExecutionContext,
    diff: Increment,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Downstream asset: IOManager can key-filter; asset processes in chunks with logging."""
    context.log.info(
        "CleanVideo diff after target-key filtering: %d added, %d changed, %d removed",
        len(diff.added),
        len(diff.changed),
        len(diff.removed),
    )

    if len(diff.added) == 0:
        context.log.info("No new videos to clean in branch subsample run")
        return CleanVideo.spec()

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
            cleaned = chunk.with_columns(
                nw.col("payload")
                .str.replace("example", "cleaned")
                .alias("clean_payload")
            )
            store.write_metadata(CleanVideo, cleaned)

    return CleanVideo.spec()


defs = dg.Definitions(
    assets=[raw_video, clean_video],
    resources={
        "store": store_resource,
        "io_manager": metaxy_io_manager,
    },
)
