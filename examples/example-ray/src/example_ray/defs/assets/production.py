"""Example 3: Production pipeline with branching IO."""

import dagster as dg
import metaxy as mx
import metaxy.ext.dagster as mxd
import polars as pl
import ray
import ray.data

from example_ray.defs.assets import _ensure_ray
from example_ray.defs.ops.counters import PipelineCounters
from example_ray.defs.ops.error_handling import FailurePolicy
from example_ray.defs.ops.run_config import RunConfig, select_samples
from example_ray.defs.ops.worker import timed_worker
from example_ray.features import Crop, SpeechToText

# --8<-- [start:production_asset]


def process_video(row: dict) -> dict:
    """Simulated video processing that produces STT and crop results."""
    if row.get("sample_uid", "").endswith("_bad"):
        msg = "corrupt audio stream"
        raise ValueError(msg)
    return row


@mxd.metaxify
@dg.asset(
    metadata={"metaxy/feature": "ray/stt"},
    deps=[dg.AssetKey(["ray", "video"])],
)
def video_processing(
    context: dg.AssetExecutionContext,
    store: dg.ResourceParam[mx.MetadataStore],
    config: RunConfig,
) -> None:
    """Process video with branching IO: write STT and Crop per row."""
    _ensure_ray()

    counters = PipelineCounters()
    policy = FailurePolicy(max_failed_fraction=0.3)

    with store:
        stt_increment = store.resolve_update(SpeechToText)
        crop_increment = store.resolve_update(Crop)

    stt_samples = pl.concat(
        [stt_increment.new.to_polars(), stt_increment.stale.to_polars()]
    ).unique("sample_uid")
    crop_samples = pl.concat(
        [crop_increment.new.to_polars(), crop_increment.stale.to_polars()]
    ).unique("sample_uid")

    # Union of sample_uids that need processing for either feature
    all_uids = pl.concat(
        [
            stt_samples.select("sample_uid"),
            crop_samples.select("sample_uid"),
        ]
    ).unique()

    counters.input_count = len(all_uids)

    if counters.input_count == 0:
        context.log.info("No samples to process")
        context.add_output_metadata(counters.to_metadata())
        return

    selected_uids = select_samples(all_uids, config)
    counters.selected_count = len(selected_uids)

    ds = ray.data.from_arrow(selected_uids.to_arrow())
    ds = ds.map(lambda row: timed_worker(row, process_video))

    # Collect results from Ray
    result_rows: list[dict] = []
    for batch in ds.iter_batches(batch_format="pyarrow"):
        for row in pl.DataFrame(batch).iter_rows(named=True):
            result_rows.append(row)
            counters.processed_count += 1
            counters.total_processing_seconds += row["processing_time_seconds"]

    results = pl.DataFrame(result_rows)
    error_rows = results.filter(pl.col("error") != "")
    counters.failed_count = len(error_rows)

    # Branching IO: start from resolve_update frames (preserves system columns),
    # join operational columns from Ray results, and write both features
    operational_cols = results.select("sample_uid", "processing_time_seconds", "error")
    stt_df = stt_samples.join(operational_cols, on="sample_uid")
    crop_df = crop_samples.filter(pl.col("sample_uid").is_in(results["sample_uid"]))

    with store.open("w"):
        store.write(SpeechToText, stt_df)
        store.write(Crop, crop_df)

    error_samples = error_rows["error"].to_list()[:10] if len(error_rows) > 0 else []
    policy.check(counters.processed_count, counters.failed_count, error_samples)

    context.add_output_metadata(counters.to_metadata())


# --8<-- [end:production_asset]
