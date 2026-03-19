"""Example 1: Ray Data with MetaxyDatasource and MetaxyDatasink.

Run with: python -m example_ray.basic_pipeline
"""

import metaxy as mx
import polars as pl
import ray
import ray.data
from metaxy.ext.ray import MetaxyDatasink, MetaxyDatasource
from metaxy.metadata_store.system.storage import SystemTableStorage
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD

from example_ray.features import SpeechToText, Video

# --8<-- [start:basic_pipeline]


def seed_video_metadata(store: mx.MetadataStore) -> None:
    """Write synthetic root video metadata into the store."""
    samples = pl.DataFrame(
        {
            "sample_uid": ["vid_001", "vid_002", "vid_003"],
            "frames": [120, 240, 180],
            "duration": [4.0, 8.0, 6.0],
            "size": [1024, 2048, 1536],
            METAXY_PROVENANCE_BY_FIELD: [
                {"audio": "audio_hash_001", "frames": "frames_hash_001"},
                {"audio": "audio_hash_002", "frames": "frames_hash_002"},
                {"audio": "audio_hash_003", "frames": "frames_hash_003"},
            ],
        }
    )
    with store.open("w"):
        store.write(Video, samples)
    print(f"  Seeded {len(samples)} video samples")


def transcribe(row: dict) -> dict:
    """Simulated speech-to-text transcription."""
    row["processing_time_seconds"] = 1.5
    row["error"] = ""
    return row


def run_stt_pipeline(store: mx.MetadataStore, config: mx.MetaxyConfig) -> None:
    """Read incremental samples via MetaxyDatasource, write via MetaxyDatasink."""
    datasource = MetaxyDatasource(
        feature=SpeechToText,
        store=store,
        config=config,
        incremental=True,
    )

    ds = ray.data.read_datasource(datasource)
    ds = ds.map(transcribe)

    datasink = MetaxyDatasink(
        feature=SpeechToText,
        store=store,
        config=config,
    )
    ds.write_datasink(datasink)

    result = datasink.result
    if result.rows_written == 0:
        print("  Nothing to process (idempotent)")
    else:
        print(f"  Processed {result.rows_written} samples")


# --8<-- [end:basic_pipeline]

if __name__ == "__main__":
    ray.init(_skip_env_hook=True)

    config = mx.init()
    store = config.get_store()

    print("Step 1: Seed video metadata")
    seed_video_metadata(store)

    print("Step 2: Push graph snapshot")
    with store.open("w"):
        SystemTableStorage(store).push_graph_snapshot()

    print("Step 3: Run STT pipeline (first run)")
    run_stt_pipeline(store, config)

    print("Step 4: Re-run STT pipeline (idempotent)")
    run_stt_pipeline(store, config)
