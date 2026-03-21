"""Example 2: Ray Data inside Dagster assets with per-row writes."""

import dagster as dg
import metaxy as mx
import metaxy.ext.dagster as mxd
import polars as pl
import ray
import ray.data
from metaxy.ext.ray import MetaxyDatasource
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD

from example_ray.features import Crop, FaceDetection, SpeechToText, Video


def _ensure_ray() -> None:
    if not ray.is_initialized():
        ray.init(_skip_env_hook=True)


# --8<-- [start:root_asset]


@mxd.metaxify
@dg.asset(
    metadata={"metaxy/feature": "ray/video"},
)
def video_metadata(
    store: dg.ResourceParam[mx.MetadataStore],
) -> None:
    """Seed root video metadata."""
    df = pl.DataFrame(
        {
            "sample_uid": ["vid_001", "vid_002", "vid_003", "vid_004_bad"],
            "frames": [120, 240, 180, 90],
            "duration": [4.0, 8.0, 6.0, 3.0],
            "size": [1024, 2048, 1536, 512],
            METAXY_PROVENANCE_BY_FIELD: [
                {"audio": "audio_hash_001", "frames": "frames_hash_001"},
                {"audio": "audio_hash_002", "frames": "frames_hash_002"},
                {"audio": "audio_hash_003", "frames": "frames_hash_003"},
                {"audio": "audio_hash_004", "frames": "frames_hash_004"},
            ],
        }
    )
    with store.open("w"):
        store.write(Video, df)


# --8<-- [end:root_asset]

# --8<-- [start:crop_asset]


@mxd.metaxify
@dg.asset(
    metadata={"metaxy/feature": "ray/crop"},
    deps=[dg.AssetKey(["ray", "video"])],
)
def crop_metadata(
    store: dg.ResourceParam[mx.MetadataStore],
) -> None:
    """Produce crop metadata from video samples."""
    with store.open("w"):
        increment = store.resolve_update(Crop)
        all_samples = pl.concat(
            [increment.new.to_polars(), increment.stale.to_polars()]
        )
        store.write(Crop, all_samples)


# --8<-- [end:crop_asset]

# --8<-- [start:stt_asset]


def transcribe(row: dict) -> dict:
    """Simulated speech-to-text transcription."""
    row["processing_time_seconds"] = 1.5
    row["error"] = ""
    return row


@mxd.metaxify
@dg.asset(
    metadata={"metaxy/feature": "ray/stt"},
    deps=[dg.AssetKey(["ray", "video"])],
)
def speech_to_text(
    context: dg.AssetExecutionContext,
    store: dg.ResourceParam[mx.MetadataStore],
) -> None:
    """Transcribe audio using Ray Data with per-row writes."""
    _ensure_ray()
    config = mx.MetaxyConfig.get()

    with store:
        increment = store.resolve_update(SpeechToText)
    total = len(increment.new) + len(increment.stale)

    if total == 0:
        context.log.info("No samples to process")
        return

    datasource = MetaxyDatasource(
        feature=SpeechToText,
        store=store,
        config=config,
        incremental=True,
    )
    ds = ray.data.read_datasource(datasource)
    ds = ds.map(transcribe)

    with mx.BufferedMetadataWriter(store) as writer:
        for batch in ds.iter_batches(batch_format="pyarrow"):
            for row in pl.DataFrame(batch).iter_rows(named=True):
                writer.put({SpeechToText: pl.DataFrame([row])})

    context.add_output_metadata({"rows_processed": total})


# --8<-- [end:stt_asset]

# --8<-- [start:face_detection_asset]


def detect_faces(row: dict) -> dict:
    """Simulated face detection."""
    row["processing_time_seconds"] = 2.0
    row["tokens_used"] = 100
    row["error"] = ""
    return row


@mxd.metaxify
@dg.asset(
    metadata={"metaxy/feature": "ray/face_detection"},
    deps=[dg.AssetKey(["ray", "crop"])],
)
def face_detection(
    context: dg.AssetExecutionContext,
    store: dg.ResourceParam[mx.MetadataStore],
) -> None:
    """Detect faces in cropped frames using Ray Data."""
    _ensure_ray()
    config = mx.MetaxyConfig.get()

    with store:
        increment = store.resolve_update(FaceDetection)
    total = len(increment.new) + len(increment.stale)

    if total == 0:
        context.log.info("No samples to process")
        return

    datasource = MetaxyDatasource(
        feature=FaceDetection,
        store=store,
        config=config,
        incremental=True,
    )
    ds = ray.data.read_datasource(datasource)
    ds = ds.map(detect_faces)

    with mx.BufferedMetadataWriter(store) as writer:
        for batch in ds.iter_batches(batch_format="pyarrow"):
            for row in pl.DataFrame(batch).iter_rows(named=True):
                writer.put({FaceDetection: pl.DataFrame([row])})

    context.add_output_metadata({"rows_processed": total})


# --8<-- [end:face_detection_asset]
