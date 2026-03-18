"""Introductory pipeline demonstrating Metaxy's core workflow.

This pipeline processes video files through a multi-step ML pipeline:
  Video -> FaceDetection (mocked)
  Video -> AudioTranscription (mocked) -> Summary (mocked)

Each step uses resolve_update to determine which samples need processing,
then writes the results back to the metadata store.
"""

import metaxy as mx
import polars as pl

from example_intro.features import AudioTranscription, FaceDetection, Summary, Video


def main() -> None:
    # --8<-- [start:init]
    # Initialize Metaxy: loads metaxy.toml and discovers features via entrypoints
    config = mx.init()
    store = config.get_store()
    # --8<-- [end:init]

    print("Intro Pipeline")
    print("=" * 60)

    # --8<-- [start:video_samples]
    # Step 1: Register video samples with provenance tracking.
    # metaxy_provenance_by_field tracks the data version of each field.
    # When provenance changes, downstream features are recomputed.
    video_samples = pl.DataFrame(
        {
            "video_id": ["v001", "v002", "v003"],
            "path": [
                "/data/videos/v001.mp4",
                "/data/videos/v002.mp4",
                "/data/videos/v003.mp4",
            ],
            "metaxy_provenance_by_field": [
                {"default": "v001_hash_abc"},
                {"default": "v002_hash_def"},
                {"default": "v003_hash_ghi"},
            ],
        }
    )

    with store.open("w"):
        video_increment = store.resolve_update(Video, samples=video_samples)
        video_to_write = pl.concat(
            [video_increment.new.to_polars(), video_increment.stale.to_polars()]
        )
        if len(video_to_write) > 0:
            store.write(Video, video_to_write)
            print(f"\n[Video] Wrote {len(video_to_write)} samples")
        else:
            print("\n[Video] No new or changed samples")
    # --8<-- [end:video_samples]

    # --8<-- [start:face_detection]
    # Step 2: Run face detection on videos.
    # resolve_update without samples auto-resolves from upstream (Video).
    # It compares upstream provenance to find new/stale samples.
    with store.open("w"):
        face_increment = store.resolve_update(FaceDetection)
        face_to_process = pl.concat(
            [face_increment.new.to_polars(), face_increment.stale.to_polars()]
        )
        if len(face_to_process) > 0:
            # Mock face detection: add computed columns, then select only
            # the columns FaceDetection declares (id + output + metaxy system).
            # This avoids writing upstream columns (e.g. path) into the table.
            face_results = face_to_process.with_columns(
                pl.lit(2).alias("num_faces"),
                (
                    pl.lit("/data/faces/")
                    + pl.col("video_id")
                    + pl.lit("/embeddings.npy")
                ).alias("face_embeddings_path"),
            ).select(
                "video_id", "num_faces", "face_embeddings_path", pl.col("^metaxy_.*$")
            )
            store.write(FaceDetection, face_results)
            print(f"[FaceDetection] Materialized {len(face_to_process)} samples")
        else:
            print("[FaceDetection] No new or changed samples")
    # --8<-- [end:face_detection]

    # --8<-- [start:audio_transcription]
    # Step 3: Transcribe audio from videos.
    # Same pattern: resolve_update finds what needs processing.
    with store.open("w"):
        transcript_increment = store.resolve_update(AudioTranscription)
        transcript_to_process = pl.concat(
            [
                transcript_increment.new.to_polars(),
                transcript_increment.stale.to_polars(),
            ]
        )
        if len(transcript_to_process) > 0:
            # Mock transcription: add computed columns, select only declared columns
            transcript_results = transcript_to_process.with_columns(
                (pl.lit("Transcript for ") + pl.col("video_id") + pl.lit(".")).alias(
                    "transcript"
                ),
                pl.lit("en").alias("language"),
            ).select("video_id", "transcript", "language", pl.col("^metaxy_.*$"))
            store.write(AudioTranscription, transcript_results)
            print(
                f"[AudioTranscription] Materialized {len(transcript_to_process)} samples"
            )
        else:
            print("[AudioTranscription] No new or changed samples")
    # --8<-- [end:audio_transcription]

    # --8<-- [start:summary]
    # Step 4: Generate summaries from transcriptions.
    # Summary depends on AudioTranscription, so it auto-resolves from there.
    with store.open("w"):
        summary_increment = store.resolve_update(Summary)
        summary_to_process = pl.concat(
            [summary_increment.new.to_polars(), summary_increment.stale.to_polars()]
        )
        if len(summary_to_process) > 0:
            # Mock summarization: add computed columns, select only declared columns
            summary_results = summary_to_process.with_columns(
                (pl.lit("Summary of ") + pl.col("video_id") + pl.lit(".")).alias(
                    "summary_text"
                ),
            ).select("video_id", "summary_text", pl.col("^metaxy_.*$"))
            store.write(Summary, summary_results)
            print(f"[Summary] Materialized {len(summary_to_process)} samples")
        else:
            print("[Summary] No new or changed samples")
    # --8<-- [end:summary]

    # Check if anything was processed at all
    total = (
        len(video_increment.new)
        + len(video_increment.stale)
        + len(face_increment.new)
        + len(face_increment.stale)
        + len(transcript_increment.new)
        + len(transcript_increment.stale)
        + len(summary_increment.new)
        + len(summary_increment.stale)
    )
    if total == 0:
        print("\nNo changes detected (idempotent)")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
