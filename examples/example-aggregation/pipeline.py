"""Pipeline demonstrating N:1 aggregation lineage.

This example shows how Metaxy handles aggregation lineage where multiple
upstream records (audio recordings) are aggregated into a single downstream
record (speaker embedding).

Key concepts:
- Multiple audio recordings per speaker are aggregated into one embedding
- When any audio for a speaker changes, the speaker embedding needs recomputation
"""

import metaxy as mx
import polars as pl
from example_aggregation.features import Audio, SpeakerEmbedding

# Audio samples: 2 speakers with 2 recordings each
AUDIO_SAMPLES = pl.DataFrame(
    [
        {
            "audio_id": "a1",
            "speaker_id": "s1",
            "duration_seconds": 30.5,
            "path": "audio/s1_recording1.wav",
            "metaxy_provenance_by_field": {"default": "a1_v1"},
        },
        {
            "audio_id": "a2",
            "speaker_id": "s1",
            "duration_seconds": 45.2,
            "path": "audio/s1_recording2.wav",
            "metaxy_provenance_by_field": {"default": "a2_v1"},
        },
        {
            "audio_id": "a3",
            "speaker_id": "s2",
            "duration_seconds": 60.0,
            "path": "audio/s2_recording1.wav",
            "metaxy_provenance_by_field": {"default": "a3_v1"},
        },
        {
            "audio_id": "a4",
            "speaker_id": "s2",
            "duration_seconds": 35.8,
            "path": "audio/s2_recording2.wav",
            "metaxy_provenance_by_field": {"default": "a4_v1"},
        },
    ]
)


def main():
    cfg = mx.init_metaxy()
    store = cfg.get_store("dev")

    # Step 1: Write audio metadata
    with store:
        diff = store.resolve_update(Audio, samples=AUDIO_SAMPLES)
        if len(diff.added) > 0:
            print(f"Found {len(diff.added)} new audio recordings")
            store.write_metadata(Audio, diff.added)
        elif len(diff.changed) > 0:
            print(f"Found {len(diff.changed)} changed audio recordings")
            store.write_metadata(Audio, diff.changed)
        else:
            print("No new or changed audio recordings")

    # Step 2: Compute speaker embeddings
    with store:
        diff = store.resolve_update(SpeakerEmbedding)

        added_df = diff.added.to_polars()
        changed_df = diff.changed.to_polars()

        speakers_to_process = (
            pl.concat([added_df, changed_df]).select("speaker_id").unique().sort("speaker_id").to_series().to_list()
        )

        print(f"Found {len(speakers_to_process)} speakers that need embedding computation")

        if speakers_to_process:
            embedding_data = []

            for speaker_id in speakers_to_process:
                speaker_rows = pl.concat([added_df, changed_df]).filter(pl.col("speaker_id") == speaker_id)

                provenance_by_field = speaker_rows["metaxy_provenance_by_field"][0]
                provenance = speaker_rows["metaxy_provenance"][0]

                n_audio = len(speaker_rows)
                print(f"  Computing embedding for speaker {speaker_id} from {n_audio} audio recordings")

                embedding_data.append(
                    {
                        "speaker_id": speaker_id,
                        "n_dim": 512,
                        "path": f"embeddings/{speaker_id}.npy",
                        "metaxy_provenance_by_field": provenance_by_field,
                        "metaxy_provenance": provenance,
                    }
                )

            embedding_df = pl.DataFrame(embedding_data)
            print(f"Writing embeddings for {len(embedding_data)} speakers")
            store.write_metadata(SpeakerEmbedding, embedding_df)


if __name__ == "__main__":
    main()
