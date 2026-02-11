"""Quickstart pipeline demonstrating Metaxy's resolve_update API."""

import metaxy as mx
import polars as pl

from example_quickstart.features import Audio, Video

config = mx.init()

print("Quickstart Pipeline")
print("=" * 60)
store = config.get_store()

# --8<-- [start:resolve_video]
# Prepare a DataFrame with incoming metadata
samples = pl.DataFrame(
    {
        "id": ["vid_001", "vid_002", "vid_003"],
        "raw_video_path": [
            "/data/raw/vid_001.mp4",
            "/data/raw/vid_002.mp4",
            "/data/raw/vid_003.mp4",
        ],
        "metaxy_provenance_by_field": [
            {"default": "a1n892ja"},  # can be a hash sum
            {"default": "2024-01-15T10:30:00Z"},  # or a modified_on timestamp
            {"default": "v1.2.3"},  # or just any string
        ],
    }
)

with store:
    increment = store.resolve_update(Video, samples=samples)
# --8<-- [end:resolve_video]

n_video = len(increment.new) + len(increment.stale)
print(f"\n[Video] {len(increment.new)} new, {len(increment.stale)} stale")

# --8<-- [start:process_video]
to_process = pl.concat([increment.new.to_polars(), increment.stale.to_polars()])

result = []
for row in to_process.iter_rows(named=True):
    path = f"/data/processed/{row['id']}/{row['metaxy_data_version']}/video.mp4"
    result.append({**row, "path": path})
# --8<-- [end:process_video]

# --8<-- [start:write_video]
if result:
    with store.open("w"):
        store.write(Video, pl.DataFrame(result))
# --8<-- [end:write_video]

if result:
    print(f"[Video] Wrote {len(result)} samples")

# --8<-- [start:resolve_audio]
with store:
    audio_increment = store.resolve_update(Audio)
# --8<-- [end:resolve_audio]

n_audio = len(audio_increment.new) + len(audio_increment.stale)
print(f"\n[Audio] {len(audio_increment.new)} new, {len(audio_increment.stale)} stale")

audio_to_process = pl.concat(
    [audio_increment.new.to_polars(), audio_increment.stale.to_polars()]
)

audio_result = []
for row in audio_to_process.iter_rows(named=True):
    path = f"/data/processed/{row['id']}/{row['metaxy_provenance']}/audio.wav"
    audio_result.append({**row, "path": path})

if audio_result:
    with store.open("w"):
        store.write(Audio, pl.DataFrame(audio_result))
    print(f"[Audio] Wrote {len(audio_result)} samples")

if n_video == 0 and n_audio == 0:
    print("\nNo changes detected (idempotent)")

print("\n✅ Pipeline complete!")
