import os
import random

import metaxy as mx
import narwhals as nw
import polars as pl

from example_expansion.features import FaceRecognition, Video, VideoChunk
from example_expansion.utils import split_video_into_chunks


def main():
    # Set random seed from environment if provided (for deterministic testing)
    if seed_str := os.environ.get("RANDOM_SEED"):
        random.seed(int(seed_str))
    cfg = mx.init()
    store = cfg.get_store("dev")

    # let's pretend somebody has already created the videos for us
    samples = pl.DataFrame(
        {
            "video_id": [1, 2, 3],
            "path": ["video1.mp4", "video2.mp4", "video3.mp4"],
            "metaxy_provenance_by_field": [
                {"audio": "v1", "frames": "v1"},
                {"audio": "v2", "frames": "v2"},
                {"audio": "v3", "frames": "v3"},
            ],
        }
    )

    with store.open("w"):
        # showcase: resolve incremental update for a root feature
        increment = store.resolve_update(Video, samples=nw.from_native(samples))
        if len(increment.new) > 0:
            print(f"Found {len(increment.new)} new videos")
            store.write(Video, increment.new)

    # Resolve videos that need to be split into chunks
    with store.open("w"):
        increment = store.resolve_update(VideoChunk)
        # the DataFrame dimensions matches Video (with ID column renamed)

        print(
            f"Found {len(increment.new)} new videos and {len(increment.stale)} stale videos that need chunking"
        )

        for row_dict in pl.concat(
            [increment.new.to_polars(), increment.stale.to_polars()]
        ).iter_rows(named=True):
            print(f"Processing video: {row_dict}")
            # let's split each video to 3-5 chunks randomly

            video_id = row_dict["video_id"]
            path = row_dict["path"]

            provenance_by_field = row_dict["metaxy_provenance_by_field"]
            provenance = row_dict["metaxy_provenance"]

            # pretend we split the video into chunks
            chunk_paths = split_video_into_chunks(path)

            # Generate chunk IDs based on the parent video ID
            chunk_ids = [f"{video_id}_{i}" for i in range(len(chunk_paths))]

            # write the chunks to the store
            # CRUSIAL: all the chunks **must share the same provenance values**
            chunk_df = pl.DataFrame(
                {
                    "video_id": [video_id] * len(chunk_paths),
                    "video_chunk_id": chunk_ids,
                    "path": chunk_paths,
                    "metaxy_provenance_by_field": [provenance_by_field]
                    * len(chunk_paths),
                    "metaxy_provenance": [provenance] * len(chunk_paths),
                }
            )
            print(f"Writing {len(chunk_paths)} chunks for video {video_id}")
            store.write(VideoChunk, nw.from_native(chunk_df))

    # Process face recognition on video chunks
    with store.open("w"):
        increment = store.resolve_update(FaceRecognition)
        print(
            f"Found {len(increment.new)} new video chunks and {len(increment.stale)} stale video chunks that need face recognition"
        )

        if len(increment.new) > 0:
            # simulate face detection on each chunk
            face_data = []
            for row_dict in pl.concat(
                [increment.new.to_polars(), increment.stale.to_polars()]
            ).iter_rows(named=True):
                video_chunk_id = row_dict["video_chunk_id"]
                provenance_by_field = row_dict["metaxy_provenance_by_field"]
                provenance = row_dict["metaxy_provenance"]

                # simulate detecting random number of faces
                num_faces = random.randint(0, 10)

                face_data.append(
                    {
                        "video_chunk_id": video_chunk_id,
                        "num_faces": num_faces,
                        "metaxy_provenance_by_field": provenance_by_field,
                        "metaxy_provenance": provenance,
                    }
                )

            face_df = pl.DataFrame(face_data)
            print(f"Writing face recognition results for {len(face_data)} chunks")
            store.write(FaceRecognition, nw.from_native(face_df))


if __name__ == "__main__":
    main()
