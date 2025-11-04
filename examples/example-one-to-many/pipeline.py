import narwhals as nw
import polars as pl
from example_one_to_many.features import Video, VideoChunk
from example_one_to_many.utils import split_video_into_chunks

from metaxy import init_metaxy


def main():
    cfg = init_metaxy()
    store = cfg.get_store("dev")

    # let's pretend somebody has already created the videos for us
    samples = pl.DataFrame(
        {
            "video_id": [1, 2, 3],
            "path": ["video1.mp4", "video2.mp4", "video3.mp4"],
            "provenance_by_field": [
                {"audio": "v1", "frames": "v1"},
                {"audio": "v2", "frames": "v2"},
                {"audio": "v3", "frames": "v3"},
            ],
        }
    )
    with store:
        # showcase: resolve incremental update for a root feature
        diff = store.resolve_update(Video, samples=nw.from_native(samples))
        if len(diff.added) > 0:
            store.write_metadata(Video, diff.added)

    # now we are going to resolve the videos that have to be split to chunks
    with store:
        diff = store.resolve_update(VideoChunk)
        # the DataFrame dimensions matches Video (with ID column renamed)

        print(f"Found {len(diff.added)} videos to process for chunking")

        for row_dict in diff.added.to_polars().iter_rows(named=True):
            print(f"Processing video: {row_dict}")
            # let's split each video to 3-5 chunks randomly

            # The ID column was renamed from video_id to video_chunk_id
            video_chunk_id = row_dict["video_chunk_id"]
            path = row_dict["path"]
            provenance_by_field = row_dict["provenance_by_field"]

            # pretend we split the video into chunks
            chunk_paths = split_video_into_chunks(path)

            # Generate chunk IDs based on the parent video ID
            chunk_ids = [f"{video_chunk_id}_{i}" for i in range(len(chunk_paths))]

            # write the chunks to the store
            chunk_df = pl.DataFrame(
                {
                    "video_chunk_id": chunk_ids,
                    "path": chunk_paths,
                    "provenance_by_field": [provenance_by_field] * len(chunk_paths),
                }
            )
            print(f"Writing {len(chunk_paths)} chunks for video {video_chunk_id}")
            store.write_metadata(VideoChunk, nw.from_native(chunk_df))


if __name__ == "__main__":
    main()
