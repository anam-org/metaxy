import polars as pl
from example_one_to_many.features import Video

from metaxy import init_metaxy


def main():
    cfg = init_metaxy()
    store = cfg.get_store("dev")

    # let's pretend somebody has already created the videos for us

    with store:
        store.write_metadata(
            Video,
            pl.DataFrame(
                {
                    "video_id": [1, 2, 3],
                    "path": ["video1.mp4", "video2.mp4", "video3.mp4"],
                    "provenance_by_field": [
                        {"audio": "v1", "frames": "v1"},
                        {"audio": "v2", "frames": "v2"},
                        {"audio": "v3", "frames": "v3"},
                    ],
                }
            ),
        )


if __name__ == "__main__":
    main()
