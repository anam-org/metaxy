import metaxy as mx


class Video(
    mx.BaseFeature,
    spec=mx.BaseFeatureSpec(
        key="video/raw", id_columns=["video_id"], fields=["audio", "frames"]
    ),
):
    path: str  # where the video is stored


class VideoChunk(
    mx.BaseFeature,
    spec=mx.BaseFeatureSpec(
        key="video/chunk",
        id_columns=["video_chunk_id"],
        fields=["audio", "frames"],
        deps=[
            mx.FeatureDep(
                feature=Video, id_columns_mapping={"video_id": "video_chunk_id"}
            )
        ],
    ),
):
    path: str  # where the video chunk is stored
