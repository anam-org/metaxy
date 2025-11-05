import metaxy as mx


class Video(
    mx.BaseFeature,
    spec=mx.BaseFeatureSpec(
        key="video/raw",
        id_columns=["video_id"],
        fields=[
            mx.FieldSpec(key="audio", code_version="1"),
            mx.FieldSpec(key="frames", code_version="1"),
        ],
    ),
):
    video_id: str
    path: str  # where the video is stored


class VideoChunk(
    mx.BaseFeature,
    spec=mx.BaseFeatureSpec(
        key="video/chunk",
        id_columns=["video_chunk_id"],
        fields=[
            mx.FieldSpec(key="audio", code_version="1"),
            mx.FieldSpec(key="frames", code_version="1"),
        ],
        deps=[mx.FeatureDep(feature="video/raw")],
        lineage=mx.LineageRelationship.expansion(on=["video_id"]),
    ),
):
    video_id: str  # points to the parent video
    video_chunk_id: str
    path: str  # where the video chunk is stored
