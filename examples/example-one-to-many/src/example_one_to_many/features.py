import metaxy as mx


class Video(
    mx.Feature,
    spec=mx.FeatureSpec(
        key=["video", "raw"],
        id_columns=["video_id"],
        fields=[
            mx.FieldSpec(key=["audio"], code_version="1"),
            mx.FieldSpec(key=["frames"], code_version="1"),
        ],
    ),
):
    video_id: str
    path: str  # where the video is stored


class VideoChunk(
    mx.Feature,
    spec=mx.FeatureSpec(
        key=["video", "chunk"],
        id_columns=["video_chunk_id"],
        deps=[mx.FeatureDep(feature=["video", "raw"])],
        fields=[
            mx.FieldSpec(key=["audio"], code_version="1"),
            mx.FieldSpec(key=["frames"], code_version="1"),
        ],
        lineage=mx.LineageRelationship.expansion(on=["video_id"]),
    ),
):
    video_id: str  # points to the parent video
    video_chunk_id: str
    path: str  # where the video chunk is stored
