import metaxy as mx


class Video(
    mx.Feature,
    spec=mx.FeatureSpec(
        key=mx.FeatureKey(["video", "raw"]),
        id_columns=["video_id"],
        fields=["audio", "frames"],
    ),
):
    video_id: str
    path: str  # where the video is stored


class VideoChunk(
    mx.Feature,
    spec=mx.FeatureSpec(
        key=mx.FeatureKey(["video", "chunk"]),
        id_columns=["video_chunk_id"],
        fields=["audio", "frames"],
        deps=[mx.FeatureDep(feature=Video)],
        lineage=mx.LineageRelationship.expansion()
    ),
):
    video_id: str  # points to the parent video
    video_chunk_id: str
    path: str  # where the video chunk is stored
