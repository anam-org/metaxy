# --8<-- [start:video]
import metaxy as mx


class Video(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/raw",
        id_columns=["video_id"],
        fields=[
            mx.FieldSpec(key="audio", code_version="1"),
            "frames",
        ],
    ),
):
    video_id: str
    path: str  # where the video is stored


# --8<-- [end:video]


# --8<-- [start:video_chunk]
class VideoChunk(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key=["video", "chunk"],
        id_columns=["video_chunk_id"],
        deps=[
            mx.FeatureDep(
                feature=Video,
                lineage=mx.LineageRelationship.expansion(on=["video_id"]),
            )
        ],
        fields=["audio", "frames"],
    ),
):
    video_id: str  # points to the parent video
    video_chunk_id: str
    path: str  # where the video chunk is stored


# --8<-- [end:video_chunk]


# --8<-- [start:face_recognition]
class FaceRecognition(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key=["video", "faces"],
        id_columns=["video_chunk_id"],
        deps=[
            mx.FeatureDep(
                feature=VideoChunk,
                fields_mapping=mx.FieldsMapping.specific(mapping={mx.FieldKey("faces"): {mx.FieldKey("frames")}}),
            )
        ],
        fields=["faces"],
    ),
):
    video_chunk_id: str
    num_faces: int  # number of faces detected


# --8<-- [end:face_recognition]
