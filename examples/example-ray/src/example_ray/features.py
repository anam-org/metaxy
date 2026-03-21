"""Feature definitions for the Ray example (video processing domain)."""

import metaxy as mx

# --8<-- [start:features]


class Video(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="ray/video",
        fields=[
            mx.FieldSpec(key="audio", code_version="1"),
            mx.FieldSpec(key="frames", code_version="1"),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Video metadata feature (root)."""

    frames: int
    duration: float
    size: int


class SpeechToText(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="ray/stt",
        deps=[mx.FeatureDep(feature=Video)],
        fields=[
            mx.FieldSpec(
                key="transcription",
                code_version="1",
                deps=[mx.FieldDep(feature=Video, fields=["audio"])],
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Speech-to-text transcription derived from video audio."""

    processing_time_seconds: float
    error: str | None


class Crop(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="ray/crop",
        deps=[mx.FeatureDep(feature=Video)],
        fields=[
            mx.FieldSpec(
                key="audio",
                code_version="1",
                deps=[mx.FieldDep(feature=Video, fields=["audio"])],
            ),
            mx.FieldSpec(
                key="frames",
                code_version="1",
                deps=[mx.FieldDep(feature=Video, fields=["frames"])],
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Cropped video segments."""


class FaceDetection(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="ray/face_detection",
        deps=[mx.FeatureDep(feature=Crop)],
        fields=[
            mx.FieldSpec(
                key="faces",
                code_version="1",
                deps=[mx.FieldDep(feature=Crop, fields=["frames"])],
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Face detection results from cropped video frames."""

    processing_time_seconds: float
    tokens_used: int
    error: str | None


# --8<-- [end:features]
