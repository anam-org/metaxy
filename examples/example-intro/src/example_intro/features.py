"""Feature definitions for a video/audio ML pipeline."""

# --8<-- [start:video_feature]
import metaxy as mx


class Video(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video",
        id_columns=["video_id"],
    ),
):
    """Raw video metadata. Root feature with no dependencies."""

    video_id: str
    path: str


# --8<-- [end:video_feature]


# --8<-- [start:face_detection_feature]
class FaceDetection(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/faces",
        deps=[Video],
        id_columns=["video_id"],
        fields=[
            mx.FieldSpec(key="faces", code_version="1"),
        ],
    ),
):
    """Face detection results extracted from video frames."""

    video_id: str
    num_faces: int
    face_embeddings_path: str


# --8<-- [end:face_detection_feature]


# --8<-- [start:audio_transcription_feature]
class AudioTranscription(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/transcription",
        deps=[Video],
        id_columns=["video_id"],
        fields=[
            mx.FieldSpec(key="transcript", code_version="1"),
        ],
    ),
):
    """Audio transcription extracted from the video's audio track."""

    video_id: str
    transcript: str
    language: str


# --8<-- [end:audio_transcription_feature]


# --8<-- [start:summary_feature]
class Summary(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/summary",
        deps=[AudioTranscription],
        id_columns=["video_id"],
        fields=[
            mx.FieldSpec(key="summary", code_version="1"),
        ],
    ),
):
    """Summary generated from the audio transcription."""

    video_id: str
    summary_text: str


# --8<-- [end:summary_feature]
