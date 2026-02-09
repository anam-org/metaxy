"""Feature definitions for quickstart example."""

# --8<-- [start:video_feature]
import metaxy as mx
from pydantic import Field


class Video(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video",
        id_columns=["id"],
    ),
):
    raw_video_path: str = Field(description="Path to the raw video file")
    id: str = Field(description="Unique identifier for the video")
    path: str = Field(description="Path to the processed video file")


# --8<-- [end:video_feature]


# --8<-- [start:audio_feature]
class Audio(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="audio",
        deps=[Video],
        id_columns=["id"],
    ),
):
    id: str = Field(description="Unique identifier for the audio")
    path: str = Field(description="Path to the audio file")


# --8<-- [end:audio_feature]
